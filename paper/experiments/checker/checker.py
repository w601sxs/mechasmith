"""Independent checker for robot models (paper Q2): scores any MJCF the same way.

    python checker.py <model.xml> [--reference ref.xml] [--map ours=theirs ...] [--stand] [--pd KP,KD]

It uses MuJoCo to compile the model and trimesh/manifold3d for exact mesh geometry, and nothing
from mechasmith, so models written by hand, by an agent in raw MJCF/URDF, by code CAD or by
mechasmith are judged by identical code. Output is one JSON object:

  loads              the model compiles
  mass_kg            total mass
  floating_parts     geoms not connected to the root body (see "Connected" below)
  detached_bodies    bodies not connected to their parent body
  split_parts        geoms whose mesh is in several disconnected pieces (a part drawn apart from
                     itself, which the connection tests above cannot see inside one geom)
  overlaps           pairs of geoms in different bodies whose intersection exceeds --overlap mm^3
  joints             with --reference: per joint (matched by name, or --map), the axis error and
                     axis-line distance as defined by Le et al. (Articulate-Anything, ICLR 2025):
                     axis error = min(angle, pi - angle); origin error = distance between the two
                     axis lines (revolute) or anchors (prismatic); success = both under 0.25 rad
                     and 50 mm. Both models are taken at qpos0, in the frame of their root body.
  stand              with --stand, floating-base models: holds the home keyframe (or qpos0) for 3 s
                     on its actuators and reports whether it stays up (tilt under 30 deg, root height
                     over 70% of the start). Position actuators track the pose; torque actuators
                     need --pd KP,KD.

Connected: two geoms are connected if their surfaces are within --touch mm (same body, or a body
and its parent), or if they belong to a body and its parent and the axis of a hinge joint between
those bodies passes within --pin mm of both (a pin through a fork or a bored hole, which models
leave out). The pin rule is the one mechasmith's own `hinge` enforces; any model gets it.

Shapes: when a model has visual geoms (contype = conaffinity = 0), those are the parts; otherwise
all geoms are. Meshes are used as triangle meshes, not MuJoCo's convex hulls, so nested or
concave parts are judged correctly.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict

import mujoco
import numpy as np
import trimesh

FREE = mujoco.mjtJoint.mjJNT_FREE
SLIDE = mujoco.mjtJoint.mjJNT_SLIDE


# ---------------------------------------------------------------- geometry

def _geom_mesh(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> trimesh.Trimesh | None:
    """The geom as a triangle mesh in world coordinates (metres), or None for planes and heightfields."""
    t, size = m.geom_type[g], m.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_MESH:
        k = m.geom_dataid[g]
        va, vn, fa, fn = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
        mesh = trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), process=True)
    elif t == G.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif t == G.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=size[0])
    elif t == G.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=48)
    elif t == G.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[24, 24])
    elif t == G.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    else:
        return None
    T = np.eye(4)
    T[:3, :3] = d.geom_xmat[g].reshape(3, 3)
    T[:3, 3] = d.geom_xpos[g]
    mesh.apply_transform(T)
    return mesh


def _distance(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Smallest surface-to-surface distance (0 if they touch or overlap), from vertices and samples."""
    def one_way(src, dst):
        pts = src.vertices
        if len(pts) < 2000:
            pts = np.vstack([pts, src.sample(2000 - len(pts))])
        _, dist, _ = trimesh.proximity.closest_point(dst, pts)
        inside = dst.contains(pts) if dst.is_watertight else np.zeros(len(pts), bool)
        return 0.0 if inside.any() else float(dist.min())
    return min(one_way(a, b), one_way(b, a))


def _pieces(mesh: trimesh.Trimesh, tol: float) -> int:
    """Number of pieces of a triangle mesh more than `tol` (m) apart. Edge-connected patches come
    first; CAD tessellations do not always share vertices along seams, so patches that touch
    (within tol) are merged."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(mesh.faces)
    if n == 0:
        return 0
    adj = mesh.face_adjacency
    g = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    k, label = connected_components(g, directed=False)
    if k == 1:
        return 1
    patches = [mesh.submesh([np.flatnonzero(label == i)], append=True) for i in range(k)]
    parent = list(range(k))
    find = lambda i: i if parent[i] == i else find(parent[i])
    for i, j in itertools.combinations(range(k), 2):
        if find(i) != find(j) and _aabb_gap(patches[i], patches[j]) <= tol:
            _, d, _ = trimesh.proximity.closest_point(patches[j], patches[i].vertices)
            if d.min() <= tol:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(k)})


def _overlap_mm3(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    if not (a.is_watertight and b.is_watertight):
        return 0.0
    try:
        inter = trimesh.boolean.intersection([a, b], engine="manifold")
    except Exception:
        return 0.0
    return float(inter.volume) * 1e9 if inter is not None and len(inter.faces) else 0.0


def _aabb_gap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    return float(np.linalg.norm(np.clip(lo - hi, 0, None)))


def _part_geoms(m: mujoco.MjModel) -> list[int]:
    visual = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
              and m.geom_bodyid[g] != 0]
    parts = visual or [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    return [g for g in parts if m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)]


def _line_distance(mesh: trimesh.Trimesh, anchor, axis, half_length=0.2, step=0.0005) -> float:
    """Smallest distance from the axis line (a segment of +-half_length m) to the mesh surface."""
    s = np.arange(-half_length, half_length + step, step)
    pts = np.asarray(anchor)[None, :] + s[:, None] * np.asarray(axis)[None, :]
    lo, hi = mesh.bounds[0] - 0.01, mesh.bounds[1] + 0.01
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if not len(pts):
        return np.inf
    if mesh.is_watertight and mesh.contains(pts).any():
        return 0.0
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    return float(dist.min())


def geometry_checks(m: mujoco.MjModel, d: mujoco.MjData, touch_mm: float, overlap_mm3: float,
                    pin_mm: float = 3.0) -> dict:
    geoms = _part_geoms(m)
    meshes = {g: _geom_mesh(m, d, g) for g in geoms}
    geoms = [g for g in geoms if meshes[g] is not None]
    body = {g: int(m.geom_bodyid[g]) for g in geoms}
    parent = lambda b: int(m.body_parentid[b])
    name = lambda g: m.geom(g).name or f"geom{g}"
    bname = lambda b: m.body(b).name or f"body{b}"
    tol = touch_mm / 1000.0

    # Touching pairs among geoms of the same body or of a body and its parent.
    touching, overlaps = set(), []
    for g, h in itertools.combinations(geoms, 2):
        bg, bh = body[g], body[h]
        near = _aabb_gap(meshes[g], meshes[h]) <= tol
        if not near:
            continue
        if bg != bh:   # overlap between parts of different bodies
            v = _overlap_mm3(meshes[g], meshes[h])
            if v > overlap_mm3:
                overlaps.append({"a": name(g), "b": name(h), "bodies": [bname(bg), bname(bh)], "overlap_mm3": round(v, 1)})
        if (bg == bh or parent(bg) == bh or parent(bh) == bg) and _distance(meshes[g], meshes[h]) <= tol:
            touching.add((g, h))

    # Pins: a hinge between a body and its parent connects them if its axis passes through both.
    pinned = set()
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = int(m.jnt_bodyid[j])
        mine = [g for g in geoms if body[g] == b]
        theirs = [g for g in geoms if body[g] == parent(b)]
        near = lambda gs: [g for g in gs if _line_distance(meshes[g], d.xanchor[j], d.xaxis[j]) <= pin_mm / 1000.0]
        a_side, b_side = near(mine), near(theirs)
        if a_side and b_side:
            pinned.add((min(a_side[0], b_side[0]), max(a_side[0], b_side[0])))
    touching |= pinned

    # Connectivity from the root body's geoms.
    adj = defaultdict(set)
    for g, h in touching:
        adj[g].add(h)
        adj[h].add(g)
    roots = [g for g in geoms if parent(body[g]) == 0]
    seen, stack = set(roots), list(roots)
    while stack:
        for h in adj[stack.pop()]:
            if h not in seen:
                seen.add(h)
                stack.append(h)
    floating = [{"geom": name(g), "body": bname(body[g])} for g in geoms if g not in seen]

    detached = []
    for b in sorted(set(body.values())):
        if parent(b) == 0:
            continue
        mine = [g for g in geoms if body[g] == b]
        theirs = [g for g in geoms if body[g] == parent(b)]
        if theirs and not any((min(g, h), max(g, h)) in touching for g in mine for h in theirs):
            detached.append(bname(b))
    split = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            n = _pieces(meshes[g], tol)
            if n > 1:
                split.append({"geom": name(g), "pieces": n})
    return {"parts": len(geoms), "pins": len(pinned), "split_parts": split, "floating_parts": floating, "detached_bodies": detached,
            "overlaps": sorted(overlaps, key=lambda o: -o["overlap_mm3"])}


# ---------------------------------------------------------------- joints

def _root_body(m: mujoco.MjModel) -> int:
    return next(b for b in range(1, m.nbody) if m.body_parentid[b] == 0)


def joint_lines(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Per non-free joint: (type, axis, anchor) in the root body's frame, at the current pose."""
    r = _root_body(m)
    R, p = d.xmat[r].reshape(3, 3), d.xpos[r]
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == FREE:
            continue
        out[m.joint(j).name] = (int(m.jnt_type[j]), R.T @ d.xaxis[j], R.T @ (d.xanchor[j] - p))
    return out


def joint_errors(ours: dict, ref: dict, mapping: dict) -> dict:
    res, ok = {}, []
    for jo, (to, ao, po) in ours.items():
        jr = mapping.get(jo, jo)
        if jr not in ref:
            continue
        tr, ar, pr = ref[jr]
        cos = float(np.clip(np.dot(ao, ar), -1, 1))
        ang = float(np.arccos(cos))
        axis_err = min(ang, np.pi - ang)
        if to == SLIDE or tr == SLIDE:
            origin = float(np.linalg.norm(po - pr))
        else:   # distance between the two axis lines
            n = np.cross(ao, ar)
            w = pr - po
            origin = float(abs(np.dot(w, n)) / np.linalg.norm(n)) if np.linalg.norm(n) > 1e-6 \
                else float(np.linalg.norm(np.cross(w, ao)))
        success = axis_err < 0.25 and origin < 0.05
        ok.append(success)
        res[jo] = {"reference": jr, "axis_err_rad": round(axis_err, 4), "same_sign": cos > 0,
                   "origin_err_mm": round(origin * 1000, 2), "type_match": to == tr, "success": success}
    mapped = {mapping.get(o, o) for o in ours}
    missing = [j for j in ref if j not in mapped]
    # Joints the reference has and we lack count as failures.
    rate = float(np.sum(ok) / (len(ok) + len(missing))) if ok or missing else None
    return {"matched": len(res), "missing_in_ours": missing, "success_rate": rate, "per_joint": res}


# ---------------------------------------------------------------- stand test

def stand_test(xml: str, pd: tuple[float, float] | None, seconds: float = 3.0) -> dict:
    spec = mujoco.MjSpec.from_file(xml)
    m0 = spec.compile()
    if not any(m0.jnt_type == FREE):
        return {"skipped": "fixed base"}
    joint_acts = [a for a in range(m0.nu) if m0.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
    if not joint_acts:
        return {"skipped": "no joint actuators (e.g. a rotorcraft)"}
    torque = [a for a in joint_acts if m0.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE]
    if torque and pd is None:
        return {"skipped": "torque actuators need --pd KP,KD"}
    d0 = mujoco.MjData(m0)
    key = 0 if m0.nkey else None
    if key is not None:
        mujoco.mj_resetDataKeyframe(m0, d0, key)
    mujoco.mj_forward(m0, d0)
    # Floor just under the lowest collision geometry at the start pose.
    colliders = [_geom_mesh(m0, d0, g) for g in range(m0.ngeom) if m0.geom_contype[g] or m0.geom_conaffinity[g]]
    low = min(c.bounds[0][2] for c in colliders if c is not None)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], pos=[0, 0, low - 1e-4],
                            contype=1, conaffinity=1)
    m = spec.compile()
    d = mujoco.MjData(m)
    if key is not None:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    q_home = d.qpos.copy()
    r = _root_body(m)
    z0 = float(d.xpos[r][2])
    for _ in range(int(seconds / m.opt.timestep)):
        for a in joint_acts:
            j = m.actuator_trnid[a][0]
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            if a in torque:
                kp, kd = pd
                d.ctrl[a] = kp * (q_home[qa] - d.qpos[qa]) - kd * d.qvel[va]
            else:
                d.ctrl[a] = q_home[qa]
        mujoco.mj_step(m, d)
    up = d.xmat[r].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    z = float(d.xpos[r][2])
    return {"stands": tilt < 30 and z > 0.7 * z0, "tilt_deg": round(tilt, 1),
            "height_ratio": round(z / z0, 3) if z0 > 0 else None, "seconds": seconds}


# ---------------------------------------------------------------- main

def check(xml: str, reference: str | None = None, mapping: dict | None = None, touch_mm: float = 0.2,
          overlap_mm3: float = 1.0, stand: bool = False, pd=None, pin_mm: float = 3.0) -> dict:
    out = {"model": xml}
    try:
        m = mujoco.MjModel.from_xml_path(xml)
    except Exception as e:   # the first thing an agent's model can get wrong
        return {**out, "loads": False, "error": str(e)[:500]}
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out["loads"] = True
    out["mass_kg"] = round(float(m.body_subtreemass[0]), 4)
    out.update(geometry_checks(m, d, touch_mm, overlap_mm3, pin_mm))
    if reference:
        mr = mujoco.MjModel.from_xml_path(reference)
        dr = mujoco.MjData(mr)
        mujoco.mj_forward(mr, dr)
        out["reference_mass_kg"] = round(float(mr.body_subtreemass[0]), 4)
        out["joints"] = joint_errors(joint_lines(m, d), joint_lines(mr, dr), mapping or {})
    if stand:
        out["stand"] = stand_test(xml, pd)
    out["valid"] = (out["loads"] and not out["floating_parts"] and not out["detached_bodies"]
                    and not out["overlaps"] and not out["split_parts"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--reference")
    ap.add_argument("--map", action="append", default=[], help="joint name pairs ours=theirs")
    ap.add_argument("--touch", type=float, default=0.2, help="mm within which two parts count as touching")
    ap.add_argument("--overlap", type=float, default=1.0, help="mm^3 of intersection that counts as an overlap")
    ap.add_argument("--pin", type=float, default=3.0, help="mm within which a hinge axis must pass through both parts")
    ap.add_argument("--stand", action="store_true")
    ap.add_argument("--pd", help="KP,KD for torque actuators in the stand test")
    a = ap.parse_args()
    mapping = dict(x.split("=", 1) for x in a.map)
    pd = tuple(float(v) for v in a.pd.split(",")) if a.pd else None
    print(json.dumps(check(a.xml, a.reference, mapping, a.touch, a.overlap, a.stand, pd, a.pin), indent=2))


if __name__ == "__main__":
    main()
