"""Design checks an agent can act on. Every check returns plain dicts (JSON-able).

Interference uses exact mesh booleans (manifold3d) on the exported visual
meshes, posed with MuJoCo forward kinematics, so it tests the real geometry
rather than collision proxies.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import manifold3d as mf
import mujoco
import numpy as np
import trimesh

TOL_M3 = 1e-9  # 1 mm^3: anything smaller is touching faces / tessellation noise


class PosedGeometry:
    def __init__(self, scene_xml: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        mesh_dir = Path(scene_xml).parent / "meshes"
        self.parts = []  # (part name, body id, Manifold in body frame, body-frame vertices)
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
                continue
            mesh_id = m.geom_dataid[g]
            name = m.mesh(mesh_id).name
            tm = trimesh.load(mesh_dir / f"{name}.stl", process=True)
            v = np.asarray(tm.vertices) * 1e-3
            man = mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32),
                                      tri_verts=np.asarray(tm.faces, np.uint32)))
            self.parts.append((name, int(m.geom_bodyid[g]), man, v))
        self.joint_names = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        # Pins may overlap the parts they join: hinge joints with a pin radius (from the manifest)
        # excuse overlap inside that radius for body pairs one or two joints apart.
        import json
        man = Path(scene_xml).parent / "manifest.json"
        self.pins = {}
        if man.exists():
            for b in json.loads(man.read_text())["bodies"].values():
                jd = b.get("joint")
                if jd and jd.get("pin_mm"):
                    self.pins[jd["name"]] = jd["pin_mm"] * 1e-3

    def set_pose(self, q: dict[str, float]):
        from .sim import solve_pose
        m, d = self.model, self.data
        qpos = solve_pose(m, d, q)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)

    def _posed(self, idx):
        name, b, man, v = self.parts[idx]
        R, p = self.data.xmat[b].reshape(3, 3), self.data.xpos[b]
        T = np.hstack([R, p[:, None]])
        aabb_v = v @ R.T + p
        return man.transform(T), aabb_v.min(0), aabb_v.max(0)

    def _pins_between(self, b1: int, b2: int):
        """Pinned joints on the tree path between two bodies, if they are at most two joints apart."""
        m = self.model
        def up(b):
            return [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        for lo, hi in ((b1, b2), (b2, b1)):
            path, b = [], lo
            for _ in range(2):
                path += up(b)
                b = m.body_parentid[b]
                if b == hi:
                    names = [m.joint(j).name for j in path]
                    return [(n, self.pins[n]) for n in names if n in self.pins]
        return []

    def _pin_solid(self, joint_name: str, r: float):
        m, d = self.model, self.data
        j = m.joint(joint_name).id
        p, a = d.xanchor[j], d.xaxis[j]
        cyl = mf.Manifold.cylinder(1.0, r, r, 64, True)              # along z, centred, length 1 m
        z = a / np.linalg.norm(a)
        x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z, x), z])
        return cyl.transform(np.hstack([R, p[:, None]]))

    def collisions(self) -> list[dict]:
        posed = [self._posed(i) for i in range(len(self.parts))]
        hits = []
        for i, j in itertools.combinations(range(len(self.parts)), 2):
            if self.parts[i][1] == self.parts[j][1]:
                continue  # same rigid body
            (a, alo, ahi), (b, blo, bhi) = posed[i], posed[j]
            if np.any(alo > bhi) or np.any(blo > ahi):
                continue
            inter = a ^ b
            for jn, r in self._pins_between(self.parts[i][1], self.parts[j][1]):
                if inter.volume() <= TOL_M3:
                    break
                inter = inter - self._pin_solid(jn, r)
            vol = inter.volume()
            if vol > TOL_M3:
                hits.append({"a": self.parts[i][0], "b": self.parts[j][0],
                             "overlap_mm3": round(vol * 1e9, 1)})
        return sorted(hits, key=lambda h: -h["overlap_mm3"])


def interference_report(scene_xml: Path, steps: int = 24, base_pose: dict | None = None) -> dict:
    """Home-pose interference, then a sweep of each joint over its declared range
    (others held at the base pose). Reports the usable collision-free range. In a model
    with loops or couplings only the driven joints are swept; the rest follow."""
    from .sim import driven_joints
    pg = PosedGeometry(scene_xml)
    pg.joint_names = driven_joints(pg.model)
    m = pg.model
    base = dict(base_pose or {})
    pg.set_pose(base)
    report = {"home_collisions": pg.collisions(), "joints": {}}

    for jn in pg.joint_names:
        lo, hi = m.jnt_range[m.joint(jn).id]
        q0 = float(np.clip(base.get(jn, 0.0), lo, hi))
        usable = [q0, q0]
        blocked = {}
        for direction, end in ((+1, hi), (-1, lo)):
            for q in np.linspace(q0, end, steps // 2 + 1)[1:]:
                pg.set_pose({**base, jn: q})
                hits = [h for h in pg.collisions() if h not in report["home_collisions"]]
                if hits:
                    blocked["upper" if direction > 0 else "lower"] = {"at_rad": round(float(q), 3), "first_hit": hits[0]}
                    break
                usable[0 if direction < 0 else 1] = float(q)
        report["joints"][jn] = {
            "declared": [round(float(lo), 3), round(float(hi), 3)],
            "usable": [round(usable[0], 3), round(usable[1], 3)],
            "blocked": blocked,
        }
    return report


def mass_report(manifest: dict) -> dict:
    bodies = {k: round(v["mass"] * 1000, 1) for k, v in manifest["bodies"].items()}
    return {"total_g": round(manifest["total_mass"] * 1000, 1), "bodies_g": bodies}
