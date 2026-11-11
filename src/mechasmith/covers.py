"""Covers that stay out of the way of everything that moves past them.

A part marked `clear_motion=True` (a cover from `enclose`) is finished at export, after
mirroring: for every joint, the parts on the other side of it are posed at the robot's home
pose and swept through the joint's whole range relative to the cover, and that region is cut
out of the cover. This is the motion the joint sweep check tests (one joint at a time, the
others at home), so covers made this way leave every joint the range it has without them.

Parents win: a cover on the parent side of a joint ignores covers on the child side, and a
child-side cover clears the parent side's finished covers, so two covers never both give way.
Each moving part is cut into `CHUNK_MM` cubes and each piece counts as its convex hull grown
by `CLEAR_MM` (so a hollow part does not fill in), and the hull of two positions `STEP_RAD`
apart covers the motion between them, so the cut is conservative by at most a chunk's hull.
"""
from __future__ import annotations

import numpy as np

CHUNK_MM = 30.0
STEP_RAD = 0.1
CLEAR_MM = 1.5       # covers stay at least about this far from what moves past them
_DIRS = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)], float) - 1
_DIRS /= np.linalg.norm(_DIRS, axis=1)[:, None]


def _grown_hull(piece, by: float):
    """The convex hull of a piece grown outward by about `by` mm (26 directions)."""
    import manifold3d as mf
    v = np.asarray(piece.hull().to_mesh().vert_properties, float)[:, :3]
    pts = (v[:, None, :] + by * _DIRS[None]).reshape(-1, 3)
    return mf.Manifold.hull_points(pts.tolist())


def _rot(axis, t):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _about(anchor, axis, t):
    T = np.eye(4)
    R = _rot(axis, t)
    A = np.asarray(anchor, float)
    T[:3, :3], T[:3, 3] = R, A - R @ A
    return T


def home_transforms(robot, home: dict | None) -> dict:
    """{link name: 4x4 world transform at the home pose, from the zero pose}."""
    home = home or {}
    T = {robot.root.name: np.eye(4)}
    todo = list(robot.joints)
    while todo:
        j = next((j for j in todo if j.parent.name in T), None)
        if j is None:
            break
        todo.remove(j)
        T[j.child.name] = T[j.parent.name] @ _about(j.anchor, j.axis, home.get(j.name, 0.0))
    for name in robot.links:
        T.setdefault(name, np.eye(4))
    return T


def _subtree(robot, link) -> set:
    out, stack = set(), [link.name]
    while stack:
        n = stack.pop()
        out.add(n)
        stack += [j.child.name for j in robot.joints if j.parent.name == n]
    return out


def _pieces(mesh):
    """Convex pieces of a mesh: its intersection with each CHUNK_MM cube, hulled."""
    import manifold3d as mf
    bb = mesh.bounding_box()
    lo, hi = np.asarray(bb[:3], float), np.asarray(bb[3:], float)
    n = np.maximum(np.ceil((hi - lo) / CHUNK_MM), 1).astype(int)
    if np.prod(n) == 1:
        return [_grown_hull(mesh, CLEAR_MM)]
    out = []
    size = (hi - lo) / n
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = lo + size * (np.array([i, j, k]) + 0.5)
                cube = mf.Manifold.cube(tuple(size + 1e-3), True).translate(tuple(c))
                piece = mesh ^ cube
                if not piece.is_empty():
                    out.append(_grown_hull(piece, CLEAR_MM))
    return out


def _cylindrical(pts, anchor, axis):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    p = pts - anchor
    ax = p @ a
    r = np.hypot(p @ e1, p @ e2)
    th = np.arctan2(p @ e2, p @ e1)
    return r, ax, th


def _may_reach(piece_pts, cover_cyl, anchor, axis, rng, margin=4.0) -> bool:
    """Whether a piece swept through rng about the axis can come near any cover vertex."""
    r, ax, th = _cylindrical(piece_pts, anchor, axis)
    cr, cax, cth = cover_cyl
    m = (cr >= r.min() - margin) & (cr <= r.max() + margin) & (cax >= ax.min() - margin) & (cax <= ax.max() + margin)
    if not m.any():
        return False
    span = rng[1] - rng[0]
    if r.min() < margin or span >= 2 * np.pi:
        return True
    # the piece's angles, unwrapped round their mean, swept through rng
    mean = np.angle(np.exp(1j * th).mean())
    rel = np.angle(np.exp(1j * (th - mean)))
    if rel.max() - rel.min() > np.pi:
        return True
    lo = mean + rel.min() + rng[0] - margin / max(r.min(), 1.0)
    hi = mean + rel.max() + rng[1] + margin / max(r.min(), 1.0)
    c = np.angle(np.exp(1j * (cth[m] - lo)))
    c = np.where(c < 0, c + 2 * np.pi, c)
    return bool((c <= hi - lo).any())


def _samples(mesh, spacing: float = 4.0):
    """Points on a mesh's surface no further apart than `spacing` (long flat faces included)."""
    import trimesh
    m = mesh.to_mesh()
    v, f = trimesh.remesh.subdivide_to_size(np.asarray(m.vert_properties, float)[:, :3],
                                            np.asarray(m.tri_verts, int), max_edge=spacing)
    return v


def resolve_covers(robot, home: dict | None = None) -> list[str]:
    """Cut each `clear_motion` part free of everything that moves past it (see the module
    docstring). Returns notes on what was cut; runs once."""
    import manifold3d as mf

    from .shells import _to_manifold
    if getattr(robot, "_covers_resolved", False):
        return []
    robot._covers_resolved = True
    covers = [(l, p) for l in robot.links.values() for p in l.parts if getattr(p, "clear_motion", False)]
    if not covers:
        return []
    T = home_transforms(robot, home)
    subs = {j.name: _subtree(robot, j.child) for j in robot.joints}
    depth = {robot.root.name: 0}
    for _ in robot.joints:
        for j in robot.joints:
            if j.parent.name in depth:
                depth[j.child.name] = depth[j.parent.name] + 1
    covers.sort(key=lambda lp: depth.get(lp[0].name, 0))
    meshes, pieces = {}, {}

    def mesh_of(part):
        if part.name not in meshes:
            meshes[part.name] = _to_manifold(part.shape)
        return meshes[part.name]

    def pieces_of(part):
        if part.name not in pieces:
            pieces[part.name] = _pieces(mesh_of(part))
        return pieces[part.name]

    done, notes = set(), []
    for L, P in covers:
        cover = mesh_of(P)
        cv = _samples(cover)
        TL_inv = np.linalg.inv(T[L.name])
        cuts = []
        for j in robot.joints:
            child_side = L.name in subs[j.name]
            others = [l for n, l in robot.links.items() if (n not in subs[j.name]) == child_side]
            q = (home or {}).get(j.name, 0.0)
            lo, hi = j.range
            rng = (-(hi - q), -(lo - q)) if child_side else (lo - q, hi - q)
            Tp = TL_inv @ T[j.parent.name]
            anchor = Tp[:3, :3] @ j.anchor + Tp[:3, 3]
            axis = Tp[:3, :3] @ j.axis
            cyl = _cylindrical(cv, anchor, axis)
            ts = np.append(np.arange(rng[0], rng[1], STEP_RAD), rng[1])
            hulls = []
            for o in others:
                Rel = TL_inv @ T[o.name]
                for Q in o.parts:
                    if Q is P:
                        continue
                    if getattr(Q, "clear_motion", False) and not child_side:
                        continue                            # child-side covers give way to this one
                    for pc in pieces_of(Q):
                        pc = pc.transform(Rel[:3])
                        pts = np.asarray(pc.to_mesh().vert_properties, float)[:, :3]
                        if not _may_reach(pts, cyl, anchor, axis, rng):
                            continue
                        hulls += [mf.Manifold.batch_hull([pc.transform(_about(anchor, axis, t0)[:3]),
                                                          pc.transform(_about(anchor, axis, t1)[:3])])
                                  for t0, t1 in zip(ts[:-1], ts[1:])]
            if hulls:
                cut = mf.Manifold.batch_boolean(hulls, mf.OpType.Add)
                before = cover.volume()
                cover = cover - cut
                if before - cover.volume() > 1.0:
                    cuts.append(f"{j.name} {before - cover.volume():.0f} mm3")
        if cuts:
            # From here on only the mesh is used (mass properties, export, checks); the shape is
            # a box of the same extent that carries it, for the primitive fit and the floor height.
            from .geom import box
            mesh = cover.to_mesh()
            v = np.asarray(mesh.vert_properties, np.float32)[:, :3].copy()
            lo, hi = v.min(0).astype(float), v.max(0).astype(float)
            P.shape = box(tuple(lo), tuple(hi))
            P.shape._mesh = (v, np.asarray(mesh.tri_verts, np.uint32), tuple(lo), tuple(hi))
            meshes[P.name] = cover
            pieces.pop(P.name, None)
            notes.append(f"{P.name}: cut clear of " + ", ".join(cuts))
        done.add(P.name)
    return notes
