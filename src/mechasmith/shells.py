"""Smooth shell primitives for the parts people look at (heads, bodies, covers)."""
from __future__ import annotations

import numpy as np
from build123d import Box, Pos, Sphere, fillet, scale


def ellipsoid(center, radii):
    return Pos(*center) * scale(Sphere(1.0), by=tuple(float(r) for r in radii))


def ellipsoid_shell(center, radii, wall: float):
    inner = [r - wall for r in radii]
    return ellipsoid(center, radii) - ellipsoid(center, inner)


def rounded_box(lo, hi, radius: float):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = Box(*(hi - lo))
    b = fillet(b.edges(), radius)
    return Pos(*((lo + hi) / 2)) * b


def rounded_box_cavity(lo, hi, radius: float, wall: float):
    """The empty inside of rounded_box_shell(...) with the same arguments."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return rounded_box(lo + wall, hi - wall, max(radius - wall, 0.5))


def rounded_box_shell(lo, hi, radius: float, wall: float):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return rounded_box(lo, hi, radius) - rounded_box(lo + wall, hi - wall, max(radius - wall, 0.5))


def arch(width: float, height: float, r_top: float, r_bottom: float = 2.0):
    """Face centred at the origin: a rectangle with large top corners and small bottom
    ones (the front outline of helmets, heads, arched windows)."""
    from build123d import RectangleRounded
    r_top = min(r_top, width / 2 - 0.5, height / 2 - 0.5)
    r_bottom = min(r_bottom, width / 2 - 0.5, height / 4 - 0.5)
    upper = RectangleRounded(width, height, r_top)
    lower = Pos(0, -height / 4) * RectangleRounded(width, height / 2, max(r_bottom, 0.5))
    return upper + lower


def loft_body(stations, bottom: float, radius: float = 12.0, bottom_radius: float | None = None):
    """Solid lofted along x through rounded-rectangle sections.

    stations: [(x, half_width, top), ...] in mm; every section spans z from `bottom`
    to `top`. Good for heads, noses and fairings that are wider and taller in places.
    With `bottom_radius`, sections are arches: `radius` on the top corners only.
    """
    from build123d import Plane, RectangleRounded, loft
    secs = []
    for x, hw, top in stations:
        h = top - bottom
        if bottom_radius is None:
            r = min(radius, hw - 0.5, h / 2 - 0.5)
            f = RectangleRounded(2 * hw, h, r)
        else:
            f = arch(2 * hw, h, radius, bottom_radius)
        secs.append(Plane.YZ.offset(x) * (Pos(0, (top + bottom) / 2) * f))
    return loft(secs)


def loft_cavity(stations, bottom: float, wall: float, radius: float = 12.0, open_bottom: bool = True,
                bottom_radius: float | None = None):
    """The empty inside of loft_shell(...) with the same arguments. Intersect a part with
    it (`plate & cavity`) to make the part reach exactly to the shell's inner walls."""
    inner_st = [(x, hw - wall, top - wall) for x, hw, top in stations]
    # pull the end sections inward so the front and back stay closed
    (x0, hw0, t0), (x1, hw1, t1) = inner_st[0], inner_st[-1]
    inner_st[0] = (x0 + wall, hw0, t0)
    inner_st[-1] = (x1 - wall, hw1, t1)
    return loft_body(inner_st, bottom - 50 if open_bottom else bottom + wall, max(radius - wall, 1.0),
                     None if bottom_radius is None else max(bottom_radius - wall, 0.5))


def loft_shell(stations, bottom: float, wall: float, radius: float = 12.0, open_bottom: bool = True,
               bottom_radius: float | None = None):
    """Hollow version of loft_body with walls of `wall` mm (open underneath by default)."""
    return (loft_body(stations, bottom, radius, bottom_radius)
            - loft_cavity(stations, bottom, wall, radius, open_bottom, bottom_radius))


def _surface_points(shapes, spacing: float):
    """Points on the surfaces of build123d `shapes` (mm), no further apart than `spacing`."""
    import trimesh
    pts = []
    for s in shapes:
        v, f = s.tessellate(0.2, 0.3)
        v = np.array([tuple(p) for p in v], float)
        f = np.asarray(f, int)
        if len(f):
            v, f = trimesh.remesh.subdivide_to_size(v, f, max_edge=spacing)
        pts.append(v)
    return np.concatenate(pts)


def enclose(shapes, axis, up=(0, 0, 1.0), *, wall: float = 2.0, offset: float | None = None,
            radius: float = 15.0, stations: int = 8, pad=(0.0, 0.0), ends: float = 0.6, taper: float = 0.6,
            clear=(), clip=None, sections=None, origin=None, contain: bool = True, web: bool | float = False,
            step: float = 3.0):
    """A smooth hollow cover around `shapes` (the cast or moulded shell of a torso, a thigh, a head).

    The cover runs along `axis` through rounded-rectangle sections, one every `step` mm,
    interpolated without overshoot through `stations` fitted sections. Each fitted section
    spans the shapes near it in the `up` direction and the direction across, grown by
    `offset` (default: the wall, so the inside of the cover rests on the outermost points of
    what it covers, which is where it is fixed). `pad` = (up, across) grows the sections
    further, for a fuller look. No section edge moves in or out faster than `taper` mm per
    mm along the axis (sections only grow to meet that), so the cover is smooth and still
    covers everything; the ends round off to `ends` of the end sections' size.

    `sections` replaces the fit (with `contain`, each grows to contain the shapes near it): [(t, up_lo, up_hi, across_lo, across_hi), ...] in mm along
    the axis from `origin` (default: the shapes' centre), across = axis x up. Measure them
    from a drawing or a reference mesh with `enclose_sections`; `shapes` then only matter
    for `clear` and the caller's own checks.
    `web` adds a rib (`wall` thick, or this many mm) inside the cover through its axis and
    `up`, cut round the covered parts and resting on them: what fixes a cover that is larger
    than its parts.

    Shapes in `clear` (solids or meshes: swept regions of the links that move past, from
    `sweep`, motors that turn inside) are cut out of the cover and `clip` (a solid) trims it. The booleans are exact mesh
    booleans; the result is a faceted solid.
    """
    import manifold3d as mf
    from scipy.interpolate import PchipInterpolator
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    e1 = np.asarray(up, float) - np.dot(up, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    off = wall if offset is None else offset
    secs = sections if sections is not None else enclose_sections(shapes, axis, up, stations=stations)
    centre = (np.asarray(origin, float) if origin is not None else
              np.asarray(secs[0][5]) if len(secs[0]) > 5 else _centre(shapes))
    ts = np.array([s_[0] for s_ in secs], float)
    lo = np.array([[s_[1], s_[3]] for s_ in secs], float) - np.asarray(pad, float)
    hi = np.array([[s_[2], s_[4]] for s_ in secs], float) + np.asarray(pad, float)
    if sections is not None and contain and shapes:
        # given sections grow wherever the shapes would poke through them
        p = _surface_points(shapes, 4.0) - centre
        t, u, v = p @ a, p @ e1, p @ e2
        gap = np.diff(ts).max() if len(ts) > 1 else 1.0
        for i, ti in enumerate(ts):
            m = np.abs(t - ti) <= gap
            if m.any():
                lo[i] = np.minimum(lo[i], [u[m].min(), v[m].min()])
                hi[i] = np.maximum(hi[i], [u[m].max(), v[m].max()])
        # parts beyond the end sections get end sections of their own
        for end, sel in ((t.min(), t < ts[0] - gap / 2), (t.max(), t > ts[-1] + gap / 2)):
            if sel.any():
                m = np.abs(t - end) <= gap
                row_lo, row_hi = [u[m].min(), v[m].min()], [u[m].max(), v[m].max()]
                k = 0 if end < ts[0] else len(ts)
                ts = np.insert(ts, k, end)
                lo = np.insert(lo, k, row_lo, axis=0)
                hi = np.insert(hi, k, row_hi, axis=0)
    if taper:
        d = np.abs(ts[:, None] - ts[None, :]) * taper
        lo = np.stack([(lo[None, :, k] + d).min(1) for k in range(2)], 1)
        hi = np.stack([(hi[None, :, k] - d).max(1) for k in range(2)], 1)
    dims = np.hstack([lo, hi])                       # up_lo, across_lo, up_hi, across_hi
    if len(ts) == 1:
        ts, dims = np.array([ts[0] - 0.5, ts[0] + 0.5]), np.vstack([dims, dims])
    f = PchipInterpolator(ts, dims, axis=0)

    def body(grow, r):
        n = max(int(np.ceil((ts[-1] - ts[0]) / step)), 2)
        tt = list(np.linspace(ts[0], ts[-1], n + 1))
        rows = [f(t) for t in tt]
        # rounded ends: a quarter ellipse `grow` (plus a little) beyond each end section
        cap = max(grow, 0.5) + 0.25 * min(rows[0][2] - rows[0][0], rows[0][3] - rows[0][1])
        caps = []
        for side, t0, row in ((-1, ts[0], rows[0]), (1, ts[-1], rows[-1])):
            ring = []
            for q in np.linspace(0, 1, 7)[1:]:
                k = ends + (1 - ends) * np.sqrt(1 - q ** 2)
                c = (row[:2] + row[2:]) / 2
                h = (row[2:] - row[:2]) / 2
                ring.append((t0 + side * q * cap, np.hstack([c - k * h, c + k * h])))
            caps.append(ring)
        stack = ([(t, rw) for t, rw in reversed(caps[0])] + list(zip(tt, rows)) + caps[1])
        rings = []
        for t, (u0, v0, u1, v1) in stack:
            u0, v0, u1, v1 = u0 - grow, v0 - grow, u1 + grow, v1 + grow
            rr = max(min(r, (u1 - u0) / 2 - 0.2, (v1 - v0) / 2 - 0.2), 0.2)
            rings.append(_rounded_ring(t, u0, u1, v0, v1, rr))
        rings = np.array(rings)                           # (rings, points, 3) in (t, u, v)
        P = rings.shape[1]
        verts = rings.reshape(-1, 3)
        tris = []
        for i in range(len(rings) - 1):
            for j in range(P):
                a0, a1 = i * P + j, i * P + (j + 1) % P
                b0, b1 = a0 + P, a1 + P
                tris += [(a0, b0, b1), (a0, b1, a1)]
        c0, c1 = len(verts), len(verts) + 1
        verts = np.vstack([verts, rings[0].mean(0), rings[-1].mean(0)])
        last = (len(rings) - 1) * P
        tris += [(c0, j, (j + 1) % P) for j in range(P)] + [(c1, last + (j + 1) % P, last + j) for j in range(P)]
        w = centre + verts[:, :1] * a + verts[:, 1:2] * e1 + verts[:, 2:3] * e2
        tris = np.asarray(tris, np.uint32)
        m = mf.Manifold(mf.Mesh(vert_properties=w.astype(np.float32), tri_verts=tris))
        if m.status() == mf.Error.NoError and m.volume() <= 0 or np.linalg.det(np.array([a, e1, e2])) < 0:
            m = mf.Manifold(mf.Mesh(vert_properties=w.astype(np.float32), tri_verts=tris[:, ::-1].copy()))
        if m.status() != mf.Error.NoError or m.volume() <= 0:
            raise ValueError(f"enclose: could not mesh the cover ({m.status()})")
        return m

    inner = body(off - wall, max(radius - wall, 0.5))
    cover = body(off, radius) - inner
    if web:
        # a rib inside the cover through its axis and `up`, ending on the parts it covers
        th = wall if web is True else float(web)
        n = np.cross(a, e1)
        R = np.column_stack([np.cross(e1, n), e1, n])         # slab frame: x along the axis, z normal
        big = 4000.0
        slab = mf.Manifold.cube((big, big, th), True).transform(np.hstack([R, centre[:, None]]))
        rib = inner ^ slab
        for s_ in shapes:
            rib = rib - _to_manifold(s_)
        cover = cover + rib
    for c in clear:
        cover = cover - _to_manifold(c)
    if clip is not None:
        cover = cover ^ _to_manifold(clip)
    solid = _to_solid(cover)
    mesh, bb = cover.to_mesh(), solid.bounding_box()
    solid._mesh = (np.asarray(mesh.vert_properties, np.float32)[:, :3].copy(), np.asarray(mesh.tri_verts, np.uint32),
                   tuple(bb.min), tuple(bb.max))
    return solid


def _rounded_ring(t, u0, u1, v0, v1, r, k: int = 8):
    """4k points round a rounded rectangle in the (u, v) plane at axial position t."""
    pts = []
    for cu, cv, a0 in ((u1 - r, v1 - r, 0.0), (u0 + r, v1 - r, 0.5 * np.pi), (u0 + r, v0 + r, np.pi),
                       (u1 - r, v0 + r, 1.5 * np.pi)):
        for q in np.linspace(a0, a0 + 0.5 * np.pi, k):
            pts.append((t, cu + r * np.cos(q), cv + r * np.sin(q)))
    return pts


def sweep(shapes, anchor, axis, range, step: float = 0.1):
    """The region `shapes` sweep as they turn about the line (anchor, axis) through `range`
    (radians), as a mesh to give `enclose(clear=...)`, so a cover stays out of the way of a
    link that moves past it. Pose the shapes as the link sits when the joint turns (e.g. the
    robot's home pose) with `turn`. Conservative: each shape counts as its convex hull, and
    the hull of two positions `step` apart covers the motion between them."""
    import manifold3d as mf
    hulls = [_to_manifold(s_).hull() for s_ in shapes]
    lo, hi = range
    ts = np.append(np.arange(lo, hi, step), hi)
    A = np.asarray(anchor, float)

    def at(m, t):
        R = _rotation(axis, t)
        return m.transform(np.hstack([R, (A - R @ A)[:, None]]))
    out = [mf.Manifold.batch_hull([at(h, t0), at(h, t1)]) for h in hulls for t0, t1 in zip(ts[:-1], ts[1:])]
    return mf.Manifold.batch_boolean(out, mf.OpType.Add)


def turn(shape, anchor, axis, angle: float):
    """`shape` (a solid or a mesh) turned by `angle` (radians, right-handed) about the line
    (anchor, axis)."""
    import manifold3d as mf
    from build123d import Axis
    if isinstance(shape, mf.Manifold):
        R, A = _rotation(axis, angle), np.asarray(anchor, float)
        return shape.transform(np.hstack([R, (A - R @ A)[:, None]]))
    return shape.rotate(Axis(tuple(float(x) for x in anchor), tuple(float(x) for x in axis)), float(np.degrees(angle)))


def _rotation(axis, t):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def as_mesh(shape):
    """`shape` as an exact manifold3d mesh (for `sweep`, `turn`, `enclose(clear=...)`); covers
    made by `enclose` keep theirs, so this costs nothing for them."""
    return _to_manifold(shape)


def _to_manifold(shape):
    import manifold3d as mf
    import trimesh
    if isinstance(shape, mf.Manifold):
        return shape
    cached = getattr(shape, "_mesh", None)
    if cached is not None:
        v, f, lo, hi = cached
        bb = shape.bounding_box()
        if np.allclose(lo, tuple(bb.min), atol=0.5) and np.allclose(hi, tuple(bb.max), atol=0.5):
            return mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=f))
    v, f = shape.tessellate(0.05, 0.15)
    tm = trimesh.Trimesh(np.array([tuple(p) for p in v], float), np.asarray(f, int), process=True)
    tm.merge_vertices()
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                               tri_verts=np.asarray(tm.faces, np.uint32)))


def _to_solid(m):
    """A build123d faceted solid from a manifold3d mesh: each closed surface becomes a solid,
    or a void in the smallest solid around it."""
    from build123d import Compound, Shell, Solid
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid, \
        BRepBuilderAPI_Sewing
    from OCP.gp import gp_Pnt
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties, float)[:, :3]
    sew = BRepBuilderAPI_Sewing(1e-4)
    tv = np.asarray(mesh.tri_verts)
    area = np.linalg.norm(np.cross(v[tv[:, 1]] - v[tv[:, 0]], v[tv[:, 2]] - v[tv[:, 0]]), axis=1)
    for tri in tv[area > 1e-6]:                        # slivers from the booleans: sewing closes them
        poly = BRepBuilderAPI_MakePolygon(*(gp_Pnt(*v[i]) for i in tri), True)
        if not poly.IsDone():
            continue
        face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
        if face.IsDone():
            sew.Add(face.Face())
    sew.Perform()
    shells = []
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    while ex.More():
        sh = TopoDS.Shell(ex.Current())
        alone = Solid(BRepBuilderAPI_MakeSolid(sh).Solid())
        bb = alone.bounding_box()
        shells.append((abs(alone.volume), sh, np.array([bb.min.X, bb.min.Y, bb.min.Z]),
                       np.array([bb.max.X, bb.max.Y, bb.max.Z])))
        ex.Next()
    shells.sort(key=lambda x: -x[0])
    outers = []                                        # [shell, lo, hi, voids]
    for vol, sh, lo, hi in shells:
        host = [o for o in outers if np.all(o[1] <= lo + 1e-6) and np.all(hi <= o[2] + 1e-6)]
        if host:
            min(host, key=lambda o: np.prod(o[2] - o[1]))[3].append(sh)
        else:
            outers.append([sh, lo, hi, []])
    solids = []
    expect = m.volume()
    for sh, _, _, voids in outers:
        mk = BRepBuilderAPI_MakeSolid(sh)
        for vsh in voids:
            mk.Add(vsh)
        fix = ShapeFix_Solid(mk.Solid())
        fix.Perform()
        fixed = fix.Shape()
        if fixed.ShapeType() == TopAbs_SOLID:
            solids.append(Solid(TopoDS.Solid(fixed)))
        else:
            ex = TopExp_Explorer(fixed, TopAbs_SOLID)
            while ex.More():
                solids.append(Solid(TopoDS.Solid(ex.Current())))
                ex.Next()
    out = solids[0] if len(solids) == 1 else Compound(solids)
    if abs(out.volume - expect) > 0.01 * expect:
        # OpenCascade could not close it into solids; keep the sewn surface for its extent. The
        # mesh (cached by the callers) carries the volume, mass properties and export.
        return Compound([Shell(sh) for _, sh, _, _ in shells])
    return out


def _centre(shapes):
    from build123d import Compound
    bb = Compound(list(shapes)).bounding_box()
    return np.array([(bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2, (bb.min.Z + bb.max.Z) / 2])


def enclose_sections(shapes, axis, up=(0, 0, 1.0), *, stations: int = 8, spacing: float = 4.0, origin=None,
                     span=None):
    """The sections `enclose` fits around `shapes` (solids, or an (N, 3) array of surface
    points, e.g. from a reference mesh): [(t, up_lo, up_hi, across_lo, across_hi, origin)] in
    mm, t along `axis` from `origin` (default: the shapes' centre), `across` = axis x up.
    Each spans everything within one station spacing of it, so the cover between stations
    still covers the shapes. `span` = (t0, t1) limits the stations to part of the length."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    e1 = np.asarray(up, float) - np.dot(up, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    pts = np.asarray(shapes, float) if isinstance(shapes, np.ndarray) else _surface_points(shapes, spacing)
    centre = np.asarray(origin, float) if origin is not None else (
        (pts.min(0) + pts.max(0)) / 2 if isinstance(shapes, np.ndarray) else _centre(shapes))
    p = pts - centre
    t, u, v = p @ a, p @ e1, p @ e2
    t0, t1 = span if span is not None else (t.min(), t.max())
    ts = np.linspace(t0, t1, max(stations, 2))
    step = ts[1] - ts[0]
    out = []
    for ti in ts:
        m = np.abs(t - ti) <= step
        if m.any():
            out.append((float(ti), float(u[m].min()), float(u[m].max()), float(v[m].min()), float(v[m].max()), centre))
    return out
