"""Printed links: the body that turns on one servo and carries the next.

Almost every printed arm or leg part is the same thing: a solid that spans the full
width of the servo it pivots on (cheeks on the horn and on the back), hollowed wherever
that servo's case sweeps as the joint turns (that is what leaves the fork), plus a
pocket with walls around the servo it carries. Say that, and mechasmith draws it:

    asm.joint("shoulder_lift", lift, (-3.3, 0.17))
    upper = asm.link("upper_arm", around=lift, path=[lift.anchor, elbow_at], width=24,
                     holds=Hold(elbow, at=elbow_at, axis=BACK, body=UP))
    asm.joint("elbow_flex", elbow, (-0.17, 3.14))

* The clearance comes from the joint range you declared, so the fork clears the case
  over exactly that range (and the walls of a pocket the case sits in, if any).
* The horn cheek is flush on the horn and the held servo is seated in its pocket, so
  both contacts hold by construction. A link that comes out in pieces is an error.
* `path` is a centre line in the joint's plane, thickened to `width` with round ends.
  `add` / `cut` take extra world-space shapes for everything else (fingers, feet, holes).

Root parts (a base that holds the first servo) use `printed(...)` directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import unit
from .materials import PLA, Material

SWEEP_STEP = np.radians(5.0)


@dataclass
class Hold:
    """Where a link carries a servo: its anchor (point on the output axis, case centre),
    the world direction its horn faces, and the direction from the axis to the long end.

    By default the servo sits in a closed cup (a hole for the horn). `band=` makes it a
    ring instead: walls round the case's sides only, `band` mm wide along the axis and open
    on both axial faces (how most printed motor holders look). `wrap=` covers only that many
    mm of the case, measured from its long end (the rest of the servo hangs out)."""
    unit: object
    at: tuple
    axis: tuple
    body: tuple
    band: float | None = None
    wrap: float | None = None


def _plane_basis(a):
    a = unit(a)
    u = unit(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]))
    return u, np.cross(a, u), a


def _servo_frame(at, axis, body):
    """World rotation (columns = servo local X, Y, Z) and origin for a servo pose."""
    x, z = unit(axis), -unit(body)
    if abs(np.dot(x, z)) > 1e-6:
        raise ValueError("Hold: axis and body must be perpendicular")
    return np.column_stack([x, np.cross(z, x), z]), np.asarray(at, float)


def _local_box(R, o, lo, hi):
    """Box given in a servo's local frame, placed in the world."""
    from build123d import Box, Location, Plane, Pos
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = Pos(*((lo + hi) / 2)) * Box(*(hi - lo))
    return b.moved(Location(Plane(origin=tuple(o), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2]))))


def _face_from_ring(ring_uv, origin, u, v, a):
    from build123d import Plane, Polygon
    pl = Plane(origin=tuple(origin), x_dir=tuple(u), z_dir=tuple(a))
    return pl * Polygon(*[tuple(map(float, p)) for p in ring_uv], align=None)


def _prism(rings_uv, origin, u, v, a, s0, s1):
    """Extrude 2D rings (plane through `origin`, basis u, v) along a from s0 to s1."""
    from build123d import extrude
    solid = None
    for ring in rings_uv:
        f = _face_from_ring(ring, np.asarray(origin) + a * s0, u, v, a)
        piece = extrude(f, amount=s1 - s0, dir=tuple(a))
        solid = piece if solid is None else solid + piece
    return solid


def _swept_rings(rect_uv, angles):
    """Union of a polygon rotated through `angles` (about the 2D origin), as rings."""
    from manifold3d import CrossSection
    pts = np.asarray(rect_uv, float)

    def rot(t):
        c, s = np.cos(t), np.sin(t)
        return pts @ np.array([[c, s], [-s, c]])

    hulls = [CrossSection.hull_points(np.vstack([rot(t0), rot(t1)]).tolist())
             for t0, t1 in zip(angles[:-1], angles[1:])] or [CrossSection.hull_points(rot(angles[0]).tolist())]
    cs = CrossSection.batch_boolean(hulls, __import__("manifold3d").OpType.Add) if len(hulls) > 1 else hulls[0]
    return [np.asarray(r) for r in cs.to_polygons()]


def _holder(hold: Hold, wall: float):
    """(outer solid, cavity solids) for a pocket holding hold.unit, plus its back-face frame."""
    from build123d import Align, Cylinder, Location, Plane, Pos, Rot
    s = hold.unit.spec
    R, o = _servo_frame(hold.at, hold.axis, hold.body)
    hx, hy = s.case_half_axial, s.case_width / 2
    fw = min(wall, s.horn_thickness - 0.5)
    z0 = -s.case_below - wall
    z1 = s.case_above + wall if hold.wrap is None else min(s.case_above + wall, -s.case_below + hold.wrap)
    if hold.band is None:
        x0, x1, cx0, cx1 = -hx - wall, hx + fw, -hx, hx
    else:
        b = min(hold.band, 2 * hx)
        x0, x1, cx0, cx1 = -b / 2, b / 2, -hx - 1, hx + 1
    outer = _local_box(R, o, (x0, -hy - wall, z0), (x1, hy + wall, z1))
    try:
        from build123d import fillet
        outer = fillet(outer.edges(), min(2.0, wall * 0.9))
    except Exception:
        pass
    case = _local_box(R, o, (cx0, -hy, -s.case_below), (cx1, hy, s.case_above + (0 if hold.wrap is None else 1)))
    hole = (Pos(hx - 1, 0, 0) * Rot(0, 90, 0) * Cylinder(s.horn_radius + 1.5, fw + 2,
                                                        align=(Align.CENTER, Align.CENTER, Align.MIN)))
    hole = hole.moved(Location(Plane(origin=tuple(o), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2]))))
    return outer, [case, hole], (o - R[:, 0] * hx, R[:, 0], R[:, 2])


def swept_link(anchor, axis, path, width, range, span, gap: float = 1.0, reverse: bool = False):
    """The region a link with this centre `path` and `width` sweeps as it turns about
    (anchor, axis) through the joint `range`, over `span` = (s0, s1) mm along the axis.
    Give it to the parent part's `clear=` so the parent stays out of the link's way."""
    A = np.asarray(anchor, float)
    u, v, a = _plane_basis(axis)
    from manifold3d import CrossSection, JoinType
    uv = [(float(np.dot(np.asarray(p, float) - A, u)), float(np.dot(np.asarray(p, float) - A, v))) for p in path]
    lo, hi = range
    q = np.append(np.arange(lo, hi, SWEEP_STEP), hi) * (-1 if reverse else 1)
    r = width / 2 + gap
    pieces = []
    for t in q:
        c, s_ = np.cos(t), np.sin(t)
        pts = [(x * c - y * s_, x * s_ + y * c) for x, y in uv]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            circ = [(p0[0] + r * np.cos(k), p0[1] + r * np.sin(k)) for k in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
            circ += [(p1[0] + r * np.cos(k), p1[1] + r * np.sin(k)) for k in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
            pieces.append(CrossSection.hull_points(circ))
    import manifold3d
    cs = CrossSection.batch_boolean(pieces, manifold3d.OpType.Add)
    rings = [np.asarray(r_) for r_ in cs.to_polygons()]
    return _prism(rings, A, u, v, a, span[0] - gap, span[1] + gap)


def build_link(name, *, around=None, sweep=None, path=None, width=24.0, holds: Hold | None = None,
               add=(), cut=(), clear=(), wall=2.5, cheek=4.0, gap=0.8, round=2.0, bearing=None,
               back_cheek=True):
    """The link's world shape, its `holds` connector frame and the bearing seat, if any
    (center, axis, od, id, width). `sweep`: angles the driver's case turns through
    relative to this link. `bearing=(od, id, width)` seats a bearing in the back cheek."""
    from build123d import fillet
    shapes, cuts = list(add), list(cut)
    seat = None
    if around is not None:
        s = around.spec
        A, a = around.anchor, around.axis
        u, v, a = _plane_basis(a)
        pts = [np.asarray(p, float) for p in (path or [])]
        if len(pts) < 2:
            raise ValueError(f"link {name}: give a path of at least two world points (start near the axis)")
        uv = [(float(np.dot(p - A, u)), float(np.dot(p - A, v))) for p in pts]
        held_radial, held_back = getattr(around, "_held", (0.0, 0.0))
        back = max(s.back_face, s.case_half_axial + held_back)
        s0, s1 = -(back + gap + (cheek if back_cheek else 0.0)), s.horn_face + cheek
        if bearing is not None and not back_cheek:
            raise ValueError(f"link {name}: a bearing needs the back cheek")
        from .geom import polygon
        from build123d import Plane, extrude
        face = Plane(origin=tuple(A + a * s0), x_dir=tuple(u), z_dir=tuple(a)) * polygon(uv, round=width / 2)
        body = extrude(face, amount=s1 - s0, dir=tuple(a))
        if round > 0:
            try:
                body = fillet(body.edges(), round)
            except Exception:
                pass
        shapes.append(body)
        # what the driver's case (and its pocket walls) sweeps relative to this link
        Rc = around.case.pose[:3, :3]
        g = held_radial + gap
        corners = [Rc @ np.array([0.0, y, z]) for y in (-s.case_width / 2 - g, s.case_width / 2 + g)
                   for z in (-s.case_below - g, s.case_above + g)]
        rect = [(float(np.dot(c, u)), float(np.dot(c, v))) for c in corners]
        rect = [rect[0], rect[1], rect[3], rect[2]]
        angles = sweep if sweep is not None else np.linspace(0, 2 * np.pi, 73)
        rings = _swept_rings(rect, angles)
        cuts.append(_prism(rings, A, u, v, a, -(back + gap), s.horn_face + 0.01))
        if bearing is not None:
            od, id_, bw = bearing
            from .geom import cyl
            c = A + a * (s0 + bw / 2)
            cuts.append(cyl(c, a, od / 2, bw))
            seat = (c, a, od, id_, bw)
    if not shapes and holds is None:
        raise ValueError(f"link {name}: nothing to build")
    shape = None
    for sh in shapes:
        shape = sh if shape is None else shape + sh
    for c in clear:                      # room for moving neighbours; the pocket goes in after
        if shape is not None:
            shape = shape - c
    hold_frame = None
    if holds is not None:
        outer, cavity, hold_frame = _holder(holds, wall)
        shape = outer if shape is None else shape + outer
        cuts.extend(cavity)
        holds.unit._held = (wall, wall if holds.band is None else 0.0)   # (radial, behind the back)
    for c in cuts:
        shape = shape - c
    solids = shape.solids()
    if len(solids) != 1:
        vols = sorted((round_(sd.volume) for sd in solids), reverse=True)
        from .assembly import AttachmentError
        raise AttachmentError(f"link {name} came out in {len(solids)} separate pieces (volumes {vols} mm^3): "
                              f"its parts do not overlap; extend the path or add a connecting shape")
    return solids[0], hold_frame, seat


def round_(x):
    return int(x + 0.5)


def printed(name, *, holds: Hold | None = None, add=(), cut=(), clear=(), wall=2.5, material: Material = PLA,
            color=(0.93, 0.91, 0.86, 1.0), collision=False):
    """A printed part with no fork (a base, a gripper body): `add` shapes, minus `cut`, plus
    a walled pocket for `holds`. Connector `holds` seats that servo (fasten it to the case's
    `back`). Attach it to something, or use it as the assembly root."""
    from .assembly import Component
    shape, hf, _ = build_link(name, holds=holds, add=add, cut=cut, clear=clear, wall=wall)
    c = Component.world(name, shape, material, color, collision=collision)
    if hf is not None:
        c.connector("holds", hf[0], hf[1], hf[2], check=False)
    return c


def bearing(name, center, axis, od: float = 22.0, id: float = 16.0, width: float = 4.0, mass_g: float | None = None,
            color=(0.55, 0.57, 0.6, 1.0)):
    """A ball bearing (steel ring, mass from its volume unless `mass_g` is given). Attach it to
    the part it is pressed into."""
    from .assembly import Component
    from .geom import cyl
    from .materials import STEEL
    shape = cyl(center, axis, od / 2, width) - cyl(center, axis, id / 2, width + 1)
    return Component.world(name, shape, STEEL, color, mass_override=None if mass_g is None else mass_g / 1000)
