"""Parametric brackets that attach to servos.

Each function returns (Part, MountFace): the bracket and the face the next
component should be mounted on. Brackets are generated from the servo's own
frame, so they stay correct when the servo moves.
"""
from __future__ import annotations

import numpy as np
from build123d import Align, Box, Circle, Plane, Pos, Rectangle, extrude

from .components.servo import MountFace, Servo
from .core import Part
from .frames import frame, unit
from .materials import PLA, Material


def u_bracket(servo: Servo, toward, reach: float, *, name: str | None = None,
              thickness: float = 2.5, width: float | None = None, idler_gap: float = 0.4,
              material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0)) -> tuple[Part, MountFace]:
    """Classic servo U-bracket: one cheek bolted to the horn, one riding on the back
    boss, joined by a bridge `reach` mm from the axis in direction `toward`.

    `toward` must be perpendicular to the servo axis. Full rotation is collision
    free only if `reach` exceeds the case's swept radius; the sweep check reports
    the real usable range either way.
    """
    s = servo.spec
    t = thickness
    w = width or 2 * (s.horn_radius + 1.5)
    toward = unit(toward)
    if abs(np.dot(toward, servo.axis)) > 1e-6:
        raise ValueError("u_bracket: `toward` must be perpendicular to the servo axis")

    # Bracket-local frame: X = servo axis, Z = toward, origin at the servo anchor.
    x_horn = s.horn_face                 # horn-side cheek inner face (bolted, touching)
    x_idle = -(s.horn_face + idler_gap)  # idler-side cheek inner face (running clearance)
    r = w / 2

    def cheek(x0: float, direction: int):
        prof = Circle(r) + Rectangle(w, reach + t, align=(Align.CENTER, Align.MIN))
        pl = Plane.YZ.offset(x0) if direction > 0 else Plane.YZ.offset(x0 - t)
        return extrude(pl * prof, amount=t)

    span_min, span_max = x_idle - t, x_horn + t
    bridge = Pos((span_min + span_max) / 2, 0, reach + t / 2) * Box(span_max - span_min, w, t)
    shape = cheek(x_horn, +1) + cheek(x_idle, -1) + bridge

    loc = frame(servo.anchor, servo.axis, toward)
    part = Part(name or f"{servo.name}_ubracket", shape.moved(loc), material, color)
    mid = servo.anchor + servo.axis * (span_min + span_max) / 2 + toward * (reach + t)
    return part, MountFace(mid, toward)


def horn_plate(servo: Servo, size: tuple[float, float], *, name: str | None = None,
               thickness: float = 2.5, offset=(0.0, 0.0), material: Material = PLA,
               color=(0.93, 0.91, 0.86, 1.0)) -> tuple[Part, MountFace]:
    """Flat plate bolted to the horn face (single-sided). `size` is (along servo
    local Y, along servo local Z); `offset` shifts the plate centre in that plane."""
    s = servo.spec
    plate = Pos(s.horn_face + thickness / 2, offset[0], offset[1]) * Box(thickness, size[0], size[1])
    part = Part(name or f"{servo.name}_hornplate", plate.moved(servo.loc), material, color)
    centre = servo.anchor + servo.axis * (s.horn_face + thickness) + servo.y * offset[0] + servo.z * offset[1]
    return part, MountFace(centre, servo.axis.copy())


# ---- axis-aligned connectors ------------------------------------------------
# These build plates as world-axis-aligned boxes, which covers the common
# case of servos placed on world axes. Each returns a Part.

def _box_from_bounds(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Pos(*((lo + hi) / 2)) * Box(*(hi - lo))


def _axis_index(n) -> int:
    n = unit(n)
    i = int(np.argmax(np.abs(n)))
    if abs(abs(n[i]) - 1) > 1e-6:
        raise ValueError(f"normal {n} is not a world axis")
    return i


def _plate_bounds(face: MountFace, t: float):
    """(lo, hi) along the face normal axis for a plate of thickness t on the face."""
    i = _axis_index(face.normal)
    s = np.sign(face.normal[i])
    a, b = face.point[i], face.point[i] + s * t
    return i, min(a, b), max(a, b)


def strap(face_a: MountFace, face_b: MountFace, *, width: float = 19.0, thickness: float = 2.5,
          name: str, material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0)) -> Part:
    """Flat plate over two coplanar faces with the same normal (e.g. a shin plate
    from the knee horn down to the ankle case)."""
    i, lo_n, hi_n = _plate_bounds(face_a, thickness)
    pts = np.array([face_a.point, face_b.point])
    lo, hi = pts.min(0) - width / 2, pts.max(0) + width / 2
    lo[i], hi[i] = lo_n, hi_n
    return Part(name, _box_from_bounds(lo, hi), material, color)


def l_bracket(face_a: MountFace, face_b: MountFace, *, width: float = 19.0, thickness: float = 2.5,
              name: str, material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0)) -> Part:
    """Two plates joined at a corner, one on each of two perpendicular faces."""
    ia, a_lo, a_hi = _plate_bounds(face_a, thickness)
    ib, b_lo, b_hi = _plate_bounds(face_b, thickness)
    if ia == ib:
        raise ValueError("l_bracket: faces must be perpendicular (use strap or spacer)")
    pa, pb = face_a.point, face_b.point
    k = 3 - ia - ib  # the shared in-plane axis
    k_lo = min(pa[k], pb[k]) - width / 2
    k_hi = max(pa[k], pb[k]) + width / 2
    # plate A: on face A, spans along ib from its own centre to plate B's far side
    lo_a, hi_a = np.zeros(3), np.zeros(3)
    lo_a[ia], hi_a[ia] = a_lo, a_hi
    lo_a[ib], hi_a[ib] = min(pa[ib] - width / 2, b_lo), max(pa[ib] + width / 2, b_hi)
    lo_a[k], hi_a[k] = k_lo, k_hi
    lo_b, hi_b = np.zeros(3), np.zeros(3)
    lo_b[ib], hi_b[ib] = b_lo, b_hi
    lo_b[ia], hi_b[ia] = min(pb[ia] - width / 2, a_lo), max(pb[ia] + width / 2, a_hi)
    lo_b[k], hi_b[k] = k_lo, k_hi
    shape = _box_from_bounds(lo_a, hi_a) + _box_from_bounds(lo_b, hi_b)
    return Part(name, shape, material, color)


def spacer(face_a: MountFace, face_b: MountFace, *, width: float = 19.0, name: str,
           material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0)) -> Part:
    """Solid block filling the gap between two parallel, facing faces."""
    i = _axis_index(face_a.normal)
    pa, pb = face_a.point, face_b.point
    c = (pa + pb) / 2
    lo, hi = c - width / 2, c + width / 2
    lo[i], hi[i] = min(pa[i], pb[i]), max(pa[i], pb[i])
    return Part(name, _box_from_bounds(lo, hi), material, color)


def cover_plate(parts, normal, *, thickness: float = 2.5, margin: float = 1.0, name: str,
                material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0)) -> Part:
    """Plate over the outermost faces of `parts` on side `normal`, covering their
    combined outline (ties several servo cases into one rigid link)."""
    i = _axis_index(normal)
    s = np.sign(np.asarray(normal, float)[i])
    bbs = [p.shape.bounding_box() for p in parts]
    lo = np.min([[b.min.X, b.min.Y, b.min.Z] for b in bbs], 0) - margin
    hi = np.max([[b.max.X, b.max.Y, b.max.Z] for b in bbs], 0) + margin
    edge = (hi[i] - margin) if s > 0 else (lo[i] + margin)
    lo[i], hi[i] = (edge, edge + thickness) if s > 0 else (edge - thickness, edge)
    return Part(name, _box_from_bounds(lo, hi), material, color)
