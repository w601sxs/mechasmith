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
