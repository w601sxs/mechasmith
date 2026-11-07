"""Servo specs (geometry, mass and fitted actuator model) and their shapes.

Designs use `mechasmith.servo(name, XL330)`; this module holds the spec and the
internal shape builder. Servo frame:
    +X  output axis, horn on the +X face, idler/back boss on the -X face
    +Z  toward the SHORT end of the case (the axis sits near this end)
    -Z  toward the long end of the case ("body direction")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import Align, Box, Cylinder, Pos, Rot, Shape

from ..core import ActuatorModel, Part
from ..frames import frame, local_axes, unit
from ..materials import BOUGHT


@dataclass(frozen=True)
class ServoSpec:
    name: str
    mass: float                 # kg
    case_half_axial: float      # mm, half thickness of the case along the axis
    case_width: float           # mm, along local Y
    case_above: float           # mm, axis -> short end (+Z)
    case_below: float           # mm, axis -> long end (-Z)
    horn_radius: float          # mm
    horn_thickness: float       # mm (same for horn and back boss)
    actuator: ActuatorModel
    color: tuple = (0.17, 0.17, 0.18, 1.0)
    horn_color: tuple = (0.75, 0.75, 0.78, 1.0)
    boss_thickness: float | None = None  # rear idler boss; None = same as the horn, 0 = no boss

    @property
    def horn_face(self) -> float:
        """Axial distance from the anchor to the outer face of the horn."""
        return self.case_half_axial + self.horn_thickness

    @property
    def boss(self) -> float:
        """Thickness of the rear idler boss (0 if the servo has none)."""
        return self.horn_thickness if self.boss_thickness is None else self.boss_thickness

    @property
    def back_face(self) -> float:
        """Axial distance from the anchor to the rearmost face (boss, or the case if none)."""
        return self.case_half_axial + self.boss


# Dynamixel XL330 (as used on Microduck).
# Geometry: case 23 x 20 x 34 mm, output axis 9.5 mm from the short end, Ø16 x 3 mm
# horn and back boss (measured from the public microduck_rl model; matches the
# 20 x 34 x 26 mm datasheet envelope). Mass from the ROBOTIS datasheet.
# Actuator: Pollen Robotics' fitted "chosen_actuator" class in microduck_rl.
XL330 = ServoSpec(
    name="XL330",
    mass=0.018,
    case_half_axial=11.5,
    case_width=20.0,
    case_above=9.5,
    case_below=24.5,
    horn_radius=8.0,
    horn_thickness=3.0,
    actuator=ActuatorModel(
        kp=0.55, kv=0.0, forcerange=0.96, damping=0.053, frictionloss=0.0048,
        armature=0.0018, max_velocity=10.8,
        source="pollen-robotics/microduck_rl joints_properties.xml (chosen_actuator)",
    ),
)


# Feetech STS3215 (SO-100 / SO-101 arms, Open Duck Mini), 1/345 gearing.
# Geometry: case 45.4 x 24.8 x 35 mm, output axis 10.2 mm from one end, Ø20 x 3.4 mm horn,
# no rear boss (measured from the SO-ARM100 MJCF). Mass from the Feetech datasheet.
# Actuator: SO-ARM100 joints_properties.xml, adapted from Open Duck Mini's fit.
STS3215 = ServoSpec(
    name="STS3215",
    mass=0.055,
    case_half_axial=17.5,
    case_width=24.8,
    case_above=10.2,
    case_below=35.2,
    horn_radius=10.0,
    horn_thickness=3.4,
    boss_thickness=0.0,
    actuator=ActuatorModel(
        kp=17.8, kv=0.0, forcerange=3.35, damping=0.60, frictionloss=0.052, armature=0.028,
        max_velocity=4.7,
        source="TheRobotStudio/SO-ARM100 Simulation/SO101/joints_properties.xml (from Open Duck Mini)",
    ),
    color=(0.1, 0.1, 0.1, 1.0),
)


@dataclass
class MountFace:
    """A flat face: centre point (mm) + outward normal."""
    point: np.ndarray
    normal: np.ndarray


class ServoGeometry:
    """Internal: servo case and horn shapes in a given frame. Designs use mechasmith.servo()."""

    def __init__(self, name: str, spec: ServoSpec, anchor, axis, body_dir):
        self.name = name
        self.spec = spec
        self.anchor = np.asarray(anchor, float)
        self.axis = unit(axis)
        self.body_dir = unit(body_dir)
        self.x, self.y, self.z = local_axes(self.axis, -self.body_dir)
        self.loc = frame(self.anchor, self.x, self.z)

    # ---- parts -------------------------------------------------------------
    def _local(self, shape: Shape) -> Shape:
        return shape.moved(self.loc)

    def case_part(self) -> Part:
        """Case + back boss: belongs to the link the servo is bolted to."""
        s = self.spec
        h = s.case_above + s.case_below
        case = Pos(0, 0, (s.case_above - s.case_below) / 2) * Box(2 * s.case_half_axial, s.case_width, h)
        if s.boss > 0:
            case = case + Pos(-s.case_half_axial, 0, 0) * Rot(0, -90, 0) * Cylinder(
                s.horn_radius, s.boss, align=(Align.CENTER, Align.CENTER, Align.MIN))
        return Part(f"{self.name}_case", self._local(case), BOUGHT, s.color,
                    mass_override=s.mass * 0.97)

    def horn_part(self) -> Part:
        """Output horn: belongs to the link the servo drives."""
        s = self.spec
        horn = Pos(s.case_half_axial, 0, 0) * Rot(0, 90, 0) * Cylinder(
            s.horn_radius, s.horn_thickness, align=(Align.CENTER, Align.CENTER, Align.MIN))
        return Part(f"{self.name}_horn", self._local(horn), BOUGHT, s.horn_color,
                    mass_override=s.mass * 0.03)
