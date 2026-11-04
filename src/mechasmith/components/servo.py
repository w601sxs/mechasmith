"""Smart servos as reusable components: geometry, mass and actuator model in one place.

Local servo frame (used only inside this module):
    +X  output axis, horn on the +X face, idler/back boss on the -X face
    +Z  toward the SHORT end of the case (the axis sits near this end)
    -Z  toward the long end of the case ("body direction")
Callers place a servo with a world `axis` and a world `body_dir`.
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

    @property
    def horn_face(self) -> float:
        """Axial distance from the anchor to the outer face of the horn / back boss."""
        return self.case_half_axial + self.horn_thickness


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


@dataclass
class MountFace:
    """A flat face something can be mounted on: centre point (mm) + outward normal."""
    point: np.ndarray
    normal: np.ndarray


class Servo:
    """A servo placed in the world at the home pose."""

    def __init__(self, name: str, spec: ServoSpec, anchor, axis, body_dir):
        self.name = name
        self.spec = spec
        self.anchor = np.asarray(anchor, float)
        self.axis = unit(axis)
        self.body_dir = unit(body_dir)
        self.x, self.y, self.z = local_axes(self.axis, -self.body_dir)
        self.loc = frame(self.anchor, self.x, self.z)

    # ---- placement -------------------------------------------------------
    @classmethod
    def mount_on(cls, name, spec: ServoSpec, face: MountFace, axis, body_dir, along_axis: float = 0.0):
        """Place a servo so one case face sits flat on `face`.

        The case face touching the mount is the one whose outward normal is
        -face.normal; the anchor is pushed out from the face by that extent.
        """
        tmp = cls(name, spec, face.point, axis, body_dir)
        d = tmp.case_extent(-face.normal)
        anchor = face.point + unit(face.normal) * d + tmp.axis * along_axis
        return cls(name, spec, anchor, axis, body_dir)

    def case_extent(self, direction) -> float:
        """Distance from the anchor to the case face whose outward normal is `direction`."""
        d = unit(direction)
        s = self.spec
        for vec, pos, neg in ((self.x, s.case_half_axial, s.case_half_axial),
                              (self.y, s.case_width / 2, s.case_width / 2),
                              (self.z, s.case_above, s.case_below)):
            c = float(np.dot(d, vec))
            if abs(abs(c) - 1) < 1e-6:
                return pos if c > 0 else neg
        raise ValueError(f"{self.name}: direction {d} is not aligned with a case face")

    @classmethod
    def horn_on(cls, name, spec: ServoSpec, face: MountFace, body_dir):
        """Place a servo so its horn sits flat on `face` (inverted mounting: the
        case rides on the child link, the horn is bolted to the parent)."""
        n = unit(face.normal)
        return cls(name, spec, face.point + n * spec.horn_face, -n, body_dir)

    def horn_face(self) -> MountFace:
        return MountFace(self.anchor + self.axis * self.spec.horn_face, self.axis.copy())

    def boss_face(self) -> MountFace:
        return MountFace(self.anchor - self.axis * self.spec.horn_face, -self.axis)

    def case_center(self) -> np.ndarray:
        s = self.spec
        return self.anchor + self.z * (s.case_above - s.case_below) / 2

    def case_face(self, direction) -> MountFace:
        """Centre of the case face whose outward normal is `direction` (axial faces
        return the case face, not the horn/boss face)."""
        d = unit(direction)
        c = self.case_center()
        s = self.spec
        half = {0: s.case_half_axial, 1: s.case_width / 2, 2: (s.case_above + s.case_below) / 2}
        for i, vec in enumerate((self.x, self.y, self.z)):
            if abs(abs(float(np.dot(d, vec))) - 1) < 1e-6:
                return MountFace(c + d * half[i], d)
        raise ValueError(f"{self.name}: direction {d} is not aligned with a case face")

    # ---- parts -------------------------------------------------------------
    def _local(self, shape: Shape) -> Shape:
        return shape.moved(self.loc)

    def case_part(self) -> Part:
        """Case + back boss: belongs to the link the servo is bolted to."""
        s = self.spec
        h = s.case_above + s.case_below
        case = Pos(0, 0, (s.case_above - s.case_below) / 2) * Box(2 * s.case_half_axial, s.case_width, h)
        boss = Pos(-s.case_half_axial, 0, 0) * Rot(0, -90, 0) * Cylinder(
            s.horn_radius, s.horn_thickness, align=(Align.CENTER, Align.CENTER, Align.MIN))
        return Part(f"{self.name}_case", self._local(case + boss), BOUGHT, s.color,
                    mass_override=s.mass * 0.97)

    def horn_part(self) -> Part:
        """Output horn: belongs to the link the servo drives."""
        s = self.spec
        horn = Pos(s.case_half_axial, 0, 0) * Rot(0, 90, 0) * Cylinder(
            s.horn_radius, s.horn_thickness, align=(Align.CENTER, Align.CENTER, Align.MIN))
        return Part(f"{self.name}_horn", self._local(horn), BOUGHT, s.horn_color,
                    mass_override=s.mass * 0.03)
