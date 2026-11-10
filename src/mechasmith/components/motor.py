"""Cylindrical joint motors (quasi-direct-drive "pancake" actuators, as on Unitree, MIT Cheetah
and most RL quadrupeds and humanoids).

A `MotorSpec` is the datasheet: size, mass, gear ratio, rotor inertia, peak torque and speed.
The simulated actuator follows from it: armature = rotor inertia x (gear x reduction)^2,
torque limit = peak torque x reduction, speed limit = output speed / reduction. Motors are
torque-controlled by default (the RL convention: the policy's PD runs outside the sim); `kp`/`kv`
are the suggested PD gains, used by the checks and written to the manifest.

Motor frame (same as a servo's): +X output axis, the flange on the +X face, the anchor at the
centre of the case on the axis. `motor(name, spec)` gives a unit with a `case` (stator, bolted
to the parent link) and a `horn` (the output flange, bolted to the child) for `asm.joint`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import Align, Cylinder, Pos, Rot

from ..core import ActuatorModel


@dataclass(frozen=True)
class MotorSpec:
    name: str
    mass: float                  # kg
    diameter: float              # mm, case
    length: float                # mm, case along the axis
    flange_diameter: float       # mm, output flange
    flange_thickness: float      # mm
    gear: float                  # internal gear ratio
    rotor_inertia: float         # kg*m^2 on the motor side
    peak_torque: float           # N*m at the output, before any extra reduction
    max_speed: float             # rad/s at the output, before any extra reduction
    damping: float = 0.0         # N*m*s/rad at the joint
    frictionloss: float = 0.0    # N*m at the joint
    kp: float = 20.0             # suggested PD gains for the joint
    kv: float = 0.5
    flange_mass_fraction: float = 0.25
    source: str = ""

    def actuator(self, reduction: float = 1.0, mode: str = "torque", armature: float | None = None) -> ActuatorModel:
        n = self.gear * reduction
        return ActuatorModel(kp=self.kp * reduction, kv=self.kv * reduction, forcerange=self.peak_torque * reduction,
                             damping=self.damping, frictionloss=self.frictionloss,
                             armature=self.rotor_inertia * n * n if armature is None else armature,
                             max_velocity=self.max_speed / reduction, source=self.source, mode=mode)


# Unitree GO-M8010-6 (Go2, A1-class legs). Peak torque and speed from Unitree's figures and the
# Menagerie go2 model; rotor inertia chosen so the reflected inertia is Menagerie's 0.01 kg*m^2.
GO_M8010_6 = MotorSpec(
    name="GO-M8010-6", mass=0.53, diameter=96.0, length=40.0, flange_diameter=64.0, flange_thickness=4.0,
    gear=6.33, rotor_inertia=0.01 / 6.33 ** 2, peak_torque=23.7, max_speed=30.1, damping=2.0, frictionloss=0.2,
    kp=25.0, kv=0.5, source="Unitree GO-M8010-6 datasheet / Menagerie unitree_go2 (verify)",
)


class MotorGeometry:
    """Internal: case and flange shapes in the motor frame."""

    def __init__(self, spec: MotorSpec):
        self.spec = spec

    def case(self):
        s = self.spec
        return Pos(-s.length / 2, 0, 0) * Rot(0, 90, 0) * Cylinder(s.diameter / 2, s.length,
                                                                  align=(Align.CENTER, Align.CENTER, Align.MIN))

    def flange(self):
        s = self.spec
        return Pos(s.length / 2, 0, 0) * Rot(0, 90, 0) * Cylinder(s.flange_diameter / 2, s.flange_thickness,
                                                                 align=(Align.CENTER, Align.CENTER, Align.MIN))
