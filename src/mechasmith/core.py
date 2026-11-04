"""Core model: parts, links, joints, robots.

Conventions (the whole point is that an agent can reason in ONE frame):
  * Everything is authored in a single world frame at the robot's home pose.
  * Units are millimetres while authoring; exporters convert to SI.
  * x forward, y left, z up.
  * A joint is an anchor point + axis in that world frame. Exporters derive
    every body frame from those, so no hand-written body transforms exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from build123d import CenterOf, Shape

from .materials import Material


@dataclass
class ActuatorModel:
    """Position-controlled servo as MuJoCo sees it (SI units)."""

    kp: float
    kv: float
    forcerange: float          # N*m, symmetric
    damping: float             # joint damping, N*m*s/rad
    frictionloss: float        # N*m
    armature: float            # kg*m^2 (reflected rotor inertia)
    max_velocity: float        # rad/s, no-load (used for URDF + checks)
    source: str = ""           # where these numbers came from


@dataclass
class Part:
    name: str
    shape: Shape                       # world frame, home pose, mm
    material: Material
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    collision: bool = False            # also export as collision geometry
    mass_override: float | None = None # kg, for bought parts with known mass

    # ---- mass properties (SI) ------------------------------------------
    def mass_properties(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (mass kg, com m (world), inertia about com kg*m^2 (world axes))."""
        vol_mm3 = self.shape.volume
        if vol_mm3 <= 0:
            raise ValueError(f"part {self.name!r} has non-positive volume")
        if self.mass_override is not None:
            mass = self.mass_override
        else:
            mass = self.material.effective_mass(vol_mm3, self.shape.area)
        com = np.array(tuple(self.shape.center(CenterOf.MASS))) * 1e-3
        # matrix_of_inertia is for unit density, in mm^5, about the centre of mass.
        i_unit = np.array(self.shape.matrix_of_inertia, dtype=float)
        # Scale so the inertia corresponds to the actual mass, assuming the
        # mass is distributed like the solid (a stated approximation for
        # infilled prints and bought parts).
        inertia = i_unit * (mass / vol_mm3) * 1e-6  # mm^5 * kg/mm^3 -> kg*mm^2 -> kg*m^2
        return mass, com, inertia


@dataclass
class Link:
    name: str
    parts: list[Part] = field(default_factory=list)
    sites: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # world mm

    def add(self, *parts: Part) -> "Link":
        self.parts.extend(parts)
        return self

    def mass_properties(self) -> tuple[float, np.ndarray, np.ndarray]:
        props = [p.mass_properties() for p in self.parts]
        if not props:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        m = sum(p[0] for p in props)
        com = sum(p[0] * p[1] for p in props) / m
        inertia = np.zeros((3, 3))
        for mi, ci, ii in props:
            d = ci - com
            inertia += ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return m, com, inertia


@dataclass
class Joint:
    name: str
    parent: Link
    child: Link
    anchor: np.ndarray                 # world, mm
    axis: np.ndarray                   # world, unit
    range: tuple[float, float]         # rad
    actuator: ActuatorModel | None = None


class Robot:
    def __init__(self, name: str, root: Link, floating: bool = True):
        self.name = name
        self.root = root
        self.floating = floating
        self.links: dict[str, Link] = {root.name: root}
        self.joints: list[Joint] = []

    def joint(self, name, parent: Link, child: Link, anchor, axis, range, actuator=None) -> Joint:
        if child.name in self.links:
            raise ValueError(f"link {child.name!r} already has a parent (kinematic tree only)")
        if parent.name not in self.links:
            raise ValueError(f"parent link {parent.name!r} is not attached to the robot yet")
        if any(j.name == name for j in self.joints):
            raise ValueError(f"duplicate joint name {name!r}")
        axis = np.asarray(axis, float)
        axis = axis / np.linalg.norm(axis)
        j = Joint(name, parent, child, np.asarray(anchor, float), axis, tuple(range), actuator)
        self.links[child.name] = child
        self.joints.append(j)
        return j

    # ---- tree helpers --------------------------------------------------
    def parent_joint(self, link: Link) -> Joint | None:
        return next((j for j in self.joints if j.child is link), None)

    def children(self, link: Link) -> list[Joint]:
        return [j for j in self.joints if j.parent is link]

    def body_origin(self, link: Link) -> np.ndarray:
        """Body frame origin (world mm, home pose): the parent joint anchor, or 0 for root."""
        j = self.parent_joint(link)
        return np.zeros(3) if j is None else j.anchor

    def total_mass(self) -> float:
        return sum(l.mass_properties()[0] for l in self.links.values())
