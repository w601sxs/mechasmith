"""The resolved robot model: links of placed parts, joined by joints.

Designs do not build this by hand: `Assembly.robot()` produces it from connected
components, and exporters and checks read it. `Robot.mirror` is the one method
designs call directly.

Conventions (the whole point is that an agent can reason in ONE frame):
  * Everything is authored in a single world frame at the robot's home pose.
  * Units are millimetres while authoring; exporters convert to SI.
  * x forward, y left, z up.
  * A joint is an anchor point + axis in that world frame. Exporters derive
    every body frame from those, so no hand-written body transforms exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from build123d import CenterOf, Plane, Shape, mirror

# Mirror planes: name -> (build123d plane, reflection matrix)
MIRROR_PLANES = {
    "XZ": (Plane.XZ, np.diag([1.0, -1.0, 1.0])),   # left <-> right
    "YZ": (Plane.YZ, np.diag([-1.0, 1.0, 1.0])),   # front <-> back
    "XY": (Plane.XY, np.diag([1.0, 1.0, -1.0])),   # top <-> bottom
}

from .materials import Material


@dataclass
class ActuatorModel:
    """A joint actuator as MuJoCo sees it (SI units). mode="position": a position servo with
    gains kp/kv. mode="torque": a torque motor (ctrl is N*m within +-forcerange); kp/kv are then
    the suggested PD gains, used by the checks and written to the manifest."""

    kp: float
    kv: float
    forcerange: float          # N*m, symmetric
    damping: float             # joint damping, N*m*s/rad
    frictionloss: float        # N*m
    armature: float            # kg*m^2 (reflected rotor inertia)
    max_velocity: float        # rad/s, no-load (used for URDF + checks)
    source: str = ""           # where these numbers came from
    mode: str = "position"     # "position" | "torque"
    dampratio: float | None = None   # position mode: damping as a ratio of critical (MuJoCo works out kv)


@dataclass
class Part:
    name: str
    shape: Shape                       # world frame, home pose, mm
    material: Material
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    collision: bool | str = False      # also export as collision: True/"mesh", or a fitted
                                       # primitive "box", "sphere", "cylinder", "capsule"
    mass_override: float | None = None # kg, for bought parts with known mass
    contact: dict | None = None        # MJCF contact settings: friction, condim, solimp, solref, priority
    mass_budget: float | None = None   # kg: the whole link weighs this; this part takes what the rest leaves

    def mirrored(self, name: str, plane: str = "XZ") -> "Part":
        """Mirror image of this part across a world plane through the origin."""
        return replace(self, name=name, shape=mirror(self.shape, about=MIRROR_PLANES[plane][0]))

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
class Rotor:
    """A motor and propeller: thrust along `axis` at `pos`, plus the reaction torque.
    spin=+1: the propeller turns counter-clockwise seen from above (about +axis), so the
    body feels -torque_ratio * thrust about the axis."""
    name: str
    link: "Link"
    pos: np.ndarray          # world mm, home pose
    axis: np.ndarray         # world unit
    spin: int                # +1 or -1
    thrust_max: float        # N
    torque_ratio: float      # m: yaw torque per newton of thrust
    source: str = ""


@dataclass
class Joint:
    name: str
    parent: Link
    child: Link
    anchor: np.ndarray                 # world, mm
    axis: np.ndarray                   # world, unit
    range: tuple[float, float]         # rad
    actuator: ActuatorModel | None = None  # None: a passive pivot
    stiffness: float = 0.0             # N*m/rad spring, toward springref
    springref: float = 0.0             # rad
    damping: float | None = None       # N*m*s/rad; None: the actuator's
    armature: float | None = None
    pin: float = 0.0                   # mm: the pin may overlap both sides within this radius


class Robot:
    def __init__(self, name: str, root: Link, floating: bool = True):
        self.name = name
        self.root = root
        self.floating = floating
        self.links: dict[str, Link] = {root.name: root}
        self.joints: list[Joint] = []
        self.loops: list[tuple[Link, Link, np.ndarray]] = []       # pinned pairs that close a loop (world mm)
        self.couplings: list[tuple[str, str, float]] = []          # (follower, leader, ratio)
        self.rotors: list[Rotor] = []                              # propellers: thrust + reaction torque
        self.sensors: list[tuple[str, str]] = []                   # (kind, site): "gyro", "accelerometer", ...

    def couple(self, follower: str, leader: str, ratio: float = 1.0) -> "Robot":
        """Make one joint follow another (q_follower = ratio * q_leader), e.g. two fingers
        driven by one motor. The follower must not have its own actuator."""
        names = {j.name: j for j in self.joints}
        for n in (follower, leader):
            if n not in names:
                raise ValueError(f"couple: no joint {n!r}")
        if names[follower].actuator is not None:
            names[follower].actuator = None          # the leader's motor drives both
        self.couplings.append((follower, leader, float(ratio)))
        return self

    def add_joint(self, name, parent: Link, child: Link, anchor, axis, range, actuator=None, **dynamics) -> Joint:
        if child.name in self.links:
            raise ValueError(f"link {child.name!r} already has a parent (kinematic tree only)")
        if parent.name not in self.links:
            raise ValueError(f"parent link {parent.name!r} is not attached to the robot yet")
        if any(j.name == name for j in self.joints):
            raise ValueError(f"duplicate joint name {name!r}")
        axis = np.asarray(axis, float)
        axis = axis / np.linalg.norm(axis)
        j = Joint(name, parent, child, np.asarray(anchor, float), axis, tuple(range), actuator, **dynamics)
        self.links[child.name] = child
        self.joints.append(j)
        return j

    def resolve_budgets(self) -> list[str]:
        """Give every part with a mass budget what its link's other parts leave (after mirroring
        and anything else that adds parts). Returns notes for budgets the other parts exceed."""
        notes = []
        for link in self.links.values():
            for part in link.parts:
                if part.mass_budget is None:
                    continue
                others = sum(p.mass_properties()[0] for p in link.parts if p is not part)
                want = part.mass_budget - others
                if want < 0.05 * part.mass_budget:
                    notes.append(f"{link.name}: other parts weigh {others:.3f} kg, over the {part.mass_budget:.3f} kg budget")
                    want = 0.05 * part.mass_budget
                part.mass_override = want
        self.budget_notes = notes
        return notes

    def print_settings(self, infill: float | None = None, wall_mm: float | None = None,
                       materials: tuple[str, ...] = ("PLA",)) -> "Robot":
        """Reprint every printed part made of `materials` with another infill and/or wall
        thickness (mass changes; geometry does not). Returns the robot, for chaining."""
        from dataclasses import replace
        changes = {k: v for k, v in (("infill", infill), ("wall_mm", wall_mm)) if v is not None}
        for link in self.links.values():
            for p in link.parts:
                if p.material.printed and p.material.name in materials and changes:
                    p.material = replace(p.material, **changes)
        return self

    # ---- tree helpers --------------------------------------------------
    def parent_joint(self, link: Link) -> Joint | None:
        return next((j for j in self.joints if j.child is link), None)

    def children(self, link: Link) -> list[Joint]:
        return [j for j in self.joints if j.parent is link]

    def body_origin(self, link: Link) -> np.ndarray:
        """Body frame origin (world mm, home pose): the parent joint anchor, or 0 for root."""
        j = self.parent_joint(link)
        return np.zeros(3) if j is None else j.anchor

    # ---- symmetry ------------------------------------------------------
    def mirror(self, src: str = "left_", dst: str = "right_", plane: str = "XZ",
               axes: str = "vector") -> "Robot":
        """Copy everything named `src...` to its mirror image named `dst...`.

        Mirrored: every link whose name starts with `src` (with its parts, sites and
        parent joint), plus `src`-named parts and sites inside shared links such as the
        trunk. Build one side, then call `robot.mirror()`.

        axes="vector": the axis is reflected like a direction, so symmetric motion needs
            opposite joint values (q_right = -q_left) and ranges flip to (-hi, -lo).
            This is the URDF/ROS habit and what microduck_rl uses.
        axes="pseudo": the axis is reflected as a rotation axis, so the same joint value
            gives the mirror-image motion and ranges stay as they are.
        axes="closest" (alias "same"): each joint is reflected whichever way keeps its axis
            closest to the original, so axes point the same way on both sides (Unitree, most
            ROS robots): pitch-like joints keep their range, roll and yaw joints mirror it.
        """
        if axes not in ("vector", "pseudo", "closest", "same"):
            raise ValueError("axes must be 'vector', 'pseudo' or 'closest'")
        _, M = MIRROR_PLANES[plane]

        def rename(n: str) -> str:
            return dst + n[len(src):] if n.startswith(src) else dst + n

        def mpt(p) -> tuple:
            return tuple(float(v) for v in M @ np.asarray(p, float))

        if any(n.startswith(dst) for n in self.links):
            raise ValueError(f"links named {dst!r}... already exist; mirror once")
        # src-named parts and sites that live in shared links
        for link in list(self.links.values()):
            if link.name.startswith(src):
                continue
            link.parts += [p.mirrored(rename(p.name), plane) for p in link.parts if p.name.startswith(src)]
            link.sites.update({rename(k): mpt(v) for k, v in link.sites.items() if k.startswith(src)})
        # src links, in tree order so parents exist first
        for j in [j for j in self.joints if j.child.name.startswith(src)]:
            parent = self.links[rename(j.parent.name)] if j.parent.name.startswith(src) else j.parent
            child = Link(rename(j.child.name), [p.mirrored(rename(p.name), plane) for p in j.child.parts],
                         {rename(k): mpt(v) for k, v in j.child.sites.items()})
            axis = M @ j.axis if axes == "vector" else -(M @ j.axis)
            lo, hi = j.range
            rng = (-hi, -lo) if axes == "vector" else (lo, hi)
            if axes in ("closest", "same"):
                # reflect as a direction or as a rotation axis, whichever stays closest to the original
                v, pz = M @ j.axis, -(M @ j.axis)
                axis, rng = (v, (-hi, -lo)) if np.dot(v, j.axis) >= np.dot(pz, j.axis) else (pz, (lo, hi))
            keep = axes == "pseudo" or (axes in ("closest", "same") and rng == j.range)
            self.add_joint(rename(j.name), parent, child, M @ j.anchor, axis, rng, j.actuator,
                           stiffness=j.stiffness, springref=j.springref if keep else -j.springref,
                           damping=j.damping, armature=j.armature, pin=j.pin)
        for a, b, p in list(self.loops):
            if a.name.startswith(src) or b.name.startswith(src):
                self.loops.append((self.links[rename(a.name)] if a.name.startswith(src) else a,
                                   self.links[rename(b.name)] if b.name.startswith(src) else b, M @ p))
        for f, l, r in list(self.couplings):
            if f.startswith(src) or l.startswith(src):
                self.couplings.append((rename(f) if f.startswith(src) else f, rename(l) if l.startswith(src) else l, r))
        radius = getattr(self, "site_radius", {})
        for kind, site, *opts in list(self.sensors):
            if site.startswith(src):
                self.sensors.append((kind, rename(site), *opts))
                if site in radius:
                    radius[rename(site)] = radius[site]
        for r in [r for r in self.rotors if r.name.startswith(src)]:
            self.rotors.append(Rotor(rename(r.name), self.links[rename(r.link.name)] if r.link.name.startswith(src)
                                     else r.link, M @ r.pos, M @ r.axis, -r.spin, r.thrust_max, r.torque_ratio,
                                     r.source))
        return self

    def total_mass(self) -> float:
        return sum(l.mass_properties()[0] for l in self.links.values())
