"""Assemblies: say which parts join which, and how; mechasmith places them.

You never type where a servo or bracket goes. You connect it:

    asm = Assembly("biped", root=trunk_shell)          # the root is the only free part
    asm.fasten(trunk_shell.c("neck_seat"), mount.c("base"), spin=(mount.AXIS, LEFT))
    asm.fasten(mount.c("horn"), neck.horn.c("face"))
    asm.joint("neck_pitch", neck, range=(-1.57, 1.05))  # the servo is the joint
    robot = asm.robot()

Rules that keep every part attached:
  * A part gets a position only by being connected to a part that already has one,
    so declare from the root outward. A part that is never connected is not in the robot.
  * `fasten(a, b)` puts b's connector face flat against a's: the faces touch by construction.
  * `attach(base, part)` is for parts drawn in world coordinates (shells, covers). The
    contact is checked right there, and a gap raises AttachmentError naming both parts.
  * Connecting a part that is already placed checks that both connections agree.
  * Rigidly connected parts become one simulated link; `joint` splits links.

Connector frames: z is the face's outward normal, x is a reference direction on the face.
`fasten` flips b so the normals oppose and the x directions match, then applies `spin`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from build123d import Align, Location, Plane, Pos, Shape, Vertex

from .components.servo import MountFace, ServoGeometry as Servo, ServoSpec
from .core import ActuatorModel, Link, Part, Robot
from .frames import unit
from .materials import BOUGHT, PLA, Material

TOUCH_MM = 0.3      # parts closer than this count as touching
AGREE_MM = 0.05     # two connections placing the same part must agree this well
AGREE_DEG = 0.1


class AttachmentError(ValueError):
    pass


def _overlap(a: Shape, b: Shape) -> bool:
    """True when two solids share volume (e.g. a motor inside a housing)."""
    try:
        return (a & b).volume > 1e-3
    except Exception:
        return False


# ---- frames ------------------------------------------------------------------------

def _frame(origin, z, x) -> np.ndarray:
    z = unit(z)
    x = np.asarray(x, float)
    x = unit(x - np.dot(x, z) * z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, origin
    return T


def _any_perp(z) -> np.ndarray:
    z = unit(z)
    return np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    T = np.eye(4)
    if axis == "x":
        T[1:3, 1:3] = [[c, -s], [s, c]]
    else:
        T[0:2, 0:2] = [[c, -s], [s, c]]
    return T


def _location(T: np.ndarray) -> Location:
    return Location(Plane(origin=tuple(T[:3, 3]), x_dir=tuple(T[:3, 0]), z_dir=tuple(T[:3, 2])))


# ---- components --------------------------------------------------------------------

@dataclass
class Conn:
    comp: "Component"
    name: str | None = None
    toward: np.ndarray | None = None   # lazy: the connector facing this world direction

    def frame_local(self) -> np.ndarray:
        if self.name is not None:
            return self.comp.connectors[self.name]
        if self.comp.pose is None:
            raise AttachmentError(f"{self.comp.name}.face(...) needs {self.comp.name} placed first")
        R = self.comp.pose[:3, :3]
        best = max(self.comp.connectors.items(), key=lambda kv: float(np.dot(R @ kv[1][:3, 2], self.toward)))
        if np.dot(R @ best[1][:3, 2], self.toward) < 0.99:
            raise AttachmentError(f"{self.comp.name} has no connector facing {self.toward.round(3).tolist()}; "
                                  f"it has {sorted(self.comp.connectors)}")
        self.name = best[0]
        return best[1]

    def __repr__(self):
        return f"{self.comp.name}.{self.name or 'face(' + str(self.toward.round(2).tolist()) + ')'}"


@dataclass
class Component:
    """A rigid part with named connectors. `shape` is in the part's own frame;
    parts made with `Component.world(...)` are drawn in world coordinates."""
    name: str
    shape: Shape
    material: Material = PLA
    color: tuple = (0.8, 0.8, 0.8, 1.0)
    collision: bool | str = False
    mass_override: float | None = None
    contact: dict | None = None
    mass_budget: float | None = None   # kg: this part's link weighs this in all; the part takes the rest
    connectors: dict[str, np.ndarray] = field(default_factory=dict)
    sites: dict[str, tuple] = field(default_factory=dict)   # local frame, mm
    pose: np.ndarray | None = None                           # set when placed
    _world_shape: Shape | None = field(default=None, repr=False)

    @classmethod
    def world(cls, name, shape, material=PLA, color=(0.8, 0.8, 0.8, 1.0), **kw) -> "Component":
        """A part drawn directly in world coordinates (it still has to be attached)."""
        return cls(name, shape, material, color, pose=np.eye(4), **kw)

    def connector(self, name: str, at, normal, x=None, check: bool = True) -> "Component":
        """Declare a connector on the part's surface (own frame for local parts)."""
        at = np.asarray(at, float)
        if check:
            d = self.shape.distance_to(Vertex(*at))
            if d > AGREE_MM:
                raise AttachmentError(f"connector {self.name}.{name} at {at.round(2).tolist()} is {d:.2f} mm "
                                      f"off the part's surface")
        self.connectors[name] = _frame(at, normal, _any_perp(normal) if x is None else x)
        return self

    def c(self, name: str) -> Conn:
        if name not in self.connectors:
            raise KeyError(f"{self.name} has no connector {name!r}; it has {sorted(self.connectors)}")
        return Conn(self, name)

    def face(self, world_dir) -> Conn:
        """The connector currently facing `world_dir` (resolved when used)."""
        return Conn(self, None, unit(world_dir))

    @property
    def world_shape(self) -> Shape:
        if self.pose is None:
            raise AttachmentError(f"{self.name} is not placed yet")
        if self._world_shape is None:
            self._world_shape = self.shape.moved(_location(self.pose))
        return self._world_shape

    def at(self, connector: str) -> np.ndarray:
        """World position of one of this part's connectors (the part must be placed)."""
        return self.point(self.connectors[connector][:3, 3])

    def point(self, local) -> np.ndarray:
        """World position of a point given in the part's own frame."""
        return (self.pose @ np.append(np.asarray(local, float), 1.0))[:3]

    def direction(self, local) -> np.ndarray:
        return self.pose[:3, :3] @ np.asarray(local, float)


# ---- servos -------------------------------------------------------------------------

class ServoUnit:
    """A servo as two components, case and horn, sharing one frame:
    +X is the output axis (horn side), +Z points to the short end of the case.

    Case connectors: short_end, long_end, side_pos (+Y), side_neg (-Y), back (axial face
    opposite the horn), and boss (outer face of the rear boss, if the servo has one).
    Horn connector: face.
    """
    AXIS = np.array([1.0, 0, 0])   # output axis, servo frame
    BODY = np.array([0, 0, -1.0])  # toward the long end of the case, servo frame

    def __init__(self, name: str, spec: ServoSpec):
        self.name, self.spec = name, spec
        s = spec
        local = Servo(name, spec, (0, 0, 0), axis=(1, 0, 0), body_dir=(0, 0, -1))
        cp, hp = local.case_part(), local.horn_part()
        self.case = Component(cp.name, cp.shape, cp.material, cp.color, mass_override=cp.mass_override)
        self.horn = Component(hp.name, hp.shape, hp.material, hp.color, mass_override=hp.mass_override)
        X, Y, Z = np.eye(3)
        for nm, at, n, x in (("short_end", (0, 0, s.case_above), Z, X), ("long_end", (0, 0, -s.case_below), -Z, X),
                             ("side_pos", (0, s.case_width / 2, 0), Y, X), ("side_neg", (0, -s.case_width / 2, 0), -Y, X),
                             ("back", (-s.case_half_axial, 0, 0), -X, Z)):
            self.case.connector(nm, at, n, x, check=False)
        if s.boss > 0:
            self.case.connector("boss", (-s.back_face, 0, 0), -X, Z, check=False)
        self.horn.connector("face", (s.horn_face, 0, 0), X, Z, check=False)
        self.default_actuator = spec.actuator

    @property
    def placed(self) -> Component:
        for c in (self.case, self.horn):
            if c.pose is not None:
                return c
        raise AttachmentError(f"servo {self.name} is not placed yet")

    @property
    def anchor(self) -> np.ndarray:
        return self.placed.point((0, 0, 0))

    @property
    def axis(self) -> np.ndarray:
        return self.placed.direction(self.AXIS)

    def face(self, which: str) -> MountFace:
        """World MountFace of a connector ('face' is the horn), for building parts around it."""
        comp = self.horn if which == "face" else self.case
        T = comp.pose @ comp.connectors[which]
        return MountFace(T[:3, 3].copy(), T[:3, 2].copy())


def servo(name: str, spec: ServoSpec) -> ServoUnit:
    return ServoUnit(name, spec)


class MotorUnit:
    """A cylindrical joint motor as two components sharing one frame, like a servo:
    `case` (the stator, bolted to one link) and `horn` (the output flange, bolted to the other).
    +X is the output axis. Case connectors: back (the -X face), front (the +X face, a ring round
    the flange). Horn connector: face (the flange's outer face)."""
    AXIS = np.array([1.0, 0, 0])

    def __init__(self, name: str, spec, reduction: float = 1.0, mode: str = "torque",
                 armature: float | None = None):
        from .components.motor import MotorGeometry
        self.name, self.spec = name, spec
        g, s = MotorGeometry(spec), spec
        f = s.flange_mass_fraction
        self.case = Component(f"{name}_case", g.case(), BOUGHT, (0.12, 0.12, 0.13, 1.0), mass_override=s.mass * (1 - f))
        self.horn = Component(f"{name}_flange", g.flange(), BOUGHT, (0.7, 0.72, 0.75, 1.0), mass_override=s.mass * f)
        X, Z = np.eye(3)[0], np.eye(3)[2]
        self.case.connector("back", (-s.length / 2, 0, 0), -X, Z, check=False)
        self.case.connector("front", (s.length / 2, 0, s.flange_diameter / 2 + 1), X, Z, check=False)
        self.horn.connector("face", (s.length / 2 + s.flange_thickness, 0, 0), X, Z, check=False)
        self.default_actuator = spec.actuator(reduction, mode, armature)
        self._reduction = reduction

    def drive(self, reduction: float = 1.0, mode: str | None = None, armature: float | None = None) -> ActuatorModel:
        """The actuator this motor gives a joint it drives remotely, through a belt or rod with
        an extra `reduction` (mount the motor with `asm.lock(motor)`, then pass this to
        `asm.hinge(..., actuator=motor.drive(1.9))`)."""
        return self.spec.actuator(self._reduction * reduction, mode or self.default_actuator.mode, armature)

    placed = ServoUnit.placed
    anchor = ServoUnit.anchor
    axis = ServoUnit.axis
    face = ServoUnit.face


def motor(name: str, spec, *, reduction: float = 1.0, mode: str = "torque", armature: float | None = None) -> MotorUnit:
    """A joint motor from a datasheet `spec`; `reduction` is any extra stage after it (a belt,
    a linkage), which multiplies torque and reflected inertia and divides speed."""
    return MotorUnit(name, spec, reduction, mode, armature)


# ---- the assembly ---------------------------------------------------------------------

@dataclass
class _Edge:
    a: Component
    b: Component
    kind: str                       # "fixed" | "joint"
    joint: dict | None = None


class Assembly:
    def __init__(self, name: str, root: Component, floating: bool = True, root_link: str = "trunk"):
        if root.pose is None:
            root.pose = np.eye(4)
        self.name, self.root, self.floating, self.root_link = name, root, floating, root_link
        self.components: dict[str, Component] = {root.name: root}
        self.edges: list[_Edge] = []

    # -- helpers
    def _register(self, c: Component):
        other = self.components.get(c.name)
        if other is not None and other is not c:
            raise AttachmentError(f"two different parts are named {c.name!r}")
        self.components[c.name] = c

    def _require_placed(self, c: Component, what: str):
        if c.pose is None or c.name not in self.components:
            raise AttachmentError(f"{what}: {c.name} is not in the assembly yet; connect from the root outward")

    def _agree(self, comp: Component, pose: np.ndarray, how: str):
        dp = float(np.linalg.norm(comp.pose[:3, 3] - pose[:3, 3]))
        R = comp.pose[:3, :3].T @ pose[:3, :3]
        da = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        if dp > AGREE_MM or da > AGREE_DEG:
            raise AttachmentError(f"{comp.name} is already placed, and {how} would put it {dp:.2f} mm / {da:.2f} "
                                  f"degrees away: the two connections disagree")

    # -- connections
    def fasten(self, a: Conn, b: Conn, *, spin: tuple | None = None, at=None) -> Component:
        """Put b's connector face flat on a's. `spin=(b_vector, world_dir)` turns b about
        the shared normal so a vector in b's own frame points as close to world_dir as it can.
        `at` (world mm) seats b at another point of a's face instead of at a's connector:
        any coordinate along the normal is ignored, and the point must be on a's surface,
        so the parts still touch."""
        self._require_placed(a.comp, f"fasten({a}, {b})")
        Wa = a.comp.pose @ a.frame_local()
        if at is not None:
            n = Wa[:3, 2]
            target = np.asarray(at, float)
            target = target - np.dot(target - Wa[:3, 3], n) * n      # onto the face plane
            d = a.comp.world_shape.distance_to(Vertex(*target))
            if d > AGREE_MM:
                raise AttachmentError(f"fasten({a}, {b}): at={target.round(2).tolist()} is {d:.2f} mm off "
                                      f"{a.comp.name}'s surface")
            Wa = Wa.copy()
            Wa[:3, 3] = target
        Wb = Wa @ _rot("x", np.pi)                      # opposed normals, matching x
        if spin is not None:
            v_local, target = np.asarray(spin[0], float), unit(spin[1])
            Cb = b.frame_local()
            v_conn = np.linalg.inv(Cb)[:3, :3] @ v_local   # b's vector in its connector frame
            v_world = Wb[:3, :3] @ v_conn
            n = Wb[:3, 2]
            p, q = v_world - np.dot(v_world, n) * n, target - np.dot(target, n) * n
            if np.linalg.norm(p) < 1e-6 or np.linalg.norm(q) < 1e-6:
                raise AttachmentError(f"spin for {b}: the vector or direction is along the face normal")
            ang = np.arctan2(np.dot(np.cross(unit(p), unit(q)), n), np.dot(unit(p), unit(q)))
            Wb = Wb @ _rot("z", ang)
        pose = Wb @ np.linalg.inv(b.frame_local())
        if b.comp.pose is None:
            b.comp.pose = pose
            b.comp._world_shape = None
        else:
            self._agree(b.comp, pose, f"fasten({a}, {b})")
        self._register(b.comp)
        self.edges.append(_Edge(a.comp, b.comp, "fixed"))
        return b.comp

    def attach(self, base: Component, part: Component) -> Component:
        """Attach a part drawn in world coordinates; it must touch `base` (checked now)."""
        self._require_placed(base, f"attach({base.name}, {part.name})")
        if part.pose is None:
            raise AttachmentError(f"attach: {part.name} has no position; make it with Component.world(...) "
                                  f"or connect it with fasten(...)")
        a, b = base.world_shape, part.world_shape
        if getattr(a, "_mesh", None) is not None or getattr(b, "_mesh", None) is not None:
            # covers from `enclose` are faceted: measure on their exact meshes (much faster)
            from .shells import _to_manifold
            ma, mb = _to_manifold(a), _to_manifold(b)
            gap = 0.0 if (ma ^ mb).volume() > 1e-3 else ma.min_gap(mb, 10 * TOUCH_MM)
        else:
            gap = a.distance_to(b)
            if gap > TOUCH_MM and _overlap(a, b):
                gap = 0.0                               # one sits inside the other: that touches too
        if gap > TOUCH_MM:
            raise AttachmentError(f"{part.name} does not touch {base.name}: the gap is {gap:.2f} mm")
        self._register(part)
        self.edges.append(_Edge(base, part, "fixed"))
        return part

    def joint(self, name: str, unit_: ServoUnit, range: tuple[float, float], *, reverse: bool = False,
              child_link: str | None = None, actuator: ActuatorModel | None = None) -> None:
        """Make a servo a joint: its case and horn turn about its output axis. Whichever
        of the two is already placed is the parent side. reverse=True flips the axis sign."""
        case, horn = unit_.case, unit_.horn
        placed = [c for c in (case, horn) if c.pose is not None and c.name in self.components]
        if not placed:
            raise AttachmentError(f"joint {name}: connect the case or the horn of {unit_.name} first")
        if len(placed) == 2:
            raise AttachmentError(f"joint {name}: both sides of {unit_.name} are already placed (closed loop)")
        parent = placed[0]
        child = horn if parent is case else case
        child.pose = parent.pose.copy()
        child._world_shape = None
        self._register(child)
        axis = parent.direction(unit_.AXIS) * (-1 if reverse else 1)
        unit_._joint = dict(range=tuple(range), sign=-1 if reverse else 1, case_parent=parent is case)
        self.edges.append(_Edge(parent, child, "joint", dict(
            name=name, anchor=parent.point((0, 0, 0)), axis=axis, range=tuple(range),
            actuator=actuator or unit_.default_actuator, child_link=child_link or f"{name}_link")))

    def lock(self, unit_: ServoUnit) -> None:
        """A servo that is mounted but not simulated as a joint (a jaw, a gripper you do not
        train): its horn is fixed to its case, so parts can still bolt to the horn."""
        case, horn = unit_.case, unit_.horn
        placed = [c for c in (case, horn) if c.pose is not None and c.name in self.components]
        if not placed:
            raise AttachmentError(f"lock: connect the case or the horn of {unit_.name} first")
        other = horn if placed[0] is case else case
        if other.pose is None:
            other.pose = placed[0].pose.copy()
            other._world_shape = None
        self._register(other)
        self.edges.append(_Edge(case, horn, "fixed"))

    def hinge(self, name: str, a: Component, b: Component, at, axis, range: tuple[float, float] | None = None, *,
              actuator: ActuatorModel | None = None, stiffness: float = 0.0, springref: float = 0.0,
              damping: float | None = None, armature: float | None = None, pin: float = 3.0,
              child_link: str | None = None) -> None:
        """A pin between two parts, about `axis` through the world point `at` (mm), with no
        servo: passive unless you give an `actuator`. The pin's axis must pass through both parts
        (within `pin` mm, so clevis forks and bored holes count; checked).
        If `b` is not connected yet it becomes the moving side (`range` required). If `b` is
        already connected, the pin closes a loop (a four-bar, a parallel gripper): it is kept
        as a constraint (MJCF `equality connect`) and the joint range does not apply."""
        at, axis = np.asarray(at, float), unit(axis)
        self._require_placed(a, f"hinge {name}")
        if b.pose is None:
            raise AttachmentError(f"hinge {name}: {b.name} has no position; draw it with Component.world(...)")
        # The pin's axis must pass through both parts (within a pin radius: forks and holes count).
        from build123d import Edge
        line = Edge.make_line(tuple(at - axis * 200), tuple(at + axis * 200))
        for c in (a, b):
            gap = c.world_shape.distance_to(line)
            if gap > pin + TOUCH_MM:
                raise AttachmentError(f"hinge {name}: the pin axis through {at.round(2).tolist()} misses {c.name} "
                                      f"by {gap:.2f} mm")
        if b.name in self.components:
            self.edges.append(_Edge(a, b, "loop", dict(name=name, anchor=at)))
            return
        if range is None:
            raise AttachmentError(f"hinge {name}: give the joint range")
        self._register(b)
        self.edges.append(_Edge(a, b, "joint", dict(
            name=name, anchor=at, axis=unit(axis), range=tuple(range), actuator=actuator,
            child_link=child_link or f"{name}_link",
            dynamics=dict(stiffness=stiffness, springref=springref, damping=damping, armature=armature, pin=pin))))

    def rotor(self, name: str, part: Component, at, axis=(0, 0, 1.0), *, spin: int, thrust_max: float,
              torque_ratio: float, source: str = "") -> None:
        """A motor and propeller on `part`: thrust (0..thrust_max N) along `axis` through the world
        point `at`, and a reaction torque of torque_ratio * thrust about the axis, against the
        spin. spin=+1 turns counter-clockwise seen from above. The point must be on the part."""
        self._require_placed(part, f"rotor {name}")
        at = np.asarray(at, float)
        if part.world_shape.distance_to(Vertex(*at)) > TOUCH_MM and not part.world_shape.is_inside(tuple(at)):
            raise AttachmentError(f"rotor {name}: {at.round(2).tolist()} is not on {part.name}")
        if spin not in (1, -1):
            raise ValueError(f"rotor {name}: spin must be +1 or -1")
        part.sites[name] = tuple(np.linalg.inv(part.pose) @ np.append(at, 1.0))[:3]
        self.rotors = getattr(self, "rotors", []) + [dict(name=name, part=part, pos=at, axis=unit(axis), spin=spin,
                                                         thrust_max=thrust_max, torque_ratio=torque_ratio,
                                                         source=source)]

    def imu(self, site: str, kinds=("gyro", "accelerometer", "framequat"), noise: dict | None = None,
            cutoff: dict | None = None) -> None:
        """IMU sensors at a site (declared on a part's `sites`). Add "framelinvel",
        "frameangvel" and "framepos" for the body velocities and position RL policies observe.
        `noise` and `cutoff` map a kind to its MJCF noise (std dev) and cutoff."""
        self.sensors = getattr(self, "sensors", []) + [
            (k, site, {kk: vv for kk, vv in (("noise", (noise or {}).get(k)), ("cutoff", (cutoff or {}).get(k)))
                       if vv is not None}) for k in kinds]

    def touch(self, site: str, radius: float) -> None:
        """A contact (touch) sensor: a sphere of `radius` mm round a site, e.g. a foot."""
        self.sensors = getattr(self, "sensors", []) + [("touch", site)]
        self.touch_radius = {**getattr(self, "touch_radius", {}), site: radius}

    def couple(self, follower: str, leader: str, ratio: float = 1.0) -> None:
        """q_follower = ratio * q_leader (one motor driving two joints)."""
        self.couplings = getattr(self, "couplings", []) + [(follower, leader, float(ratio))]

    def chain(self, parent: Component, joints: list, **kw) -> dict:
        """A limb from a joint table (see mechasmith.limbs.ChainJoint): motors on the parent side,
        flanges and struts on the child side, no overlaps by construction. Returns the child
        structures by joint name."""
        from .limbs import build_chain
        return build_chain(self, parent, joints, **kw)

    def link(self, name: str, *, around: ServoUnit | None = None, on: Component | None = None, path=None,
             width: float = 24.0, holds=None, add=(), cut=(), clear=(), wall: float = 2.5, cheek: float = 4.0,
             gap: float = 0.8, round: float = 2.0, material: Material = PLA, color=(0.93, 0.91, 0.86, 1.0),
             collision: bool = False, bearing=None, back_cheek: bool = True) -> Component:
        """A printed link (see mechasmith.links): a fork `around` a servo whose joint is already
        declared (bolted to its horn, clear of its case over the joint range) and/or a walled
        pocket that `holds` the next servo (seated by construction). Without `around`, give
        `on`: the part it is attached to (contact checked). `bearing=(od, id, width)` presses a
        bearing into the back cheek, riding on the servo's rear boss. `back_cheek=False` makes a
        one-sided link (the servo's back is used by something else)."""
        from .links import SWEEP_STEP, build_link
        sweep = None
        if around is not None:
            if around.horn.pose is None or around.horn.name not in self.components:
                raise AttachmentError(f"link {name}: declare the joint of {around.name} first, so the "
                                      f"fork knows how far the servo turns")
            j = getattr(around, "_joint", None)
            if j is not None:
                lo, hi = j["range"]
                q = np.append(np.arange(lo, hi, SWEEP_STEP), hi)
                sweep = (-1 if j["case_parent"] else 1) * j["sign"] * q
        elif on is None:
            raise AttachmentError(f"link {name}: say what it is attached to (around=servo or on=part)")
        shape, hf, seat = build_link(name, around=around, sweep=sweep, path=path, width=width, holds=holds,
                                     add=add, cut=cut, clear=clear, wall=wall, cheek=cheek, gap=gap, round=round,
                                     bearing=bearing, back_cheek=back_cheek)
        c = Component.world(name, shape, material, color, collision=collision)
        if hf is not None:
            c.connector("holds", hf[0], hf[1], hf[2], check=False)
        self.attach(around.horn if around is not None else on, c)
        if seat is not None:
            from .links import bearing as _bearing
            self.attach(c, _bearing(f"{name}_bearing", seat[0], seat[1], seat[2], seat[3], seat[4]))
        if holds is not None:
            self.fasten(c.c("holds"), holds.unit.case.c("back"))
        return c

    # -- output
    def robot(self) -> Robot:
        """Group rigidly connected parts into links and build the Robot."""
        parent = {n: n for n in self.components}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for e in self.edges:
            if e.kind == "fixed":
                parent[find(e.a.name)] = find(e.b.name)
        joints = [e for e in self.edges if e.kind == "joint"]
        group_name = {find(self.root.name): self.root_link}
        for e in joints:
            g = find(e.b.name)
            if g in group_name:
                raise AttachmentError(f"joint {e.joint['name']}: its two sides are also rigidly connected")
            group_name[g] = e.joint["child_link"]
        links: dict[str, Link] = {}
        for c in self.components.values():
            ln = group_name[find(c.name)]
            link = links.setdefault(ln, Link(ln))
            link.add(Part(c.name, c.world_shape, c.material, c.color, c.collision, c.mass_override, c.contact,
                          getattr(c, "mass_budget", None)))
            link.sites.update({k: tuple(c.point(v)) for k, v in c.sites.items()})
        robot = Robot(self.name, links[self.root_link], floating=self.floating)
        for e in joints:
            j = e.joint
            robot.add_joint(j["name"], links[group_name[find(e.a.name)]], links[j["child_link"]],
                            j["anchor"], j["axis"], j["range"], j["actuator"], **j.get("dynamics", {}))
        for e in self.edges:
            if e.kind == "loop":
                la, lb = links[group_name[find(e.a.name)]], links[group_name[find(e.b.name)]]
                if la is lb:
                    raise AttachmentError(f"hinge {e.joint['name']}: {e.a.name} and {e.b.name} are rigidly "
                                          f"connected already, so the pin cannot turn")
                robot.loops.append((la, lb, e.joint["anchor"]))
        for f, l, r in getattr(self, "couplings", []):
            robot.couple(f, l, r)
        from .core import Rotor
        for r in getattr(self, "rotors", []):
            robot.rotors.append(Rotor(r["name"], links[group_name[find(r["part"].name)]], r["pos"], r["axis"],
                                      r["spin"], r["thrust_max"], r["torque_ratio"], r["source"]))
        robot.sensors = list(getattr(self, "sensors", []))
        robot.site_radius = dict(getattr(self, "touch_radius", {}))
        return robot


# ---- generated parts as components --------------------------------------------------

def _local_servo(spec: ServoSpec) -> Servo:
    return Servo("_", spec, (0, 0, 0), axis=(1, 0, 0), body_dir=(0, 0, -1))


def u_bracket(spec: ServoSpec, name: str, reach: float, toward: str = "short_end", **kw) -> Component:
    """U-bracket around a servo, in the servo's frame. `toward` names the case face the
    bridge sits beyond (short_end, long_end, side_pos, side_neg). Connectors: horn (the
    cheek face that bolts to the horn) and base (outer face of the bridge, `reach` mm from
    the axis)."""
    from ._bracket_shapes import u_bracket as _u
    z = {"short_end": (0, 0, 1.0), "long_end": (0, 0, -1.0), "side_pos": (0, 1.0, 0), "side_neg": (0, -1.0, 0)}[toward]
    part, face = _u(_local_servo(spec), z, reach, name=name, **kw)
    c = Component(name, part.shape, part.material, part.color)
    c.connector("horn", (spec.horn_face, 0, 0), (-1, 0, 0), (0, 0, 1), check=False)
    c.connector("base", face.point, face.normal, (1, 0, 0), check=False)
    c.AXIS = np.array([1.0, 0, 0])
    return c


def horn_plate(spec: ServoSpec, name: str, size: tuple[float, float], thickness: float = 2.5, **kw) -> Component:
    """Plate on a horn, in the servo's frame. Connectors: horn, out (the far face;
    its x reference is the servo's short-end direction)."""
    from ._bracket_shapes import horn_plate as _h
    part, face = _h(_local_servo(spec), size, name=name, thickness=thickness, **kw)
    c = Component(name, part.shape, part.material, part.color)
    c.connector("horn", (spec.horn_face, 0, 0), (-1, 0, 0), (0, 0, 1), check=False)
    c.connector("out", face.point, (1, 0, 0), (0, 0, 1), check=False)
    return c


def spacer(name: str, length: float, width: float = 19.0, height: float | None = None, **kw) -> Component:
    """Block of `length` along its own z. Connectors: a (z = 0 face), b (z = length face)."""
    from .geom import box
    h = width if height is None else height
    c = Component(name, box((-width / 2, -h / 2, 0), (width / 2, h / 2, length)), **kw)
    c.connector("a", (0, 0, 0), (0, 0, -1), (1, 0, 0), check=False)
    c.connector("b", (0, 0, length), (0, 0, 1), (1, 0, 0), check=False)
    return c


def l_bracket(name: str, a: MountFace, b: MountFace, b_x=None, **kw) -> Component:
    """L-bracket drawn in world coordinates between two perpendicular, world-aligned
    faces (e.g. a placed horn and where the next horn should go). Connectors: a, b,
    each facing the face it bolts to; `b_x` sets b's reference direction."""
    from ._bracket_shapes import l_bracket as _l
    part = _l(a, b, name=name, **kw)
    c = Component.world(name, part.shape, part.material, part.color)
    c.connector("a", a.point, -np.asarray(a.normal), _any_perp(a.normal), check=False)
    c.connector("b", b.point, -np.asarray(b.normal), _any_perp(b.normal) if b_x is None else b_x, check=False)
    return c


def servo_housing(spec: ServoSpec, name: str, wall: float = 2.5, front_wall: float | None = None,
                  **kw) -> Component:
    """Printed cup that holds a servo case (how most printed arms carry their motors), in
    the servo's frame. It wraps the case on every side; the horn side has a round opening
    for the horn, and its wall (`front_wall`, default min(wall, horn thickness - 0.5)) stays
    clear of anything bolted to the horn.

    Connectors: case (the inside of the back wall: fasten it to the servo case's `back`),
    and the outer faces back, short_end, long_end, side_pos, side_neg, named and oriented
    like the case's own, so `spin=(unit.AXIS, ...)` / `(unit.BODY, ...)` work unchanged.
    Hang the chain on the housing's faces instead of the case's: everything stays touching.
    """
    from build123d import Cylinder, Rot
    from .geom import box
    s, w = spec, float(wall)
    fw = min(w, s.horn_thickness - 0.5) if front_wall is None else float(front_wall)
    if fw >= s.horn_thickness:
        raise ValueError(f"servo_housing {name}: front_wall {fw} would reach past the horn face")
    hx, hy = s.case_half_axial, s.case_width / 2
    outer = box((-hx - w, -hy - w, -s.case_below - w), (hx + fw, hy + w, s.case_above + w))
    inner = box((-hx, -hy, -s.case_below), (hx, hy, s.case_above))
    hole = Pos(hx - 1, 0, 0) * Rot(0, 90, 0) * Cylinder(s.horn_radius + 1.5, fw + 2,
                                                        align=(Align.CENTER, Align.CENTER, Align.MIN))
    c = Component(name, outer - inner - hole, **kw)
    X, Y, Z = np.eye(3)
    c.connector("case", (-hx, 0, 0), X, Z, check=False)
    for nm, at, n, x in (("short_end", (0, 0, s.case_above + w), Z, X), ("long_end", (0, 0, -s.case_below - w), -Z, X),
                         ("side_pos", (0, hy + w, 0), Y, X), ("side_neg", (0, -hy - w, 0), -Y, X),
                         ("back", (-hx - w, 0, 0), -X, Z)):
        c.connector(nm, at, n, x, check=False)
    return c
