"""Limbs from a joint table: write the joints (anchor, axis, range, actuator), get the
structure, correct by construction.

    asm.chain(pelvis, [
        ChainJoint("left_hip_pitch_joint", at=(0, 64.5, -102.7), axis=Y, range=(-2.53, 2.88),
                   actuator=HIP, hub=(38, 45), hub_mass=0.75, mass=1.35),
        ...
    ])

For each joint the builder draws the motor (`hub`: radius, length) on the parent side, centred
on the anchor along the axis, and an output flange on the child side, on the face toward the
next joint. A strut leaves the flange along the axis, then runs to the next joint's anchor, so
it never passes back through the motor it came from. `hub=None` makes the joint a cross piece
(axes that cross or nearly do: waists, wrists, ankles driven from elsewhere), drawn as a clevis:
a pin on one side through the bored cheeks of a yoke on the other, no motor. The yoke is on the
parent, unless the parent is itself the child of a cross piece: then the parent carries both
pins (the spider of a universal joint, or a short ankle link) and the child carries the yoke.
Each side keeps out of a zone around the axis between the cheeks, wider than the struts, so the
two sides only meet at the pin, at every angle; nothing overlaps. `mass` is the child link's total; its structure gets whatever the
motors and other parts in that link do not (see `Component.mass_budget`). `add` puts extra world
shapes (a foot, a hand, a torso) on the child link.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import unit
from .materials import ALUMINIUM, BOUGHT


@dataclass
class ChainJoint:
    name: str
    at: tuple
    axis: tuple
    range: tuple
    actuator: object = None
    hub: tuple | None = (28.0, 40.0)     # motor radius, length (mm); None: a cross piece
    hub_mass: float = 0.0                # kg, the motor
    mass: float | None = None            # kg, the child link in all
    strut: float = 14.0                  # strut radius, mm
    add: tuple = ()                      # extra world shapes on the child link
    collision: str | bool = "capsule"
    contact: dict | None = None
    child_link: str | None = None


def _bar(p0, p1, r):
    from .geom import cyl
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    return None if L < 0.5 else cyl((p0 + p1) / 2, p1 - p0, r, L)


@dataclass(frozen=True)
class _Cross:
    """Clevis sizes for a cross piece (mm)."""
    r: float        # pin radius
    R: float        # cheek radius around the axis
    zone: float     # half-width (along the axis) of the space between the cheeks, and its radius
    t: float = 4.0  # cheek and bridge thickness
    c: float = 0.1  # pin clearance in the bores

    @classmethod
    def size(cls, child_strut: float, parent_strut: float):
        sp = parent_strut or child_strut
        r = 0.4 * min(child_strut, sp)
        R = r + 4.0
        return cls(r, R, max(R, child_strut, sp) + 2.0)


def _perp(v, a):
    """Unit part of v perpendicular to a (a unit); any perpendicular if v lies along a."""
    v = np.asarray(v, float) - np.dot(v, a) * a
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
    return unit(v)


def _plate(center, a, u, along_u: float, across: float, along_a: float):
    """A box with sides along u, a x u and a, centred on `center`."""
    from build123d import Box, Plane
    return Plane(origin=tuple(center), x_dir=tuple(u), z_dir=tuple(a)) * Box(along_u, across, along_a)


def _pin(A, a, k: _Cross):
    from .geom import cyl
    return cyl(A, a, k.r, 2 * (k.zone + k.t))


def _yoke(shape, A, a, u, k: _Cross, width: float):
    """Clear the zone around the axis from `shape` and add the cheeks and the bridge that joins
    them, on the side `u` (perpendicular to the axis) where the rest of the part is."""
    from .geom import cyl
    zone = cyl(A, a, k.zone, 2 * k.zone)
    shape = None if shape is None else shape - zone
    w = max(k.R, width)
    # The bridge reaches 0.5 mm into the zone so it fuses with what the cut left (a plate exactly
    # tangent to the cut is a degenerate boolean); the other side keeps 2 mm inside the zone edge.
    yoke = _plate(A + u * (k.zone + k.t / 2 - 0.25), a, u, k.t + 0.5, 2 * w, 2 * (k.zone + k.t))
    for s in (-1.0, 1.0):
        c = A + a * s * (k.zone + k.t / 2)
        yoke = yoke + cyl(c, a, k.R, k.t) + _plate(c + u * (k.zone + k.t) / 2, a, u, k.zone + k.t, 2 * k.R, k.t)
    yoke = yoke - cyl(A, a, k.r + k.c, 2 * (k.zone + k.t) + 2)                             # the bores
    return yoke if shape is None else shape + yoke


def _toward(joints, k):
    """Where joint k's child link goes: the next anchor, else the middle of its extra shapes."""
    J = joints[k]
    if k + 1 < len(joints):
        return np.asarray(joints[k + 1].at, float)
    if J.add:
        return np.mean([np.array(tuple(s.center()), float) for s in J.add], axis=0)
    return np.asarray(J.at, float) + unit(J.axis)


def _carve_swept(target, shape, joints, ks, inverse: bool = False, samples: int = 9):
    """`target` minus the room `shape` sweeps as the consecutive joints `ks` move over their
    ranges (`shape` on the child of the last; `inverse`: `target` is that child and `shape` is
    fixed above the first). Each pose is subtracted in turn: fusing the copies first is fragile
    (a yoke turned about its own axis coincides with itself)."""
    from itertools import product

    from build123d import Axis
    axes = [(tuple(np.asarray(joints[k].at, float)), tuple(unit(joints[k].axis))) for k in ks]
    fine = 4 * samples - 3                     # each joint alone, finely; together, on a coarse grid
    home = [0.0 if lo <= 0.0 <= hi else (lo + hi) / 2 for lo, hi in (joints[k].range for k in ks)]
    poses = set()
    for i, k in enumerate(ks):
        for q in np.linspace(*joints[k].range, fine):
            poses.add(tuple(home[:i] + [float(q)] + home[i + 1:]))
    if len(ks) > 1:
        poses |= {tuple(map(float, qs)) for qs in product(*[np.linspace(*joints[k].range, samples) for k in ks])}
    for qs in sorted(poses):
        s = shape
        steps = list(zip(axes, qs))
        for (A, a), q in (steps if inverse else reversed(steps)):
            s = s.rotate(Axis(A, a), float(np.degrees(-q if inverse else q)))
        target = target - s
    return target


def _single(shape, what: str):
    if len(shape.solids()) != 1:
        raise ValueError(f"chain: {what} comes apart where the next joint's parts swing past; "
                         f"move the joints apart or narrow their ranges")
    return shape


def build_chain(asm, parent, joints, *, material=ALUMINIUM, color=(0.7, 0.7, 0.7, 1.0),
                motor_color=(0.2, 0.2, 0.2, 1.0)) -> dict:
    from .assembly import Component
    from .geom import cyl
    links, prev, prev_strut = {}, parent, 0.0
    cross = [J.hub is None for J in joints]
    if cross and cross[0]:
        raise ValueError(f"chain: {joints[0].name} is a cross piece (hub=None); a chain must start with a motor")
    # The yoke of cross piece k is on its parent unless the parent is a cross piece's child.
    yoke_on_parent = [cross[k] and not (k > 0 and cross[k - 1]) for k in range(len(joints))]
    yokes = {}                                                         # joint index -> its yoke (on the parent)
    for k, J in enumerate(joints):
        A, a = np.asarray(J.at, float), unit(J.axis)
        nxt = np.asarray(joints[k + 1].at, float) if k + 1 < len(joints) else None
        toward = _toward(joints, k)
        sd = 1.0 if np.dot(toward - A, a) >= -1e-6 else -1.0
        if J.hub is not None:
            r, L = J.hub
            hub = Component.world(f"{J.name.removesuffix('_joint')}_motor", cyl(A, a, r, L), BOUGHT, motor_color,
                                  mass_override=J.hub_mass if J.hub_mass > 0 else None)
            if J.hub_mass <= 0:
                hub.mass_override = 1e-4
            asm.attach(prev, hub)
            base = hub
            flange_c, out = A + a * sd * (L / 2 + 2), A + a * sd * (L / 2 + 4)
            shape = cyl(flange_c, a, min(0.7 * r, r - 3), 4)
        else:
            base = prev
            out = A
            kx = _Cross.size(J.strut, prev_strut)
            shape = _pin(A, a, kx) if yoke_on_parent[k] else None      # this side's pin, or a yoke below
        # The strut runs to the next joint; on the last joint it runs into the extra shapes, so
        # they are held (a hand or foot drawn apart from the joint would otherwise float).
        dest = nxt if nxt is not None else (toward if J.add and J.hub is not None else None)
        pts = [out]
        if dest is not None:
            # Out along the axis, then (if the next joint lies to the side) out past the motor's
            # radius, then to the next joint: the strut never cuts back through its own motor.
            if J.hub is not None:
                pts.append(out + a * sd * (J.strut + 2))
                d = dest - A
                perp = d - np.dot(d, a) * a
                reach = J.hub[0] + J.strut + 2
                if np.linalg.norm(perp) > reach:
                    pts.append(pts[-1] + perp / np.linalg.norm(perp) * reach)
            pts.append(dest)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                seg = _bar(p0, p1, J.strut)
                if seg is not None:
                    shape = seg if shape is None else shape + seg
            for i in range(1, len(pts) - 1):                           # rounded corners
                from build123d import Pos, Sphere
                d0, d1 = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
                if np.linalg.norm(d0) < 0.5 or np.linalg.norm(d1) < 0.5 or np.dot(unit(d0), unit(d1)) > 0.9998:
                    continue                                           # no bend (or no segment): no corner
                # A sphere of exactly the strut radius is tangent to both bars along circles, which
                # makes an invalid solid; 0.02 mm larger, the surfaces cross cleanly.
                shape = shape + Pos(*pts[i]) * Sphere(J.strut + 0.02)
        for extra in J.add:
            shape = extra if shape is None else shape + extra
        if cross[k] and not yoke_on_parent[k]:                       # this side is the yoke
            shape = _yoke(shape, A, a, _perp(toward - A, a), kx, J.strut)
            if k - 1 in yokes:                                         # the far side of a universal joint
                shape = _single(_carve_swept(shape, yokes[k - 1], joints, [k - 1, k], inverse=True),
                                f"{J.name}'s link")
        if k + 1 < len(joints) and cross[k + 1]:                      # the next joint's pin or yoke
            Jn = joints[k + 1]
            An, an = np.asarray(Jn.at, float), unit(Jn.axis)
            kn = _Cross.size(Jn.strut, J.strut)
            if yoke_on_parent[k + 1]:
                came_from = pts[-2] if nxt is not None and len(pts) > 1 else A
                un = _perp(came_from - An, an)
                shape = _yoke(shape, An, an, un, kn, J.strut)
                yokes[k + 1] = _yoke(None, An, an, un, kn, J.strut)
                # The child's pin and the root of its strut turn in the yoke: carve their room.
                near = _pin(An, an, kn)
                tw = _toward(joints, k + 1) - An
                if np.linalg.norm(tw) > 0.5:
                    L = min(float(np.linalg.norm(tw)), kn.zone + 2 * Jn.strut + 10.0)
                    near = near + _bar(An, An + unit(tw) * L, Jn.strut + 0.5)
                if k + 2 < len(joints) and cross[k + 2] and not yoke_on_parent[k + 2]:
                    J2 = joints[k + 2]
                    near = near + _pin(np.asarray(J2.at, float), unit(J2.axis), _Cross.size(J2.strut, Jn.strut))
                shape = _single(_carve_swept(shape, near, joints, [k + 1]), f"{J.name}'s link")
            else:
                shape = shape + _pin(An, an, kn)
        if k + 2 < len(joints) and cross[k + 1] and cross[k + 2] and not yoke_on_parent[k + 2]:
            # Two cross pieces in a row (a universal joint): the far yoke swings around both axes
            # past this link, two joints up; carve its swept room out of this link.
            J2 = joints[k + 2]
            A2, a2 = np.asarray(J2.at, float), unit(J2.axis)
            k2 = _Cross.size(J2.strut, joints[k + 1].strut)
            far = _yoke(None, A2, a2, _perp(_toward(joints, k + 2) - A2, a2), k2, J2.strut)
            shape = _single(_carve_swept(shape, far, joints, [k + 1, k + 2]), f"{J.name}'s link")
        if len(shape.solids()) != 1:                                   # held together: nothing floats
            raise ValueError(f"chain: {J.name}'s link is in {len(shape.solids())} pieces; its extra shapes "
                             f"(add=) must reach its strut or flange")
        child = Component.world(f"{J.name.removesuffix('_joint')}_structure", shape, material, color,
                                collision=J.collision, contact=J.contact)
        child.mass_budget = J.mass
        asm.hinge(J.name, base, child, tuple(A), tuple(a), J.range, actuator=J.actuator,
                  pin=3.0 if J.hub is not None else kx.r + 1.0, cross=cross[k],
                  child_link=J.child_link or f"{J.name.removesuffix('_joint')}_link")
        links[J.name] = child
        prev, prev_strut = child, J.strut
    return links
