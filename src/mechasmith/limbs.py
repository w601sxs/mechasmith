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
(axes that cross or nearly do: waists, wrists, ankles driven from elsewhere): a short pin on the
child side, no motor drawn. `mass` is the child link's total; its structure gets whatever the
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


def build_chain(asm, parent, joints, *, material=ALUMINIUM, color=(0.7, 0.7, 0.7, 1.0),
                motor_color=(0.2, 0.2, 0.2, 1.0)) -> dict:
    from .assembly import Component
    from .geom import cyl
    links, prev, prev_strut = {}, parent, 0.0
    for k, J in enumerate(joints):
        A, a = np.asarray(J.at, float), unit(J.axis)
        nxt = np.asarray(joints[k + 1].at, float) if k + 1 < len(joints) else None
        toward = nxt if nxt is not None else (np.mean([np.array(tuple(s.center()), float) for s in J.add], axis=0) if J.add else A + a)
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
            shape = cyl(A, a, J.strut * 0.8, 2 * J.strut)            # the cross pin
        if nxt is not None:
            # Out along the axis, then (if the next joint lies to the side) out past the motor's
            # radius, then to the next joint: the strut never cuts back through its own motor.
            pts = [out]
            if J.hub is not None:
                pts.append(out + a * sd * (J.strut + 2))
                d = nxt - A
                perp = d - np.dot(d, a) * a
                reach = J.hub[0] + J.strut + 2
                if np.linalg.norm(perp) > reach:
                    pts.append(pts[-1] + perp / np.linalg.norm(perp) * reach)
            pts.append(nxt)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                seg = _bar(p0, p1, J.strut)
                if seg is not None:
                    shape = shape + seg
            for p in pts[1:-1]:                                        # rounded corners
                from build123d import Pos, Sphere
                shape = shape + Pos(*p) * Sphere(J.strut)
        for extra in J.add:
            shape = shape + extra
        child = Component.world(f"{J.name.removesuffix('_joint')}_structure", shape, material, color,
                                collision=J.collision, contact=J.contact)
        child.mass_budget = J.mass
        asm.hinge(J.name, base, child, tuple(A), tuple(a), J.range, actuator=J.actuator,
                  pin=3.0 if J.hub is not None else max(J.strut, prev_strut) + 1.0,
                  child_link=J.child_link or f"{J.name.removesuffix('_joint')}_link")
        links[J.name] = child
        prev, prev_strut = child, J.strut
    return links
