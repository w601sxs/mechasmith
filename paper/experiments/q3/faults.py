"""Inject authoring faults into an MJCF model (paper Q3).

Each fault is the kind of mistake a hand- or agent-written model file contains and that
mechasmith rules out by construction. A fault edits the model through MuJoCo's MjSpec and writes a
new XML; the original is never modified.

    python faults.py <model.xml> <fault> <severity> <out.xml>
    python faults.py --list

Faults (one side only, as an authoring slip would be):
  axis_tilt    a joint axis rotated by `severity` degrees about a perpendicular (180 = flipped sign)
  anchor_shift a joint placed `severity` mm off along the link (the child body and its subtree move)
  com_shift    a part mounted `severity` mm off: the body's centre of mass moves, its joint does not
  mass_scale   a body's mass and inertia multiplied by `severity` (missing or duplicated parts)
  foot_offset  a foot's contact geometry `severity` mm off the foot (a detached sole)

Self-interpenetration is not a fault here: the MJX training model collides robot geoms only
with the floor, so overlapping links have no effect on training. That limits what training-time
checks can see, and is reported in the paper.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Fault:
    kind: str
    target: str        # joint, body or geom name
    severity: float    # degrees, mm or a factor, per kind

    @property
    def tag(self) -> str:
        return f"{self.kind}-{self.target}-{self.severity:g}"


# The Q3 grid for the Microduck training model (examples/rl_microduck/duck_mjx.xml).
# Targets are on the left leg (and the head, which dominates the duck's balance).
MICRODUCK_GRID = [
    *(Fault("axis_tilt", "left_knee", s) for s in (5, 15, 180)),
    *(Fault("anchor_shift", "left_knee_link", s) for s in (3, 10, 20)),
    *(Fault("com_shift", "head", s) for s in (5, 15, 30)),
    *(Fault("mass_scale", "trunk", s) for s in (0.75, 1.25, 1.5)),
    *(Fault("foot_offset", "left_sole_collision", s) for s in (3, 10, 20)),
]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perpendicular(a):
    """A unit vector perpendicular to a, chosen deterministically."""
    a = _unit(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(a, ref))


def _rotate(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    t = np.radians(deg)
    k = _unit(axis)
    v = np.asarray(v, float)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))


def _body_direction(spec_model: mujoco.MjModel, body: str) -> np.ndarray:
    """Unit vector, in the parent's frame, from the parent's origin to this body (along the link)."""
    b = spec_model.body(body)
    d = np.asarray(b.pos, float)
    if np.linalg.norm(d) < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return _unit(d)


def inject(xml_path: str, fault: Fault) -> str:
    """Return the XML of the model with one fault applied."""
    spec = mujoco.MjSpec.from_file(xml_path)
    ref = mujoco.MjModel.from_xml_path(xml_path)
    k, s = fault.kind, fault.severity
    if k == "axis_tilt":
        j = spec.joint(fault.target)
        if s == 180:
            j.axis = -np.asarray(j.axis, float)
        else:
            j.axis = _rotate(j.axis, _perpendicular(j.axis), s)
    elif k == "anchor_shift":
        b = spec.body(fault.target)
        b.pos = np.asarray(b.pos, float) + _body_direction(ref, fault.target) * s / 1000.0
    elif k == "com_shift":
        b = spec.body(fault.target)
        if not b.explicitinertial:
            raise ValueError(f"{fault.target} has no explicit inertial to move")
        b.ipos = np.asarray(b.ipos, float) + np.array([s, 0.0, 0.0]) / 1000.0   # forward: worst for pitch
    elif k == "mass_scale":
        b = spec.body(fault.target)
        b.mass = b.mass * s
        b.inertia = np.asarray(b.inertia, float) * s
        if np.any(np.asarray(b.fullinertia, float)):
            b.fullinertia = np.asarray(b.fullinertia, float) * s
    elif k == "foot_offset":
        g = spec.geom(fault.target)
        g.pos = np.asarray(g.pos, float) + np.array([s, 0.0, 0.0]) / 1000.0
    else:
        raise ValueError(f"unknown fault {k}")
    spec.compile()   # fails loudly if the edit made an invalid model
    return spec.to_xml()


def describe(xml_path: str, fault: Fault) -> dict:
    """What the fault changed, measured on the compiled models at the home keyframe."""
    a = mujoco.MjModel.from_xml_path(xml_path)
    b = mujoco.MjModel.from_xml_string(inject(xml_path, fault))
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    for m, d in ((a, da), (b, db)):
        key = m.keyframe("home").id if m.nkey else None
        if key is not None:
            mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)
    out = {"fault": asdict(fault), "mass_kg": [float(a.body_subtreemass[1]), float(b.body_subtreemass[1])]}
    ax = [float(np.degrees(np.arccos(np.clip(np.dot(da.xaxis[j], db.xaxis[j]), -1, 1))))   # signed: a flip is 180
          for j in range(a.njnt) if a.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    out["max_axis_change_deg"] = max(ax) if ax else 0.0
    out["max_anchor_change_mm"] = float(np.max(np.linalg.norm(da.xanchor - db.xanchor, axis=1)) * 1000)
    out["max_com_change_mm"] = float(np.max(np.linalg.norm(da.xipos - db.xipos, axis=1)) * 1000)
    out["max_geom_change_mm"] = float(np.max(np.linalg.norm(da.geom_xpos - db.geom_xpos, axis=1)) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("target", nargs="?")
    ap.add_argument("severity", nargs="?", type=float)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true", help="print the Microduck grid")
    a = ap.parse_args()
    if a.list:
        for f in MICRODUCK_GRID:
            print(f.tag)
        return
    f = Fault(a.kind, a.target, a.severity)
    xml = inject(a.xml, f)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(xml)
    print(json.dumps(describe(a.xml, f)))


if __name__ == "__main__":
    main()
