"""Which authoring faults does each kind of check catch? (paper, supports Q2/Q3)

    python detect.py [designs...] [--per 5] [--out ../../generated/detect.json]

For each design's exported model (build/<design>/robot.xml) and each fault kind and severity,
inject the fault at `--per` randomly chosen targets (seeded) and record whether it is caught by:
  compile   MuJoCo compiles the model (what writing MJCF/URDF by hand gives you)
  joint     the Articulate-Anything joint metric against the correct model fails on some joint
            (axis error >= 0.25 rad or axis-line distance >= 50 mm)
  sign      a joint's axis points the opposite way from the correct model's (a check that metric lacks)
  geometry  the checker's geometry: a floating part, a detached body or an overlap appears
            that the correct model does not have
mechasmith cannot express these faults (parts are placed only by connection and joint frames are
derived), which is the comparison the paper draws; this script measures the alternatives.

Faults:
  part_offset   a visual part of a non-root body moved by s mm in a random direction
  anchor_shift  a non-root body (with its subtree and joint) moved by s mm in a random direction
  axis_tilt     a joint axis rotated by s degrees about a random perpendicular (180 = flipped)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from checker import check, joint_errors, joint_lines  # noqa: E402

DESIGNS = ["duck_leg", "so101", "robotiq_2f85", "crazyflie2", "microduck", "microduck_jetpack", "go2", "g1"]
GRID = {"part_offset": (1, 3, 10), "anchor_shift": (1, 3, 10), "axis_tilt": (5, 15, 180)}


def _random_dir(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _rotate(v, k, deg):
    t = np.radians(deg)
    v = np.asarray(v, float)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))


def targets(m: mujoco.MjModel, kind: str) -> list[str]:
    root_children = {b for b in range(1, m.nbody) if m.body_parentid[b] == 0}
    if kind == "part_offset":
        return [m.geom(g).name for g in range(m.ngeom) if m.geom(g).name and m.geom_contype[g] == 0
                and m.geom_conaffinity[g] == 0 and m.geom_bodyid[g] not in root_children | {0}]
    if kind == "anchor_shift":
        return [m.body(b).name for b in range(1, m.nbody) if b not in root_children and m.body(b).name]
    return [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def inject(xml: Path, kind: str, target: str, s: float, rng) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml))
    if kind == "part_offset":
        g = spec.geom(target)
        g.pos = np.asarray(g.pos, float) + _random_dir(rng) * s / 1000
    elif kind == "anchor_shift":
        b = spec.body(target)
        b.pos = np.asarray(b.pos, float) + _random_dir(rng) * s / 1000
    else:
        j = spec.joint(target)
        a = np.asarray(j.axis, float) / np.linalg.norm(j.axis)
        if s == 180:
            j.axis = -a
        else:
            k = np.cross(a, _random_dir(rng))
            j.axis = _rotate(a, k / np.linalg.norm(k), s)
    return spec


def run(design: str, per: int, seed: int = 0) -> list[dict]:
    xml = ROOT / "build" / design / "robot.xml"
    m0 = mujoco.MjModel.from_xml_path(str(xml))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    ref_lines = joint_lines(m0, d0)
    base = check(str(xml))
    base_flags = ({f["geom"] for f in base["floating_parts"]}, set(base["detached_bodies"]),
                  {(o["a"], o["b"]) for o in base["overlaps"]})
    rows = []
    for kind, severities in GRID.items():
        pool = targets(m0, kind)
        if not pool:
            continue
        for s in severities:
            rng = np.random.default_rng(seed)
            for target in rng.choice(pool, size=min(per, len(pool)), replace=False):
                spec = inject(xml, kind, str(target), s, rng)
                row = {"design": design, "fault": kind, "severity": s, "target": str(target)}
                try:
                    m = spec.compile()
                    row["compile"] = False
                except Exception:
                    rows.append({**row, "compile": True, "joint": None, "sign": None, "geometry": None})
                    continue
                d = mujoco.MjData(m)
                mujoco.mj_forward(m, d)
                je = joint_errors(joint_lines(m, d), ref_lines, {})["per_joint"]
                row["joint"] = any(not v["success"] for v in je.values())
                row["sign"] = any(not v["same_sign"] for v in je.values())
                # The checker reads files: write the faulty model next to the original (same meshdir).
                with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=xml.parent, delete=True) as fh:
                    fh.write(spec.to_xml())
                    fh.flush()
                    r = check(fh.name)
                flags = ({f["geom"] for f in r["floating_parts"]}, set(r["detached_bodies"]),
                         {(o["a"], o["b"]) for o in r["overlaps"]})
                row["geometry"] = any(new - old for new, old in zip(flags, base_flags))
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for kind, severities in GRID.items():
        for s in severities:
            rs = [r for r in rows if r["fault"] == kind and r["severity"] == s]
            if not rs:
                continue
            rate = lambda k: float(np.mean([bool(r[k]) for r in rs]))
            any_ = float(np.mean([bool(r["compile"] or r["joint"] or r["sign"] or r["geometry"]) for r in rs]))
            out.append({"fault": kind, "severity": s, "n": len(rs), "compile": rate("compile"),
                        "joint": rate("joint"), "sign": rate("sign"), "geometry": rate("geometry"), "any": any_})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("designs", nargs="*", default=DESIGNS)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "generated" / "detect.json")
    a = ap.parse_args()
    rows = [r for d in a.designs for r in run(d, a.per)]
    summary = summarise(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
