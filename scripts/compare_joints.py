"""Compare a built design with a reference MJCF joint by joint (agent tool).

usage: python scripts/compare_joints.py <reference.xml> build/<design> [--site ours=theirs ...]

Joints are matched by name. Placement is compared as axis LINES (an anchor can sit
anywhere along its axis): the distance from our axis line to the reference anchor, the
angle between the axes, and whether the sign agrees. Both models are taken at their
zero pose in their own world frame, so author designs in the reference's frame.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("reference")
ap.add_argument("build", type=Path)
ap.add_argument("--site", action="append", default=[], help="site pairs to compare, ours=theirs")
a = ap.parse_args()


def load(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


rm, rd = load(a.reference)
om, od = load(a.build / "scene.xml")
rows = []
print(f"{'joint':16s} {'axis-line mm':>12s} {'angle deg':>9s} {'sign':>5s} {'range ours':>17s} {'range ref':>17s}")
for i in range(rm.njnt):
    name = rm.joint(i).name
    if rm.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    try:
        j = om.joint(name).id
    except KeyError:
        print(f"{name:16s} missing in ours")
        continue
    u = od.xaxis[j]
    off = (rd.xanchor[i] - od.xanchor[j]) * 1000
    dist = float(np.linalg.norm(off - np.dot(off, u) * u))
    c = float(np.dot(u, rd.xaxis[i]))
    ang = float(np.degrees(np.arccos(np.clip(abs(c), 0, 1))))
    ro, rr = om.jnt_range[j].round(3).tolist(), rm.jnt_range[i].round(3).tolist()
    rows.append({"joint": name, "axis_line_mm": round(dist, 1), "angle_deg": round(ang, 1), "sign_ok": c > 0,
                 "range": ro, "range_ref": rr})
    print(f"{name:16s} {dist:12.1f} {ang:9.1f} {'ok' if c > 0 else 'FLIP':>5s} {str(ro):>17s} {str(rr):>17s}")

mo, mr = om.body_subtreemass[0] * 1000, rm.body_subtreemass[0] * 1000
print(f"\ntotal mass       ours {mo:7.1f} g   ref {mr:7.1f} g   ({100 * (mo / mr - 1):+.0f}%)")
for i in range(rm.njnt):
    name = rm.joint(i).name
    if name in [om.joint(k).name for k in range(om.njnt)]:
        so = om.body_subtreemass[om.jnt_bodyid[om.joint(name).id]] * 1000
        sr = rm.body_subtreemass[rm.jnt_bodyid[i]] * 1000
        print(f"  beyond {name:14s} ours {so:7.1f} g   ref {sr:7.1f} g")
for pair in a.site:
    ours, theirs = pair.split("=")
    po, pr = od.site_xpos[om.site(ours).id] * 1000, rd.site_xpos[rm.site(theirs).id] * 1000
    print(f"site {ours:10s} ours {po.round(1).tolist()} ref {pr.round(1).tolist()}  ({np.linalg.norm(po - pr):.1f} mm apart)")
(a.build / "compare_reference.json").write_text(json.dumps({"joints": rows, "mass_g": [mo, mr]}, indent=2))
