"""Compare a built full robot against Microduck's reference numbers.

usage: python benchmarks/compare_microduck.py build/microduck

Joint placement is compared as axis LINES, not anchor points: an anchor can sit
anywhere along its axis, so the meaningful error is how far our axis line is
from Microduck's anchor, plus the angle between the axes.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ref = json.loads(Path(__file__).with_name("microduck_reference.json").read_text())
build = Path(sys.argv[1] if len(sys.argv) > 1 else "build/microduck")
m = mujoco.MjModel.from_xml_path(str(build / "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
anchor = {m.joint(i).name: d.xanchor[i] * 1000 for i in range(m.njnt)}
axis = {m.joint(i).name: d.xaxis[i] for i in range(m.njnt)}


def closest_point(p1, u1, p2, u2):
    """Midpoint of the shortest segment between two lines (where two axes 'cross')."""
    w = p1 - p2
    a, b, c, d_, e = u1 @ u1, u1 @ u2, u2 @ u2, u1 @ w, u2 @ w
    den = a * c - b * b
    s, t = (b * e - c * d_) / den, (a * e - b * d_) / den
    return (p1 + s * u1 + p2 + t * u2) / 2


def hip_origin(anc, ax):
    """Frame origin both robots share: midpoint of the two yaw/roll axis crossings."""
    return sum(closest_point(anc[f"{s}_hip_yaw"], ax[f"{s}_hip_yaw"], anc[f"{s}_hip_roll"], ax[f"{s}_hip_roll"])
               for s in ("left", "right")) / 2


origin = hip_origin(anchor, axis)
rj = ref["joints"]
ref_origin = hip_origin({k: np.array(v["anchor_mm"]) for k, v in rj.items()},
                        {k: np.array(v["axis"]) for k, v in rj.items()})

print(f"{'joint':16s} {'axis-line offset mm':>20s} {'axis angle deg':>15s}")
rows = []
for name, r in ref["joints"].items():
    if name not in anchor:
        continue
    a, u = anchor[name] - origin, axis[name]
    p = np.array(r["anchor_mm"]) - ref_origin
    off = p - a
    dist = float(np.linalg.norm(off - np.dot(off, u) * u))
    ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(u, r["axis"])), -1, 1))))
    sign = "" if np.dot(u, r["axis"]) > 0 else " (sign flipped)"
    rows.append({"joint": name, "axis_line_offset_mm": round(dist, 1), "axis_angle_deg": round(ang, 1)})
    print(f"{name:16s} {dist:20.1f} {ang:15.1f}{sign}")

def subtree(body):
    return float(m.body_subtreemass[m.body(body).id] * 1000)

ours = {"total_g": subtree("trunk"), "leg_g": subtree("left_hip_yaw_link"), "head_neck_g": subtree("neck_link"),
        "trunk_g": float(m.body_mass[m.body("trunk").id] * 1000)}
theirs = {"total_g": ref["total_mass_g"], "leg_g": ref["leg_mass_g"], "head_neck_g": ref["head_neck_mass_g"],
          "trunk_g": ref["trunk_mass_g"]}
print(f"\n{'mass':16s} {'ours':>8s} {'microduck':>10s}")
for k in ours:
    print(f"{k:16s} {ours[k]:8.1f} {theirs[k]:10.1f}  ({100 * (ours[k] / theirs[k] - 1):+.0f}%)")
site = lambda n: d.site_xpos[m.site(n).id] * 1000
stance = abs(site("left_foot")[1] - site("right_foot")[1])
ref_stance = 2 * abs(ref["left_foot_mm"][1]) if "left_foot_mm" in ref else float("nan")
print(f"\nfoot spacing     {stance:8.1f} {ref_stance:10.1f}")
print(f"foot below hips  {origin[2] - site('left_foot')[2]:8.1f} "
      f"{ref_origin[2] - ref['left_foot_mm'][2] if 'left_foot_mm' in ref else float('nan'):10.1f}  (zero pose)")
(build / "compare_microduck.json").write_text(json.dumps({"joints": rows, "mass_ours": ours, "mass_microduck": theirs,
                                                          "ankle_spacing_mm": [stance, ref_stance]}, indent=2))
