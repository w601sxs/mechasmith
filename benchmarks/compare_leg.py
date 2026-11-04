"""Compare a built leg against the Microduck reference numbers.

usage: python benchmarks/compare_leg.py build/duck_leg
"""
import json
import sys
from pathlib import Path

import numpy as np

LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]

ref = json.loads(Path(__file__).with_name("microduck_reference.json").read_text())
build = Path(sys.argv[1] if len(sys.argv) > 1 else "build/duck_leg")
man = json.loads((build / "manifest.json").read_text())
side = "left"

anchors = {j["name"].removeprefix(f"{side}_"): np.array(j["anchor_mm"]) for j in man["joints"]}
foot_link = man["bodies"][f"{side}_ankle_link"]
# foot site is stored in the MJCF; recover it from robot.xml
import mujoco
m = mujoco.MjModel.from_xml_path(str(build / "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
base = d.xpos[m.body("trunk").id] * 1000  # world offset applied by auto base height
foot = d.site_xpos[m.site(f"{side}_foot").id] * 1000 - base

rows = []
def row(name, ours, theirs, unit):
    delta = ours - theirs
    rows.append((name, ours, theirs, delta, 100 * delta / theirs if theirs else float("nan"), unit))

for a, b in zip(LEG, LEG[1:]):
    row(f"{a} -> {b}", float(np.linalg.norm(anchors[a] - anchors[b])), ref["segments_mm"][f"{a}->{b}"], "mm")
row("hip_pitch -> foot", float(np.linalg.norm(anchors["hip_pitch"] - foot)), ref["hip_pitch_to_foot_mm"], "mm")
row("hip_yaw -> foot", float(np.linalg.norm(anchors["hip_yaw"] - foot)), ref["hip_yaw_to_foot_mm"], "mm")
leg_mass = sum(v["mass"] for k, v in man["bodies"].items() if k != "trunk") * 1000
row("leg mass (yaw link down)", leg_mass, ref["leg_mass_g"], "g")

print(f"{'quantity':28s} {'ours':>8s} {'microduck':>10s} {'delta':>8s} {'%':>7s}")
for name, o, t, dlt, pct, u in rows:
    print(f"{name:28s} {o:8.1f} {t:10.1f} {dlt:+8.1f} {pct:+6.0f}%  {u}")
(build / "compare_microduck.json").write_text(json.dumps(
    [dict(zip(["quantity", "ours", "microduck", "delta", "pct", "unit"], r)) for r in rows], indent=2))
