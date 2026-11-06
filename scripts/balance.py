"""Static balance at a pose: CoM vs the feet, per-body mass moments (agent tool).

usage: python scripts/balance.py build/duck [pose-name-from-design]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from mechasmith.cli import load_design
from mechasmith.sim import place_on_floor

build = Path(sys.argv[1])
design = load_design(Path(json.loads((build / "report.json").read_text())["design"]))
pose = getattr(design, "POSES", {}).get(sys.argv[2], {}) if len(sys.argv) > 2 else design.HOME
m = mujoco.MjModel.from_xml_path(str(build / "scene.xml"))
d = mujoco.MjData(m)
place_on_floor(m, d, pose)
root = m.jnt_bodyid[0]
com = d.subtree_com[root] * 1000
soles = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] != 0 and 'sole' in (m.geom(g).name or '')]
pts = []
for g in soles:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    pts.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) * 1000)
pts = np.vstack(pts)
lo, hi = pts.min(0), pts.max(0)
print(f"CoM (mm): x {com[0]:.1f}  y {com[1]:.1f}  z {com[2]:.1f}")
print(f"feet x [{lo[0]:.1f}, {hi[0]:.1f}]  centre {(lo[0]+hi[0])/2:.1f}   y [{lo[1]:.1f}, {hi[1]:.1f}]")
print(f"CoM x margin: front {hi[0]-com[0]:.1f}  back {com[0]-lo[0]:.1f}")
print("mass moments about the feet centre (g*mm, + = forward):")
xc = (lo[0] + hi[0]) / 2
for b in range(1, m.nbody):
    mom = m.body_mass[b] * 1000 * (d.xipos[b][0] * 1000 - xc)
    if abs(mom) > 200:
        print(f"  {m.body(b).name:22s} {m.body_mass[b]*1000:6.1f} g  x {d.xipos[b][0]*1000:7.1f}  moment {mom:+9.0f}")
ank = [d.xanchor[m.joint(n).id] * 1000 for n in ("left_ankle", "right_ankle") if n in [m.joint(i).name for i in range(m.njnt)]]
if ank:
    ax = np.mean([a[0] for a in ank])
    print(f"ankle axes x {ax:.1f}  ->  CoM is {com[0]-ax:+.1f} mm from the ankle axis (+ = in front)")
