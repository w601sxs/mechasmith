"""Sweep one design parameter and report CoM-vs-ankle offset at HOME (agent tool).

usage: python scripts/sweep_balance.py designs/duck.py battery_x 20,30,40
"""
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from mechasmith import mjcf
from mechasmith.cli import load_design
from mechasmith.sim import place_on_floor

design, param = sys.argv[1], sys.argv[2]
mod = load_design(Path(design))
for v in [float(x) for x in sys.argv[3].split(",")]:
    p = dict(mod.P)
    p[param] = v
    with tempfile.TemporaryDirectory() as d:
        scene = mjcf.export(mod.build(p=p), Path(d))
        m = mujoco.MjModel.from_xml_path(str(scene))
        dd = mujoco.MjData(m)
        place_on_floor(m, dd, mod.HOME)
        ax = np.mean([dd.xanchor[m.joint(n).id][0] for n in ("left_ankle", "right_ankle")])
        print(f"{param}={v:g}: CoM - ankle = {(dd.subtree_com[1][0] - ax) * 1000:+.1f} mm, "
              f"mass {m.body_subtreemass[1] * 1000:.0f} g", flush=True)
