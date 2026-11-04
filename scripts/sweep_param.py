"""Sweep one design parameter and report a joint's usable range (agent tool).

usage: python scripts/sweep_param.py designs/duck_leg.py foot_reach left_ankle 12,16,20
"""
import json
import sys
import tempfile
from pathlib import Path

from mechasmith import mjcf
from mechasmith.checks import interference_report
from mechasmith.cli import load_design

design, param, joint = sys.argv[1], sys.argv[2], sys.argv[3]
values = [float(v) for v in sys.argv[4].split(",")]
mod = load_design(Path(design))
for v in values:
    p = dict(mod.P)
    p[param] = v
    with tempfile.TemporaryDirectory() as d:
        scene = mjcf.export(mod.build(p=p), Path(d))
        r = interference_report(scene, steps=40)
        j = r["joints"][joint]
        print(json.dumps({param: v, "usable": j["usable"], "home_hits": len(r["home_collisions"])}), flush=True)
