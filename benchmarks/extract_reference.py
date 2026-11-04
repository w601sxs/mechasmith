"""Extract sim-relevant reference numbers from Pollen's Microduck MJCF.

usage: python benchmarks/extract_reference.py <path to microduck_rl clone>
Writes benchmarks/microduck_reference.json. Only numbers are stored (no meshes).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
LEG_BODIES = ["yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left"]  # child body of each left joint

src = Path(sys.argv[1]) / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"
root = ET.parse(src).getroot()
for parent in list(root.iter()):
    for c in list(parent):
        if c.tag in ("geom", "asset", "camera"):
            parent.remove(c)
m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
d = mujoco.MjData(m)
d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
mujoco.mj_forward(m, d)

anchor = {m.joint(i).name: d.xanchor[i].copy() for i in range(1, m.njnt)}
dist = lambda a, b: float(np.linalg.norm(anchor[a] - anchor[b]) * 1000)
foot = d.site_xpos[m.site("left_foot").id]

ref = {
    "source": "pollen-robotics/microduck_rl robot_groundcontact.xml",
    "total_mass_g": float(m.body_subtreemass[m.body("trunk_base").id] * 1000),
    "leg_mass_g": float(m.body_subtreemass[m.body("yaw2roll").id] * 1000),
    "head_neck_mass_g": float(m.body_subtreemass[m.body("neck").id] * 1000),
    "trunk_mass_g": float(m.body_mass[m.body("trunk_base").id] * 1000),
    "hip_yaw_spacing_mm": dist("left_hip_yaw", "right_hip_yaw"),
    "segments_mm": {f"{a}->{b}": dist(f"left_{a}", f"left_{b}") for a, b in zip(LEG, LEG[1:])},
    "hip_pitch_to_foot_mm": float(np.linalg.norm(anchor["left_hip_pitch"] - foot) * 1000),
    "hip_yaw_to_foot_mm": float(np.linalg.norm(anchor["left_hip_yaw"] - foot) * 1000),
    "link_mass_g": {j: float(m.body_mass[m.body(b).id] * 1000) for j, b in zip(LEG, LEG_BODIES)},
    "actuator": {"kp": 0.55, "forcerange": 0.96, "damping": 0.053, "frictionloss": 0.0048, "armature": 0.0018},
}
out = Path(__file__).with_name("microduck_reference.json")
out.write_text(json.dumps(ref, indent=2))
print(json.dumps(ref, indent=2))
