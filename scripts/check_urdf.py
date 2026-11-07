"""Check a build's robot.urdf against its MJCF (agent tool).

usage: python scripts/check_urdf.py build/<design>

1. MuJoCo reads the URDF: every body's mass, centre of mass and inertia, and every
   joint's anchor, axis and range must match the MJCF.
2. yourdfpy (an independent URDF parser) runs forward kinematics: every site frame must
   land where the MJCF puts the site, at the zero pose and at random poses.
Needs yourdfpy for part 2:  uv run --with yourdfpy python scripts/check_urdf.py build/so101
"""
import re
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

build = Path(sys.argv[1])
ref = mujoco.MjModel.from_xml_path(str(build / "scene.xml"))
urdf_text = (build / "robot.urdf").read_text()

# MuJoCo needs its compiler hints inside the URDF; add them to a temporary copy only.
hint = (f'<mujoco><compiler meshdir="{(build / "meshes").resolve()}" strippath="true" '
        f'fusestatic="false" discardvisual="false" balanceinertia="false"/></mujoco>')
with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as t:
    t.write(re.sub(r"(<robot[^>]*>)", r"\1" + hint, urdf_text, count=1))
um = mujoco.MjModel.from_xml_path(t.name)
rd, ud = mujoco.MjData(ref), mujoco.MjData(um)
mujoco.mj_forward(ref, rd)
mujoco.mj_forward(um, ud)
root = ref.body(1).name
lift = rd.xpos[1] - ud.xpos[um.body(root).id]   # floating roots: MJCF lifts the robot, URDF does not

worst = {"mass_g": 0.0, "com_mm": 0.0, "inertia_rel": 0.0, "anchor_mm": 0.0, "axis_deg": 0.0, "range": 0.0}
for i in range(1, ref.nbody):
    n = ref.body(i).name
    j = um.body(n).id
    worst["mass_g"] = max(worst["mass_g"], abs(ref.body_mass[i] - um.body_mass[j]) * 1000)
    c_ref = rd.xipos[i]
    c_u = ud.xipos[j] + lift
    worst["com_mm"] = max(worst["com_mm"], np.linalg.norm(c_ref - c_u) * 1000)
    I_ref = rd.ximat[i].reshape(3, 3) @ np.diag(ref.body_inertia[i]) @ rd.ximat[i].reshape(3, 3).T
    I_u = ud.ximat[j].reshape(3, 3) @ np.diag(um.body_inertia[j]) @ ud.ximat[j].reshape(3, 3).T
    worst["inertia_rel"] = max(worst["inertia_rel"], np.abs(I_ref - I_u).max() / np.abs(I_ref).max())
for i in range(ref.njnt):
    if ref.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    n = ref.joint(i).name
    k = um.joint(n).id
    worst["anchor_mm"] = max(worst["anchor_mm"], np.linalg.norm(rd.xanchor[i] - ud.xanchor[k] - lift) * 1000)
    cosang = np.clip(np.dot(rd.xaxis[i], ud.xaxis[k]), -1, 1)
    worst["axis_deg"] = max(worst["axis_deg"], np.degrees(np.arccos(cosang)))
    worst["range"] = max(worst["range"], np.abs(ref.jnt_range[i] - um.jnt_range[k]).max())
print(f"MuJoCo reads the URDF: {um.nbody - 1} bodies, {um.njnt} joints; worst differences vs MJCF:")
print("  " + ", ".join(f"{k} {v:.2g}" for k, v in worst.items()))
ok = worst["mass_g"] < 0.01 and worst["com_mm"] < 0.01 and worst["inertia_rel"] < 1e-4 \
    and worst["anchor_mm"] < 0.01 and worst["axis_deg"] < 1e-3 and worst["range"] < 1e-6

try:
    import yourdfpy
except ImportError:
    print("yourdfpy not installed: skipped the independent parser check")
    sys.exit(0 if ok else 1)
y = yourdfpy.URDF.load(str(build / "robot.urdf"), mesh_dir=str(build), load_meshes=False)
hinges = [ref.joint(i).name for i in range(ref.njnt) if ref.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
rng = np.random.default_rng(0)
err = 0.0
for trial in range(6):
    q = {} if trial == 0 else {n: rng.uniform(*ref.jnt_range[ref.joint(n).id]) for n in hinges}
    rd.qpos[:] = ref.qpos0
    for n, v in q.items():
        rd.qpos[ref.jnt_qposadr[ref.joint(n).id]] = v
    mujoco.mj_forward(ref, rd)
    y.update_cfg({n: q.get(n, 0.0) for n in y.actuated_joint_names})
    base = rd.xpos[1] - y.get_transform(root, y.base_link)[:3, 3]
    for s in range(ref.nsite):
        p_y = y.get_transform(f"{ref.site(s).name}_frame", y.base_link)[:3, 3] + base
        err = max(err, np.linalg.norm(p_y - rd.site_xpos[s]) * 1000)
print(f"yourdfpy forward kinematics: {ref.nsite} site frames over 6 poses, worst {err:.3g} mm from the MJCF")
sys.exit(0 if ok and err < 0.01 else 1)
