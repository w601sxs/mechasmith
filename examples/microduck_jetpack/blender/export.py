"""Export a recorded jet-pack run for Blender: the visual geoms (meshes as .obj, primitives by
type and size), their colours, every geom's pose per frame, and each fan's nozzle exit, thrust
axis and throttle per frame.

usage: python export.py run.npz OUT_DIR [--box X TOP] [--puddle X HALF]

Obstacles saved in the recording (rl_jetpack/course.py --record) are exported as they are; --box
and --puddle add one more (for play.py and fly.py recordings, which don't save theirs).

run.npz comes from `fly.py --record run.npz` (or `rl_jetpack/play.py --record`): qpos per frame,
fan throttles, fps and the scene path. Then render with blender/render.py.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

EXIT = 0.047                            # fan site -> nozzle exit plane, m (as in fly.py)

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out", type=Path)
ap.add_argument("--box", type=float, nargs=2, default=None, metavar=("X", "TOP"))
ap.add_argument("--puddle", type=float, nargs=2, default=None, metavar=("X", "HALF"))
a = ap.parse_args()

rec = np.load(a.run, allow_pickle=True)
m = mujoco.MjModel.from_xml_path(str(rec["scene"]))
d = mujoco.MjData(m)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "meshes").mkdir(exist_ok=True)

G = mujoco.mjtGeom
TYPES = {int(G.mjGEOM_BOX): "box", int(G.mjGEOM_SPHERE): "sphere", int(G.mjGEOM_CYLINDER): "cylinder",
         int(G.mjGEOM_CAPSULE): "capsule", int(G.mjGEOM_ELLIPSOID): "ellipsoid", int(G.mjGEOM_MESH): "mesh"}
geoms, ids = [], []
for g in range(m.ngeom):
    if m.geom_group[g] not in (0, 2) or int(m.geom_type[g]) not in TYPES or m.geom_rgba[g, 3] == 0:
        continue
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    e = dict(name=m.geom(g).name or f"geom{g}", type=TYPES[int(m.geom_type[g])], size=m.geom_size[g].tolist(),
             rgba=[float(x) for x in rgba])
    if e["type"] == "mesh":
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        path = a.out / "meshes" / f"{k}.obj"
        if not path.exists():           # mesh vertices are already in the geom's frame
            with open(path, "w") as fh:
                fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
                fh.writelines(f"f {i + 1} {j + 1} {l + 1}\n" for i, j, l in f)
        e["mesh"] = f"meshes/{k}.obj"
    geoms.append(e)
    ids.append(g)

fans = [m.actuator_trnid[i, 0] for i in range(m.nu) if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_SITE)]
eye_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_camera")   # the lens in the head
F = len(rec["qpos"])
pose = np.zeros((F, len(ids), 7))
noz = np.zeros((F, len(fans), 7))
eye = np.zeros((F, 7))                  # head camera: position, quat (x forward, z up)
quat = np.zeros(4)
for t in range(F):
    d.qpos[:] = rec["qpos"][t]
    mujoco.mj_forward(m, d)
    if eye_site >= 0:
        mujoco.mju_mat2Quat(quat, d.site_xmat[eye_site])
        eye[t] = np.r_[d.site_xpos[eye_site], quat]
    for j, g in enumerate(ids):
        mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
        pose[t, j] = np.r_[d.geom_xpos[g], quat]
    for j, s in enumerate(fans):
        axis = d.site_xmat[s].reshape(3, 3)[:, 2]
        noz[t, j] = np.r_[d.site_xpos[s] - axis * EXIT, axis, rec["throttle"][t][j]]
np.save(a.out / "pose.npy", pose)
np.save(a.out / "nozzles.npy", noz)
if eye_site >= 0:
    np.save(a.out / "eye.npy", eye)
# obstacles: ["box", centre x, top] / ["pud", centre x, half length] (course.py --record), or --box/--puddle
obstacles = json.loads(str(rec["obstacles"])) if "obstacles" in rec.files else []
if a.box:
    obstacles.append(["box", *a.box])
if a.puddle:
    obstacles.append(["pud", *a.puddle])
json.dump(dict(fps=int(rec["fps"]), frames=F, geoms=geoms, obstacles=obstacles),
          open(a.out / "scene.json", "w"), indent=1)
print(f"exported {len(geoms)} geoms x {F} frames, {len(fans)} fans -> {a.out}")
