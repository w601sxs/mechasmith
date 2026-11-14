"""Run a trained duck policy and render it.

usage: python play.py <policy.b64 | policy.pkl> [--out walk.gif]
The policy is the job's reported chunks joined into one base64 file (see README), or
train.py's policy.pkl. The policy runs in the training environment (CPU) through a script of
commands (stand, walk forward, turn, walk back, sidestep); the motion is replayed on the full
mechasmith model (build/microduck/scene.xml) and saved as a GIF, with tracking numbers printed.
"""
import argparse
import base64
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image

from duck_env import DuckJoystick

ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--out", default="walk.gif")
ap.add_argument("--scene", default=str(Path(__file__).parents[2] / "build/microduck/scene.xml"))
a = ap.parse_args()

raw = Path(a.policy).read_bytes()
blob = pickle.loads(base64.b64decode(raw) if a.policy.endswith(".b64") else raw)
if isinstance(blob, dict):
    normalizer, policy = blob["normalizer"], jax.tree.map(lambda x: np.asarray(x, np.float32), blob["policy"])
else:
    normalizer, policy = blob[0], blob[1]

env = DuckJoystick(kick_vel=0.0)
nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                      preprocess_observations_fn=running_statistics.normalize,
                                      policy_hidden_layer_sizes=(128, 128, 128, 128))
act = jax.jit(ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True))
step, reset = jax.jit(env.step), jax.jit(env.reset)

SCRIPT = [((0.0, 0.0, 0.0), 1.0), ((0.15, 0.0, 0.0), 4.0), ((0.1, 0.0, 0.6), 3.0), ((-0.08, 0.0, 0.0), 2.0),
          ((0.0, 0.08, 0.0), 2.0), ((0.0, 0.0, 0.0), 1.0)]
state = reset(jax.random.PRNGKey(0))
qpos, log = [], []
for cmd, secs in SCRIPT:
    v_sum, n = np.zeros(3), 0
    for _ in range(int(secs / env.dt)):
        info = dict(state.info, command=jp.array(cmd), step=jp.array(1))   # hold the command (no resampling)
        state = state.replace(info=info)
        action, _ = act(state.obs, jax.random.PRNGKey(0))
        state = step(state, action)
        ps = state.pipeline_state
        R = np.asarray(ps.xmat[1]).reshape(3, 3) if hasattr(ps, "xmat") else None
        v_world = np.asarray(ps.xd.vel[0])
        w_world = np.asarray(ps.xd.ang[0])
        rot = np.asarray(ps.x.rot[0])
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, rot)
        Rm = Rm.reshape(3, 3)
        v_sum += np.array([*(Rm.T @ v_world)[:2], (Rm.T @ w_world)[2]])
        n += 1
        qpos.append(np.asarray(ps.q))
        if float(state.done):
            break
    log.append({"command": cmd, "seconds": secs, "measured": [round(x, 3) for x in v_sum / max(n, 1)],
                "fell": bool(float(state.done))})
    print(json.dumps(log[-1]))
    if float(state.done):
        break

# Replay on the full model and render every other frame (25 fps).
m = mujoco.MjModel.from_xml_path(a.scene)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 360, 480)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 0.7, -15, 140
frames = []
for q in qpos[::2]:
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.1]
    r.update_scene(d, cam)
    frames.append(Image.fromarray(r.render()))
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=40, loop=0)
print("wrote", a.out, len(frames), "frames")
