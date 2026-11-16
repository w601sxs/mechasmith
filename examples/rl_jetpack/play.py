"""Run a trained jet-pack duck policy through a course and render it.

usage: python play.py <policy.b64 | policy.pkl> [--out course.gif] [--box 0.15 | --puddle 0.5]

Course: stand, walk forward to a box (--box m tall, 0.4 m long) and on past it at standing height
(the policy decides how: step, hop or fly), a moon run (fans carry 80 % of the weight), hopping on command, then a flight (climb to 0.6 m, fly forward, hover),
land, and walk on. The policy runs in the training environment (CPU, MJX); the motion is replayed
on the full mechasmith model with the box added, exhaust plumes sized by each fan's thrust.
With --puddle L the course is a long jump instead: run up and over an L m puddle, then the moon
run, flight and landing as before.
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
from PIL import Image, ImageDraw

from jet_env import BOX_HALF, PUDDLE_HALF_W, JetDuck

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "microduck_jetpack"))
from smoke import Smoke, nozzles, render as smoke_render  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--out", default="course.gif")
ap.add_argument("--box", type=float, default=0.15)
ap.add_argument("--puddle", type=float, default=None, help="long-jump course over a puddle this long (m)")
ap.add_argument("--scene", default=str(ROOT / "build/microduck_jetpack/scene.xml"))
ap.add_argument("--fps", type=int, default=15)
a = ap.parse_args()

raw = Path(a.policy).read_bytes()
blob = pickle.loads(base64.b64decode(raw) if a.policy.endswith(".b64") else raw)
if isinstance(blob, dict):
    normalizer, policy = blob["normalizer"], jax.tree.map(lambda x: np.asarray(x, np.float32), blob["policy"])
else:
    normalizer, policy = blob[0], blob[1]

env = JetDuck(kick_vel=0.0)
nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                      preprocess_observations_fn=running_statistics.normalize,
                                      policy_hidden_layer_sizes=(256, 256, 128, 128))
act = jax.jit(ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True))
step, reset = jax.jit(env.step), jax.jit(env.reset)

BOX_X, PUD_NEAR = 0.8, 1.2
PUD = None if a.puddle is None else (PUD_NEAR + a.puddle / 2, a.puddle / 2)
# (command [vx, vy, yaw rate, height above standing, lightness, hop], seconds, label)
SCRIPT = [((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0, "stand"),
          ((0.25, 0.0, 0.0, 0.0, 0.0, 0.0), 9.0, f"walk on: a {100 * a.box:.0f} cm box ahead") if PUD is None else
          ((0.8, 0.0, 0.0, 0.0, 0.0, 0.0), 5.0, f"run and jump: a {100 * a.puddle:.0f} cm puddle"),
          ((0.25, 0.0, 0.0, 0.0, 0.0, 1.0), 4.0, "hop (on command)"),
          ((0.7, 0.0, 0.0, 0.0, 0.8, 0.0), 5.0, "moon run (fans carry 80%)"),
          ((0.0, 0.0, 0.0, 0.5, 0.0, 0.0), 3.0, "take off"),
          ((0.6, 0.0, 0.0, 0.5, 0.0, 0.0), 4.0, "fly forward"),
          ((0.0, 0.0, 0.0, 0.5, 0.0, 0.0), 2.0, "hover"),
          ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 3.0, "land"),
          ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0), 4.0, "walk")]

state = reset(jax.random.PRNGKey(1))
info = dict(state.info, obstacle=jp.array(PUD is None), puddle=jp.array(PUD is not None), airborne=jp.array(False),
            box=jp.array([BOX_X, a.box] if PUD is None else [50.0, 0.0]),
            pud=jp.array(PUD if PUD is not None else (60.0, 0.1)), boost=jp.zeros(()), td_age=jp.ones(2),
            passed=jp.array(False), cmd_now=jp.zeros(6), thrust=jp.zeros(2))
state = state.replace(info=info)
frames_q, frames_t, frames_l, log = [], [], [], []
fell = False
for cmd, secs, label in SCRIPT:
    n = 0
    for _ in range(int(secs / env.dt)):
        info = dict(state.info, command=jp.array(cmd), step=jp.array(1))   # hold the command
        state = state.replace(info=info)
        action, _ = act(state.obs, jax.random.PRNGKey(0))
        state = step(state, action)
        ps = state.pipeline_state
        frames_q.append(np.asarray(ps.q))
        frames_t.append(np.asarray(state.info["thrust"]))
        frames_l.append(label)
        n += 1
        if float(state.done):
            fell = True
            break
    p = frames_q[-1][:3]
    log.append({"segment": label, "command": cmd, "end_pos": [round(float(x), 2) for x in p],
                "fans": [round(float(x), 2) for x in frames_t[-1]], "fell": fell})
    print(json.dumps(log[-1]))
    if fell:
        break
far = BOX_X + BOX_HALF[0] if PUD is None else PUD[0] + PUD[1]
past = max(q[0] for q in frames_q) > far
print(json.dumps({"obstacle": f"box {a.box} m" if PUD is None else f"puddle {a.puddle} m", "got_past": bool(past),
                  "fell": fell,
                  "max_height_m": round(float(max(q[2] for q in frames_q)), 2)}))



def add_plumes(scene, m, d, sites, throttle):
    """Exhaust plumes as in examples/microduck_jetpack/fly.py: along each nacelle's thrust axis,
    sized by the fan's thrust."""
    for s, u in zip(sites, throttle):
        R = d.site_xmat[s].reshape(3, 3)
        axis = R[:, 2]
        for length, radius, rgba in ((0.09 * u, 0.026 * np.sqrt(u), (1.0, 0.55, 0.15, 0.30)),
                                     (0.05 * u, 0.014 * np.sqrt(u), (1.0, 0.85, 0.45, 0.55))):
            if u < 0.02 or scene.ngeom >= scene.maxgeom:
                continue
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([radius, radius, length / 2]),
                                d.site_xpos[s] - axis * (0.047 + length / 2), R.flatten(), np.array(rgba, np.float32))
            scene.ngeom += 1


# Replay on the full model with the box, render with plumes.
spec = mujoco.MjSpec.from_file(a.scene)
if PUD is None:
    box = spec.worldbody.add_body(name="obstacle", pos=[BOX_X, 0, a.box - BOX_HALF[2]])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF), rgba=[0.55, 0.6, 0.7, 1])
else:   # water: a thin translucent slab, drawn only
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[PUD[0], 0, 0.001], size=[PUD[1], PUDDLE_HALF_W, 0.001],
                            rgba=[0.2, 0.5, 0.95, 0.7], contype=0, conaffinity=0)
m = spec.compile()
d = mujoco.MjData(m)
w, h = 480, 360
m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
r = mujoco.Renderer(m, h, w)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 1.1, -12
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[0] = opt.geomgroup[2] = 1
fan_sites = [m.site(n).id for n in ("jet_left_fan", "jet_right_fan")]
every = max(1, int(round(1 / (a.fps * env.dt))))
look = frames_q[0][:3].copy()
imgs = []
smoke = Smoke()


def ground(x, y):
    if PUD is not None:
        return np.zeros_like(x)
    on = (np.abs(x - BOX_X) < BOX_HALF[0]) & (np.abs(y) < BOX_HALF[1])
    return np.where(on, max(a.box, 0.0), 0.0)


for k in range(0, len(frames_q), every):
    d.qpos[:] = frames_q[k]
    mujoco.mj_forward(m, d)
    look += 0.3 * (frames_q[k][:3] - look)
    cam.lookat[:] = look
    cam.azimuth = 120 + 20 * np.sin(0.2 * k * env.dt)
    r.update_scene(d, camera=cam, scene_option=opt)
    smoke.step(every * env.dt, nozzles(d, fan_sites, frames_t[k]), ground)
    add_plumes(r.scene, m, d, fan_sites, frames_t[k])
    img = Image.fromarray(smoke_render(r, smoke))
    ImageDraw.Draw(img).text((12, 10), f"t = {k * env.dt:4.1f} s   {frames_l[k]}   fans {100 * frames_t[k][0]:3.0f}% "
                                       f"{100 * frames_t[k][1]:3.0f}%", fill=(30, 30, 30))
    imgs.append(img)
r.close()
# one palette from frames across the whole course (so the water, plumes and box keep their colours)
sample = imgs[::max(1, len(imgs) // 12)]
sheet = Image.new("RGB", (w, h * len(sample)))
for i, im in enumerate(sample):
    sheet.paste(im, (0, i * h))
pal = sheet.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print("wrote", a.out, len(imgs), "frames")
