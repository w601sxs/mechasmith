"""The showreel: one run through a course of obstacles with a trained jet-pack duck policy.

usage: python course.py <policy.b64 | policy.pkl> [--out reel.gif] [--record reel.npz]

Course (each obstacle placed relative to where the duck is when that stage starts):
  stand; walk on to a low box (it climbs it, crosses, steps down by itself); run at a puddle (it
  leaps it); hop on command; walk up to a tall box, stop at its edge, take off, fly over it and land;
  a moon run; walk on.
The policy runs in the training environment (one obstacle at a time, as in training: the current
one is the one it senses); the replay draws them all on the full mechasmith model with exhaust
plumes and smoke, as play.py does.
"""
import argparse
import base64
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image, ImageDraw

from jet_env import BOX_HALF, PUDDLE_HALF_W, JetDuck

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "microduck_jetpack"))
from smoke import Smoke, nozzles, render as smoke_render  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--out", default="reel.gif")
ap.add_argument("--low", type=float, default=0.10, help="low box (climbed), m")
ap.add_argument("--puddle", type=float, default=0.40, help="puddle (leapt), m")
ap.add_argument("--wall", type=float, default=0.40, help="tall box (flown over), m")
ap.add_argument("--scene", default=str(ROOT / "build/microduck_jetpack/scene.xml"))
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--size", default="480x360")
ap.add_argument("--record", default=None, help="also save qpos, fan throttles, commands and stage labels at 30 fps (.npz, for Blender)")
a = ap.parse_args()

STAND, WALK, RUN = (0, 0, 0, 0, 0, 0), (0.25, 0, 0, 0, 0, 0), (0.8, 0, 0, 0, 0, 0)
HOP, MOON = (0.25, 0, 0, 0, 0, 1), (0.7, 0, 0, 0, 0.8, 0)
TAKEOFF, LAND = (0.4, 0, 0, 0.5, 0, 0), (0.2, 0, 0, 0, 0, 0)
# (label, command, obstacle or None (kind, size m, distance ahead m) or "keep", end condition)
#   end: ("time", s) | ("past", margin m beyond the far edge) | ("near", m from trunk to near edge)
STAGES = [
    ("stand", STAND, None, ("time", 1.0)),
    (f"walk on: {100 * a.low:.0f} cm box", WALK, ("box", a.low, 0.5), ("past", 0.25)),
    (f"run: {100 * a.puddle:.0f} cm puddle", RUN, ("pud", a.puddle, 1.6), ("past", 0.3)),
    ("hop (on command)", HOP, None, ("time", 3.0)),
    (f"walk to a {100 * a.wall:.0f} cm box", WALK, ("box", a.wall, 1.4), ("near", 0.18)),
    ("stop at the edge", STAND, "keep", ("time", 1.0)),
    ("take off", TAKEOFF, "keep", ("past", 0.35)),
    ("land", LAND, "keep", ("time", 2.5)),
    ("moon run (fans carry 80%)", MOON, None, ("time", 4.0)),
    ("walk", WALK, None, ("time", 2.5)),
]
MAX_STAGE_S = 12.0

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
DT = float(env.dt)

NONE_BOX, NONE_PUD = jp.array([50.0, 0.0]), jp.array([60.0, 0.1])
state = reset(jax.random.PRNGKey(1))
info = dict(state.info, obstacle=jp.array(False), puddle=jp.array(False), airborne=jp.array(False), box=NONE_BOX,
            pud=NONE_PUD, boost=jp.zeros(()), td_age=jp.ones(2), passed=jp.array(False), cmd_now=jp.zeros(6),
            thrust=jp.zeros(2), edge=jp.array(False), top=jp.array(False), topped=jp.array(False),
            climbed=jp.array(False), chain=jp.array(False))
state = state.replace(info=info)

placed = []            # every obstacle, for the replay: ("box", centre x, top) / ("pud", centre x, half length)
cur = None             # the current one: (kind, near x, far x)
frames_q, frames_t, frames_l, frames_c, log = [], [], [], [], []
fell = False
for label, cmd, obst, (cond, arg) in STAGES:
    x0 = float(state.pipeline_state.x.pos[0, 0])
    info = dict(state.info)
    if obst == "keep":
        pass
    elif obst is None:
        cur = None
        info.update(obstacle=jp.array(False), puddle=jp.array(False), box=NONE_BOX, pud=NONE_PUD)
    else:
        kind, size, ahead = obst
        near = x0 + ahead
        if kind == "box":
            cx = near + BOX_HALF[0]
            placed.append(("box", cx, size))
            cur = ("box", near, near + 2 * BOX_HALF[0])
            info.update(obstacle=jp.array(True), puddle=jp.array(False), box=jp.array([cx, size]), pud=NONE_PUD)
        else:
            placed.append(("pud", near + size / 2, size / 2))
            cur = ("pud", near, near + size)
            info.update(obstacle=jp.array(False), puddle=jp.array(True), box=NONE_BOX,
                        pud=jp.array([near + size / 2, size / 2]))
        info.update(passed=jp.array(False), topped=jp.array(False), climbed=jp.array(False))
    state = state.replace(info=info)
    t, done_stage = 0.0, False
    while not done_stage and t < MAX_STAGE_S:
        # steer back to the course line (y = 0, heading +x) with the turn-rate command, like a
        # pilot on a joystick; the obstacles span |y| < 0.6
        ps = state.pipeline_state
        y = float(ps.x.pos[0, 1])
        fwd = np.asarray(ps.x.rot[0])
        yaw = np.arctan2(2 * (fwd[0] * fwd[3] + fwd[1] * fwd[2]), 1 - 2 * (fwd[2] ** 2 + fwd[3] ** 2))
        want = np.arctan2(-y, 0.8)
        c = np.array(cmd, np.float32)
        c[2] = np.clip(2.0 * (want - yaw), -0.5, 0.5) if np.linalg.norm(c[:2]) > 0.05 else np.clip(-2.0 * yaw, -0.5, 0.5)
        state = state.replace(info=dict(state.info, command=jp.array(c), step=jp.array(1)))
        action, _ = act(state.obs, jax.random.PRNGKey(0))
        state = step(state, action)
        t += DT
        frames_q.append(np.asarray(state.pipeline_state.q))
        frames_t.append(np.asarray(state.info["thrust"]))
        frames_l.append(label)
        frames_c.append(c)
        if float(state.done):
            fell = True
            break
        x = float(state.pipeline_state.x.pos[0, 0])
        if cond == "time":
            done_stage = t >= arg
        elif cond == "past":
            done_stage = cur is not None and x > cur[2] + arg and bool(np.any(np.asarray(env.foot_contact(state.pipeline_state)))) \
                or (cur is not None and x > cur[2] + arg and float(state.info["cmd_now"][3]) > 0.05)
        elif cond == "near":
            done_stage = cur is not None and cur[1] - x < arg
    p = frames_q[-1][:3]
    log.append({"stage": label, "seconds": round(t, 2), "end_pos": [round(float(v), 2) for v in p],
                "fans": [round(float(v), 2) for v in frames_t[-1]], "timed_out": not done_stage and not fell,
                "fell": fell})
    print(json.dumps(log[-1]), flush=True)
    if fell:
        break
print(json.dumps({"completed": not fell and all(not s["timed_out"] for s in log), "seconds": round(len(frames_q) * DT, 1),
                  "max_height_m": round(float(max(q[2] for q in frames_q)), 2)}))
if a.record:
    idx = np.round(np.arange(0, len(frames_q) * DT, 1 / 30) / DT).astype(int).clip(0, len(frames_q) - 1)
    np.savez(a.record, qpos=np.array(frames_q)[idx], throttle=np.array(frames_t)[idx], fps=30, scene=a.scene,
             obstacles=json.dumps(placed), command=np.array(frames_c)[idx], label=np.array(frames_l)[idx])
    print("recorded", a.record, len(idx), "frames")


def add_plumes(scene, d, sites, throttle):
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


spec = mujoco.MjSpec.from_file(a.scene)
for kind, cx, s in placed:
    if kind == "box":
        b = spec.worldbody.add_body(pos=[cx, 0, s / 2])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF[0], BOX_HALF[1], s / 2], rgba=[0.55, 0.6, 0.7, 1])
    else:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, 0, 0.001], size=[s, PUDDLE_HALF_W, 0.001],
                                rgba=[0.2, 0.5, 0.95, 0.7], contype=0, conaffinity=0)
m = spec.compile()
d = mujoco.MjData(m)
w, h = (int(v) for v in a.size.split("x"))
m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
r = mujoco.Renderer(m, h, w)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 1.7, -20
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[0] = opt.geomgroup[2] = 1
fan_sites = [m.site(n).id for n in ("jet_left_fan", "jet_right_fan")]
every = max(1, int(round(1 / (a.fps * env.dt))))
look = frames_q[0][:3].copy()
smoke = Smoke()
boxes = [(cx, s) for kind, cx, s in placed if kind == "box"]


def ground(x, y):
    z = np.zeros_like(x)
    for cx, s in boxes:
        z = np.where((np.abs(x - cx) < BOX_HALF[0]) & (np.abs(y) < BOX_HALF[1]), np.maximum(z, s), z)
    return z


imgs = []
for k in range(0, len(frames_q), every):
    d.qpos[:] = frames_q[k]
    mujoco.mj_forward(m, d)
    look += 0.3 * (frames_q[k][:3] - look)
    cam.lookat[:] = look
    cam.azimuth = 65 + 15 * np.sin(0.15 * k * env.dt)      # behind and beside: the boxes ahead never block it
    r.update_scene(d, camera=cam, scene_option=opt)
    smoke.step(every * env.dt, nozzles(d, fan_sites, frames_t[k]), ground)
    add_plumes(r.scene, d, fan_sites, frames_t[k])
    img = Image.fromarray(smoke_render(r, smoke))
    ImageDraw.Draw(img).text((12, 10), f"t = {k * env.dt:4.1f} s   {frames_l[k]}   fans {100 * frames_t[k][0]:3.0f}% "
                                       f"{100 * frames_t[k][1]:3.0f}%", fill=(30, 30, 30))
    imgs.append(img)
r.close()
sample = imgs[::max(1, len(imgs) // 12)]
sheet = Image.new("RGB", (w, h * len(sample)))
for i, im in enumerate(sample):
    sheet.paste(im, (0, i * h))
pal = sheet.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print("wrote", a.out, len(imgs), "frames")
