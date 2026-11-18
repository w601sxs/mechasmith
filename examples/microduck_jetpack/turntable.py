"""A scripted "getting ready" clip of the jet-pack Microduck for a turntable render: it stands
and looks about, runs a pre-flight check of the nacelles, blips the fans, crouches, then spools
the fans up and strains against the floor. Not simulated: the poses are set by hand and the feet
are kept on the floor; the thrust only drives the smoke in the render.

    uv run python examples/microduck_jetpack/turntable.py build/turntable.npz [--seconds 10]
    uv run python examples/microduck_jetpack/blender/export.py build/turntable.npz build/blender_tt
    blender -b -P examples/microduck_jetpack/blender/render.py -- build/blender_tt build/blender_tt_render \
        --orbit --glow --size 720 720

The .npz has the same keys as fly.py --record, so export.py takes it as it is.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--scene", type=Path, default=ROOT / "build/microduck_jetpack/scene.xml")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(a.scene))
d = mujoco.MjData(m)
home = m.keyframe("home").qpos.copy()
Q = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
foot = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]


def smooth(t, t0, t1):
    """0 before t0, 1 after t1, smoothstep between."""
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return x * x * (3 - 2 * x)


def bump(t, t0, t1):
    """0 -> 1 -> 0 over t0..t1."""
    return np.sin(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1)) ** 2


def feet_z(q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[foot, 2].copy()


z_feet = feet_z(home).mean()
flat = [d.site_xmat[f].reshape(3, 3)[2, 0] for f in foot]


def crouch_legs(q, c):
    """bend both legs by c: hip and knee fold, then each ankle is solved so its sole stays flat."""
    for side, s in (("left", 1), ("right", -1)):
        q[Q[f"{side}_hip_pitch"]] += -s * 0.35 * c
        q[Q[f"{side}_knee"]] += -s * 0.75 * c
    for k, side in enumerate(("left", "right")):
        j = Q[f"{side}_ankle"]
        for _ in range(4):                          # Newton on the sole's pitch (x axis z component)
            e0 = sole_tilt(q, k)
            q[j] += 1e-3
            g = (sole_tilt(q, k) - e0) / 1e-3
            q[j] -= 1e-3 + e0 / g
    return q


def sole_tilt(q, k):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xmat[foot[k]].reshape(3, 3)[2, 0] - flat[k]


# which way a positive bend moves the trunk: pick the sign that lowers it (a crouch)
probe = crouch_legs(home.copy(), 0.5)
d_lo = feet_z(probe).mean() - z_feet           # feet rising relative to the trunk = trunk would drop
SIGN = 1.0 if d_lo > 0 else -1.0

T = a.seconds
F = int(T * a.fps)
qpos, thr = np.zeros((F, m.nq)), np.zeros((F, 2))
rng = np.random.default_rng(3)
for i in range(F):
    t = i / a.fps
    u = t / T                                   # 0..1 through the clip
    q = home.copy()
    # --- look about (first half), then eyes front
    look = 1 - smooth(u, 0.5, 0.58)
    q[Q["head_yaw"]] += look * (0.55 * np.sin(2 * np.pi * t / 3.2) * smooth(u, 0.03, 0.1))
    q[Q["head_pitch"]] += look * 0.18 * np.sin(2 * np.pi * t / 2.1 + 1.0)
    q[Q["head_roll"]] += look * 0.12 * np.sin(2 * np.pi * t / 2.7)
    # --- pre-flight check: each nacelle sweeps forward and back, then both together
    q[Q["jet_left_tilt"]] += 0.55 * np.sin(2 * np.pi * (u - 0.18) / 0.08) * (bump(u, 0.18, 0.26))
    q[Q["jet_right_tilt"]] += 0.55 * np.sin(2 * np.pi * (u - 0.26) / 0.08) * (bump(u, 0.26, 0.34))
    both = 0.45 * np.sin(2 * np.pi * (u - 0.34) / 0.06) * bump(u, 0.34, 0.44)
    q[Q["jet_left_tilt"]] += both
    q[Q["jet_right_tilt"]] += both
    # --- crouch and lean in: head down and forward, like a sprinter in the blocks
    c = smooth(u, 0.56, 0.66) * (1 - 0.55 * smooth(u, 0.8, 1.0))   # thrust takes weight: legs ease
    q = crouch_legs(q, SIGN * 0.75 * c)
    q[Q["neck_pitch"]] += -0.25 * c
    q[Q["head_pitch"]] += -0.30 * c
    # --- fans: two blips, then the spool-up, with a flutter as it builds
    blips = 0.16 * bump(u, 0.46, 0.5) + 0.22 * bump(u, 0.52, 0.56)
    spool = smooth(u, 0.66, 0.97) ** 1.6 * 0.92
    flutter = 0.03 * np.sin(2 * np.pi * t * 7.3) * smooth(u, 0.7, 0.8)
    thr[i] = np.clip(blips + spool + flutter + 0.015 * rng.normal(size=2) * (spool > 0.05), 0, 1)
    # --- the trunk buzzes as the fans come up, and the feet stay on the floor
    shake = 0.0012 * spool * rng.normal(size=3)
    q[:3] += shake
    q[2] += z_feet - feet_z(q).mean()
    qpos[i] = q

np.savez(a.out, qpos=qpos, throttle=thr, fps=a.fps, scene=str(a.scene))
print(f"wrote {F} frames ({T:.0f} s) -> {a.out}; crouch sign {SIGN:+.0f}, peak throttle {thr.max():.2f}")
