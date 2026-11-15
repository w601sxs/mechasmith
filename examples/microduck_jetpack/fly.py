"""Fly the jet-pack Microduck: take off, hover, fly forward, turn, land, as a GIF.

    uv run mechasmith build designs/microduck_jetpack.py      # writes build/microduck_jetpack/
    uv run python examples/microduck_jetpack/fly.py          # writes build/microduck_jetpack/flight.gif

A hand-written cascaded controller, nothing learned:

  position PD (+ feed-forward)  ->  desired thrust vector  ->  desired attitude (and yaw)
  attitude PD on SO(3)          ->  body moments
  allocation                    ->  two fan thrusts and two nacelle tilts

The allocation is mechasmith's hover trim and its Jacobian (`mechasmith.mjcf.hover_trim`):
[thrust, roll, pitch, yaw] = J @ (controls - trim). The tilt servos are commanded as position
targets, so their lag and friction are in the loop. The duck's own servos hold the standing
pose throughout. The exhaust plumes are drawn for fun (their length follows each fan's thrust).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "designs"))

import os
from mechasmith.mjcf import hover_trim  # noqa: E402
from mechasmith.sim import system_inertia  # noqa: E402

DEBUG = bool(os.environ.get("FLY_DEBUG"))


def smooth(t, t0, t1, a, b):
    """Minimum-jerk blend from a (at t0) to b (at t1): position, velocity, acceleration."""
    if t <= t0:
        return a, 0.0 * a, 0.0 * a
    if t >= t1:
        return b, 0.0 * b, 0.0 * b
    T = t1 - t0
    s = (t - t0) / T
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    v = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
    return a + (b - a) * p, (b - a) * v, (b - a) * acc


class Mission:
    """Waypoints: (start, end, position xyz m, yaw rad, label)."""

    def __init__(self, z0):
        up, fwd = 0.8, 1.6
        self.legs = [
            (0.0, 1.0, (0, 0, z0), 0.0, "spool up"),
            (1.0, 3.0, (0, 0, up), 0.0, "take-off"),
            (3.0, 5.0, (0, 0, up), 0.0, "hover"),
            (5.0, 8.5, (fwd, 0, up), 0.0, "forward flight"),
            (8.5, 10.5, (fwd, 0, up), np.pi, "turn"),
            (10.5, 13.5, (0, 0, up), np.pi, "fly back"),
            (13.5, 16.0, (0, 0, z0 - 0.02), np.pi, "land"),
            (16.0, 17.0, (0, 0, z0 - 0.02), np.pi, "landed"),
        ]
        self.duration = self.legs[-1][1]

    def at(self, t):
        prev_p, prev_yaw = np.array(self.legs[0][2], float), 0.0
        for t0, t1, p, yaw, label in self.legs:
            p = np.array(p, float)
            if t < t1:
                pos, vel, acc = smooth(t, t0, t1, prev_p, p)
                y, yd, _ = smooth(t, t0, t1, np.array(prev_yaw), np.array(yaw))
                return pos, vel, acc, float(y), float(yd), label
            prev_p, prev_yaw = p, yaw
        return prev_p, 0 * prev_p, 0 * prev_p, prev_yaw, 0.0, self.legs[-1][4]


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "microduck_jetpack")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, nargs=2, default=(480, 360))
    args = ap.parse_args()
    out = args.out or args.build / "flight.gif"

    import microduck_jetpack as design
    m = mujoco.MjModel.from_xml_path(str(args.build / "scene.xml"))
    d = mujoco.MjData(m)
    _, ctrl0, free, J = hover_trim(m, design.build.home)
    Jinv = np.linalg.pinv(J)
    rot = [i for i in free if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_SITE]
    mass = float(m.body_subtreemass[1])
    g = -m.opt.gravity[2]
    W = mass * g
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)     # standing on the floor, fans off
    mujoco.mj_forward(m, d)
    I = system_inertia(m, d)
    com0 = d.subtree_com[1].copy()
    mission = Mission(com0[2])
    offset = com0 - np.array([0, 0, com0[2]])             # fly the centre of mass; start at x = y = 0

    kp_pos, kd_pos, ki_pos = np.array([3.0, 3.0, 6.0]), np.array([3.0, 3.0, 4.5]), np.array([0.5, 0.5, 2.0])
    kR, kW, kI = 64.0, 16.0, np.array([0.0, 20.0, 0.0])    # wn = 8 rad/s; integral on pitch only
    i_pos, i_att = np.zeros(3), np.zeros(3)
    max_tilt = np.radians(25)

    w_px, h_px = args.size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w_px), max(m.vis.global_.offheight, h_px)
    renderer = mujoco.Renderer(m, h_px, w_px)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 0.8, -10
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    fan_sites = [m.actuator_trnid[i, 0] for i in rot]

    frames, log = [], []
    steps_per_frame = int(round(1.0 / (args.fps * m.opt.timestep)))
    n = int(mission.duration / m.opt.timestep)
    spool = 0.0
    cam_look = d.subtree_com[1].copy()
    for k in range(n):
        t = d.time
        p_ref, v_ref, a_ref, yaw_ref, yawd_ref, label = mission.at(t)
        p_ref = p_ref + offset
        com, R = d.subtree_com[1], d.xmat[1].reshape(3, 3)
        # centre-of-mass velocity (world)
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, 1)
        v = jac @ d.qvel
        e_p, e_v = p_ref - com, v_ref - v
        airborne = label not in ("spool up", "landed")
        if airborne:
            i_pos = np.clip(i_pos + e_p * m.opt.timestep, -0.3, 0.3)
        a_des = kp_pos * e_p + kd_pos * e_v + ki_pos * i_pos + a_ref + np.array([0, 0, g])
        # tilt limit: keep the thrust vector within max_tilt of vertical
        horiz = np.hypot(a_des[0], a_des[1])
        if horiz > a_des[2] * np.tan(max_tilt):
            a_des[:2] *= a_des[2] * np.tan(max_tilt) / horiz
        F = mass * a_des
        zb = F / np.linalg.norm(F)
        xc = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        Rd = np.column_stack([np.cross(yb, zb), yb, zb])
        e_R = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        w = d.qvel[3:6]
        w_d = R.T @ np.array([0, 0, yawd_ref])
        if airborne:
            i_att = np.clip(i_att + e_R * m.opt.timestep, -0.2, 0.2)
        moments = I @ (-kR * e_R - kW * (w - w_d) - kI * i_att)
        thrust = float(F @ R[:, 2])
        # On the ground the duck cannot balance on servo stiffness alone (nor can the plain
        # Microduck: that is what its RL policy learns), so the attitude loop runs from t = 0
        # and the fans spool up fast to take the weight off the legs, then sit at 80 % of
        # hover after touchdown, holding it upright.
        if label == "spool up":
            thrust = 0.95 * W * min(1.0, t / 0.3)
        elif label == "landed":
            thrust = W * max(0.8, 1 - 0.4 * (t - 16.0))
        c = ctrl0.copy()
        c[free] += Jinv @ (np.r_[thrust, moments] - np.array([W, 0, 0, 0]))
        d.ctrl[:] = np.clip(c, lo, hi)
        mujoco.mj_step(m, d)

        if k % steps_per_frame == 0:
            cam_look += 0.3 * (d.subtree_com[1] - cam_look)
            cam.lookat[:] = cam_look + np.array([0, 0, 0.01])
            cam.azimuth = 140 + 25 * np.sin(0.25 * t)
            renderer.update_scene(d, camera=cam, scene_option=opt)
            add_plumes(renderer.scene, m, d, fan_sites, [d.ctrl[i] / hi[i] for i in rot])
            img = Image.fromarray(renderer.render())
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            draw = ImageDraw.Draw(img)
            draw.text((12, 10), f"Microduck jet pack   t = {t:4.1f} s   {label}", fill=(30, 30, 30))
            draw.text((12, 26), f"height {com[2] - com0[2]:+.2f} m   speed {np.linalg.norm(v):.2f} m/s   "
                                f"tilt {tilt:4.1f} deg   fans {100 * d.ctrl[rot[0]] / hi[rot[0]]:3.0f}% "
                                f"{100 * d.ctrl[rot[1]] / hi[rot[1]]:3.0f}%", fill=(30, 30, 30))
            frames.append(img)
        log.append((t, *com, np.linalg.norm(e_p)))
        if k % 250 == 0 and DEBUG:
            print(f"{t:.2f} {label} thrust {thrust:.2f} ctrl {d.ctrl[free].round(3)} com {com.round(3)} e_R {e_R.round(3)}")
    renderer.close()
    pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0,
                   optimize=True)
    for i, tt in enumerate((2.0, 6.5, 9.5)):                    # stills for the README
        frames[int(tt * args.fps)].convert("RGB").save(out.with_name(f"flight_{i}.png"))
    log = np.array(log)
    air = (log[:, 0] > 3.0) & (log[:, 0] < 13.5)
    print(f"wrote {out} ({len(frames)} frames)")
    print(f"max position error in flight: {1000 * log[air, 4].max():.0f} mm; "
          f"peak height {log[:, 3].max() - com0[2]:.2f} m; final height {log[-1, 3] - com0[2]:+.3f} m")


def add_plumes(scene, m, d, sites, throttle):
    """A translucent exhaust plume out of each nozzle, along that nacelle's thrust axis (so it
    tilts with the nacelle), its length and width proportional to the fan's thrust: an outer
    glow and a brighter core, both narrower than the 67 mm nozzle exit."""
    EXIT = 0.047                                        # rotor site -> nozzle exit plane, m
    for s, u in zip(sites, throttle):
        R = d.site_xmat[s].reshape(3, 3)
        axis = R[:, 2]
        for length, radius, rgba in ((0.09 * u, 0.026 * np.sqrt(u), (1.0, 0.55, 0.15, 0.30)),
                                     (0.05 * u, 0.014 * np.sqrt(u), (1.0, 0.85, 0.45, 0.55))):
            if u < 0.02 or scene.ngeom >= scene.maxgeom:
                continue
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([radius, radius, length / 2]),
                                d.site_xpos[s] - axis * (EXIT + length / 2), R.flatten(),
                                np.array(rgba, np.float32))
            scene.ngeom += 1


if __name__ == "__main__":
    main()
