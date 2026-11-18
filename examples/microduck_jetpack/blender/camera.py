"""The chase camera for render.py: fly.py's smoothed follow, which moves when a box would stand
between it and the duck.

Each frame it tries the usual view first. If the sight line from camera to duck passes through a
box, it swings round the duck (and, failing that, climbs) to the nearest view that clears every
box. The view needed is held from a second before the box gets in the way to a second after, and
eased, so the camera glides round instead of snapping.

    keyframe_camera(cam, pose, obstacles, f0, f1, fps)       # inside Blender
"""
import math

import numpy as np

DIST, EL0, EL_MAX = 1.05, 14.0, 60.0           # m; degrees below horizontal
BOX_HALF = (0.2, 0.6)                          # as rl_jetpack's boxes


def _blocked(cam, tgt, boxes, pad=0.03):
    """does the segment cam -> tgt pass through any box (slab test)?"""
    d = tgt - cam
    for lo, hi in boxes:
        t0, t1 = 0.0, 0.97                       # stop short of the duck (it may stand on the box)
        for k in range(3):
            if abs(d[k]) < 1e-9:
                if not (lo[k] - pad <= cam[k] <= hi[k] + pad):
                    break
                continue
            a, b = (lo[k] - pad - cam[k]) / d[k], (hi[k] + pad - cam[k]) / d[k]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 > t1:
                break
        else:
            return True
    return False


def track(pose, obstacles, f0, f1, fps):
    """per-frame (target, camera position) for frames f0..f1 (1-based, inclusive)."""
    boxes = [(np.array([x - BOX_HALF[0], -BOX_HALF[1], 0.0]), np.array([x + BOX_HALF[0], BOX_HALF[1], top]))
             for kind, x, top in obstacles if kind == "box"]
    centre = pose[:, :, :3].mean(axis=1)
    look = centre[f0 - 1].copy()
    # candidate views, cheapest first: swing round the duck before climbing
    cands = sorted(((off, el) for off in range(-180, 181, 10) for el in range(int(EL0), int(EL_MAX) + 1, 4)),
                   key=lambda c: abs(c[0]) / 90 + (c[1] - EL0) / 25)
    looks, base, need = [], [], []
    for t in range(f0 - 1, f1):
        look = look + (1 - 0.7 ** (15 / fps)) * (centre[t] - look)
        az0 = 140 + 25 * math.sin(0.25 * t / fps)
        tgt = look + np.array([0, 0, 0.02])
        pick = cands[0]
        for off, el in cands:
            if not _blocked(tgt - DIST * _fwd(az0 + off, el), tgt, boxes):
                pick = (off, el)
                break
        looks.append(tgt)
        base.append(az0)
        need.append(pick)
    need = np.array(need, float)
    cost = np.abs(need[:, 0]) / 90 + (need[:, 1] - EL0) / 25
    # hold the most demanding view from a second before it is needed to a second after, then ease
    w = int(fps)
    held = np.array([need[max(0, i - w) + int(np.argmax(cost[max(0, i - w):i + w + 1]))] for i in range(len(need))])
    sm = np.empty_like(held)
    sm[0] = held[0]
    k = 1 - 0.93 ** (30 / fps)
    for i in range(1, len(held)):
        sm[i] = sm[i - 1] + k * (held[i] - sm[i - 1])
    out = []
    for tgt, az0, (off, el) in zip(looks, base, sm):
        fwd = _fwd(az0 + off, el)
        out.append((tgt, tgt - DIST * fwd, fwd))
    return out


def _fwd(az_deg, el_deg):
    az, r = math.radians(az_deg), math.radians(-el_deg)
    return np.array([math.cos(r) * math.cos(az), math.cos(r) * math.sin(az), math.sin(r)])


def keyframe_camera(cam, pose, obstacles, f0, f1, fps):
    from mathutils import Vector
    cam.rotation_mode = "QUATERNION"
    for t, (tgt, pos, fwd) in zip(range(f0, f1 + 1), track(pose, obstacles, f0, f1, fps)):
        cam.location = Vector(pos)
        cam.rotation_quaternion = Vector(fwd).to_track_quat("-Z", "Y")
        cam.keyframe_insert("location", frame=t)
        cam.keyframe_insert("rotation_quaternion", frame=t)


def keyframe_orbit(cam, pose, f0, f1, dist=(0.78, 0.5), el=(10.0, 3.0), az0=150.0, turn=0.7):
    """turntable: one eased lap round the duck over the first `turn` of the clip (starting from
    the front three-quarter view), then a slow drift and a low push-in for the rest, down near
    the nozzles."""
    from mathutils import Vector
    centre = pose[f0 - 1:f1, :, :3].reshape(-1, 3).mean(axis=0)
    tgt = np.array([centre[0], centre[1], centre[2] + 0.01])
    cam.rotation_mode = "QUATERNION"
    n = f1 - f0
    for t in range(f0, f1 + 1):
        u = (t - f0) / max(n, 1)
        x = min(u / turn, 1.0)
        lap = x * x * (3 - 2 * x)
        y = max(u - turn, 0) / (1 - turn)
        push = y * y * (3 - 2 * y)
        fwd = _fwd(az0 + 360 * lap + 25 * push, el[0] + (el[1] - el[0]) * push)
        cam.location = Vector(tgt - (dist[0] + (dist[1] - dist[0]) * push) * fwd)
        cam.rotation_quaternion = Vector(fwd).to_track_quat("-Z", "Y")
        cam.keyframe_insert("location", frame=t)
        cam.keyframe_insert("rotation_quaternion", frame=t)
