"""Draw an instrument HUD over rendered frames of a recorded jet-pack run.

usage: python hud.py run.npz FRAMES_DIR OUT_DIR [--pip POV_DIR] [--jobs 8]

run.npz is the recording (`fly.py --record` / `rl_jetpack/play.py --record`); FRAMES_DIR holds
the rendered frames (blender/render.py's OUT/frames/fNNNN.png, one per recorded frame). Writes
OUT_DIR/fNNNN.png with a slim instrument dock under the scene (or over its bottom edge with
--overlay). With --pip OUT/pov (blender/pov.py), the head camera's view is inset top right.
Dock, left to right:

  mode      the segment being flown, on a course timeline with a playhead
  thrust    fan thrust bars (hover line, one-second peak marker) and each nacelle's thrust vector
            seen from the side
  gait      a scrolling 3 s strip of which sole is on the ground, with both fans' thrust traced
  lift      how much of the duck's weight the fans are carrying
  alt/speed trunk height above standing; ground speed against the commanded speed

Everything is computed from the recording with MuJoCo (contacts, thrust axes, mass), nothing is
read from the policy. Recordings without segment labels fall back to play.py's box course.
"""
import argparse
import math
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# play.py's box course, for recordings made before labels were saved
COURSE = [("stand", 1), ("walk", 9), ("hop", 4), ("moon run", 5), ("take off", 3), ("fly", 4),
          ("hover", 2), ("land", 3), ("walk", 4)]
STAND_Z = 0.129
SS = 2                                    # supersampling for smooth edges

INK = (236, 240, 245)
DIM = (150, 160, 175)
TEAL = (64, 214, 196)
AMBER = (255, 184, 64)
RED = (255, 92, 80)
BLUE = (110, 168, 255)
MODE_COL = {"stand": (130, 140, 155), "walk": (92, 200, 120), "hop": (190, 130, 255),
            "moon run": (110, 168, 255), "take off": (255, 184, 64), "fly": (255, 140, 64),
            "hover": (255, 210, 90), "land": (64, 214, 196), "run and jump": (255, 120, 150),
            "run": (255, 120, 150), "stop": (130, 140, 155)}    # "run" after "moon run": first match wins


def font(px, k=SS, bold=False):
    for p in (["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"]):
        try:
            return ImageFont.truetype(p, int(px * k), index=1 if bold and p.endswith("ttc") else 0)
        except OSError:
            pass
    return ImageFont.load_default()


def mode_name(label):
    label = str(label).lower()
    for k in MODE_COL:
        if label.startswith(k) or k in label:
            return k
    return label.split(":")[0]


def heat(f):
    """thrust colour: teal at idle, amber near hover, red when pushing hard"""
    stops = [(0.0, TEAL), (0.5, AMBER), (0.85, RED)]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if f <= b:
            t = (f - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return RED


def analyse(run):
    """per-frame telemetry from the recording"""
    rec = np.load(run, allow_pickle=True)
    m = mujoco.MjModel.from_xml_path(str(rec["scene"]))
    d = mujoco.MjData(m)
    q, thr, fps = rec["qpos"], np.asarray(rec["throttle"], float), int(rec["fps"])
    F = len(q)
    fan_act = [i for i in range(m.nu) if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_SITE)]
    sites = [m.actuator_trnid[i, 0] for i in fan_act]
    tmax = np.array([m.actuator_ctrlrange[i, 1] for i in fan_act])
    weight = m.body_subtreemass[1] * 9.81
    soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_sole_collision", "right_sole_collision")]
    tilt = np.zeros((F, len(sites)))
    lift = np.zeros(F)
    feet = np.zeros((F, 2), bool)
    for t in range(F):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        up = 0.0
        fwd = d.xmat[1].reshape(3, 3)[:, 0]          # trunk heading, for a side view
        fwd = fwd - fwd[2] * np.array([0, 0, 1.0])
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        for j, s in enumerate(sites):
            ax = d.site_xmat[s].reshape(3, 3)[:, 2]
            tilt[t, j] = math.atan2(ax @ fwd, ax[2])  # 0 = straight up, + = thrust tipped forward
            up += thr[t, j] * tmax[j] * ax[2]
        lift[t] = up / weight
        for c in d.contact[:d.ncon]:
            for k, g in enumerate(soles):
                if g in (c.geom1, c.geom2):
                    feet[t, k] = True
    pos = q[:, :3]
    v = np.gradient(pos[:, :2], 1 / fps, axis=0)
    speed = np.linalg.norm(v, axis=1)
    for t in range(1, F):                             # light smoothing so the needle doesn't buzz
        speed[t] = 0.75 * speed[t - 1] + 0.25 * speed[t]
    if "label" in rec.files:
        labels = [str(x) for x in rec["label"]]
        cmd = np.asarray(rec["command"], float)
    else:
        labels, cmd = [], None
        for name, secs in COURSE:
            labels += [name] * int(round(secs * fps))
        labels = (labels + [labels[-1]] * F)[:F]
    modes = [mode_name(x) for x in labels]
    # timeline segments
    segs, start = [], 0
    for t in range(1, F + 1):
        if t == F or modes[t] != modes[start]:
            segs.append((modes[start], start, t))
            start = t
    return dict(F=F, fps=fps, thr=thr, tmax=tmax, hover=weight / 2 / tmax, tilt=tilt, lift=lift, feet=feet,
                alt=pos[:, 2] - STAND_Z, speed=speed, cmd=cmd, modes=modes, labels=labels, segs=segs)


DW, DH = 800, 92                          # the dock in design units (scaled to the frame width)
DOCK_BG = (12, 15, 22)


def draw_dock(T, t, W, overlay):
    """One instrument strip, DW x DH design units, scaled to width W."""
    k = W / DW * SS
    H = int(round(DH * W / DW))
    S = lambda *v: [c * k for c in v]
    f_s, f_m, f_l = (font(px, k, b) for px, b in ((9, False), (12, False), (15, True)))
    im = Image.new("RGBA", (int(W * SS), int(H * SS)), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    tx = lambda x, y, s, f, fill=INK, anchor="la": dr.text((x * k, y * k), s, font=f, fill=fill, anchor=anchor)
    sep = lambda x: dr.line(S(x, 14, x, DH - 14), fill=(255, 255, 255, 30), width=max(1, int(k / 2)))
    dr.line(S(0, 0.5, DW, 0.5), fill=(255, 255, 255, 40), width=max(1, int(k / 2)))
    th = lambda j, tt=t: float(T["thr"][tt, j])

    # ---- mode + course timeline
    mode = T["modes"][t]
    col = MODE_COL.get(mode, INK)
    dr.ellipse(S(16, 20, 26, 30), fill=col)
    tx(34, 17, mode.upper(), f_l)
    sub = T["labels"][t]
    if sub.lower() != mode:
        tx(34, 37, sub[:30], f_s, fill=DIM)
    x0, x1, y = 16, 170, 64
    for name, a, b in T["segs"]:
        xa, xb = x0 + (x1 - x0) * a / T["F"], x0 + (x1 - x0) * b / T["F"]
        dr.rounded_rectangle(S(xa + 0.7, y, xb - 0.7, y + 5), radius=2.5 * k,
                             fill=MODE_COL.get(name, DIM) + ((255,) if a <= t < b else (95,)))
    px = x0 + (x1 - x0) * t / T["F"]
    dr.polygon(S(px - 3.5, y - 5, px + 3.5, y - 5, px, y - 0.5), fill=INK)
    tx(16, 74, f"{t / T['fps']:4.1f} s", f_s, fill=DIM)
    sep(184)

    # ---- fan thrust: bars + thrust vectors
    tx(196, 10, "THRUST", f_s, fill=DIM)
    top, bot = 26, 70
    for j, name in enumerate(("L", "R")):
        u = th(j)
        peak = float(T["thr"][max(0, t - T["fps"]):t + 1, j].max())
        x = 198 + j * 24
        dr.rounded_rectangle(S(x, top, x + 12, bot), radius=3 * k, fill=(255, 255, 255, 24))
        h = (bot - top) * min(u, 1)
        for kk in range(int(h * k)):              # gradient fill, one device row at a time
            yy = bot * k - kk - 1
            dr.line((x * k + k, yy, (x + 12) * k - k, yy), fill=heat(kk / k / (bot - top)) + (240,))
        hy = bot - (bot - top) * T["hover"][j]
        for xx in range(-2, 14, 4):
            dr.line(S(x + xx, hy, x + xx + 2, hy), fill=INK, width=max(1, int(k / 2)))
        py = bot - (bot - top) * min(peak, 1)
        dr.line(S(x + 1, py, x + 11, py), fill=INK + (200,), width=max(1, int(k)))
        tx(x + 6, bot + 4, f"{100 * u:.0f}", f_s, anchor="ma")
    for j in range(2):                            # side view of each nacelle's thrust
        cx, cy, R = 264 + j * 30, 46, 11
        dr.ellipse(S(cx - R, cy - R, cx + R, cy + R), outline=(255, 255, 255, 60), width=max(1, int(k / 2)))
        a = T["tilt"][t, j]
        L = R * (0.35 + 0.65 * min(th(j) / 0.6, 1))
        ex, ey = cx + L * math.sin(a), cy - L * math.cos(a)
        c = heat(th(j))
        dr.line(S(cx, cy, ex, ey), fill=c, width=int(2 * k))
        hd = 4.5
        dr.polygon(S(ex + hd * math.sin(a), ey - hd * math.cos(a), ex + hd * .6 * math.cos(a),
                     ey + hd * .6 * math.sin(a), ex - hd * .6 * math.cos(a), ey - hd * .6 * math.sin(a)), fill=c)
        tx(cx, cy + R + 4, f"{math.degrees(a):+.0f}°", f_s, fill=DIM, anchor="ma")
    tx(279, 10, "VECTOR", f_s, fill=DIM, anchor="ma")
    sep(320)

    # ---- gait strip + thrust traces, last 3 s
    gx0, gx1 = 346, 556
    tx(332, 10, "GAIT", f_s, fill=DIM)
    tx(gx1, 10, "3 s", f_s, fill=DIM, anchor="ra")
    win = 3 * T["fps"]
    for kf, (ry, nm) in enumerate(((26, "L"), (39, "R"))):
        tx(332, ry + 4, nm, f_s, fill=DIM, anchor="lm")
        dr.rounded_rectangle(S(gx0, ry, gx1, ry + 8), radius=2.5 * k, fill=(255, 255, 255, 18))
        run, a0 = False, 0
        for i in range(win + 1):
            tt = t - win + i
            on = 0 <= tt < T["F"] and T["feet"][tt, kf]
            if on and not run:
                run, a0 = True, i
            if run and (not on or i == win):
                xa, xb = gx0 + (gx1 - gx0) * a0 / win, gx0 + (gx1 - gx0) * i / win
                dr.rounded_rectangle(S(xa, ry, max(xb, xa + 2), ry + 8), radius=2.5 * k, fill=TEAL + (230,))
                run = False
    yb, yt = 80, 54
    dr.line(S(gx0, yb, gx1, yb), fill=(255, 255, 255, 25), width=max(1, int(k / 2)))
    for j, c in enumerate((AMBER, RED)):
        pts = [(gx0 + (gx1 - gx0) * i / win, yb - (yb - yt) * min(th(j, t - win + i), 1))
               for i in range(win + 1) if t - win + i >= 0]
        if len(pts) > 1:
            dr.line([v * k for p in pts for v in p], fill=c + (220,), width=int(1.5 * k))
    tx(332, (yb + yt) / 2, "fan", f_s, fill=DIM, anchor="lm")
    sep(572)

    # ---- lift ring: share of the weight on the fans
    cx, cy, R = 612, 48, 27
    frac = float(T["lift"][t])
    a0, a1 = 135, 405                             # 270 deg = 0 .. 150 % of weight
    dr.arc(S(cx - R, cy - R, cx + R, cy + R), a0, a1, fill=(255, 255, 255, 32), width=int(6 * k))
    end = a0 + (a1 - a0) * min(max(frac, 0) / 1.5, 1)
    if end > a0 + 1:
        dr.arc(S(cx - R, cy - R, cx + R, cy + R), a0, end, fill=heat(min(frac, 1.2) / 1.4) + (240,), width=int(6 * k))
    tk = math.radians(a0 + (a1 - a0) / 1.5)
    dr.line(S(cx + (R - 8) * math.cos(tk), cy + (R - 8) * math.sin(tk),
              cx + (R + 5) * math.cos(tk), cy + (R + 5) * math.sin(tk)), fill=INK, width=max(1, int(k)))
    tx(cx, cy - 2, f"{100 * max(frac, 0):.0f}%", f_m, anchor="mm")
    tx(cx, cy + 11, "lift", f_s, fill=DIM, anchor="mm")
    sep(652)

    # ---- altitude and speed
    def meter(y, label, val, unit, frac_, colr, tick=None):
        tx(664, y, label, f_s, fill=DIM)
        tx(788, y - 2, f"{val}{unit}", f_m, anchor="ra")
        dr.rounded_rectangle(S(664, y + 16, 788, y + 22), radius=3 * k, fill=(255, 255, 255, 26))
        w = 124 * min(max(frac_, 0), 1)
        if w > 1.5:
            dr.rounded_rectangle(S(664, y + 16, 664 + w, y + 22), radius=3 * k, fill=colr + (235,))
        if tick is not None:
            xt = 664 + 124 * min(max(tick, 0), 1)
            dr.line(S(xt, y + 13, xt, y + 25), fill=INK, width=max(1, int(k)))
    alt = max(float(T["alt"][t]), 0)
    meter(12, "ALT", f"{100 * alt:.0f}", " cm", alt / 1.2, AMBER)
    sp = float(T["speed"][t])
    c = None if T["cmd"] is None else math.hypot(*T["cmd"][t][:2])
    meter(50, "SPEED", f"{sp:.2f}", " m/s", sp / 1.0, BLUE, None if c is None else c / 1.0)

    bg = Image.new("RGBA", im.size, DOCK_BG + ((185,) if overlay else (255,)))
    bg.alpha_composite(im)
    return bg.resize((W, H), Image.LANCZOS)


def draw_pip(base, pip, frac=0.3, margin=0.016):
    """the head camera's view, inset in the top-right corner of the frame, with a HEAD CAM tag."""
    W, H = base.size
    w = int(W * frac)
    h = round(w * pip.height / pip.width)
    m = int(W * margin)
    k = SS
    lay = Image.new("RGBA", ((w + 4) * k, (h + 4) * k), (0, 0, 0, 0))
    mask = Image.new("L", lay.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((2 * k, 2 * k, (w + 2) * k, (h + 2) * k), radius=8 * k, fill=255)
    lay.paste(pip.convert("RGBA").resize(lay.size, Image.LANCZOS), (0, 0), mask)
    dr = ImageDraw.Draw(lay)
    dr.rounded_rectangle((k, k, (w + 3) * k, (h + 3) * k), radius=9 * k, outline=INK + (210,), width=2 * k)
    f = font(10, k, bold=True)
    tag = "HEAD CAM"
    tw = dr.textlength(tag, font=f) / k
    dr.rounded_rectangle((8 * k, 8 * k, (8 + 20 + tw) * k, 26 * k), radius=5 * k, fill=DOCK_BG + (200,))
    dr.ellipse((14 * k, 13.5 * k, 21 * k, 20.5 * k), fill=(255, 72, 72, 255))
    dr.text((25 * k, 17 * k), tag, font=f, fill=INK, anchor="lm")
    base.alpha_composite(lay.resize((w + 4, h + 4), Image.LANCZOS), (W - w - m - 2, m - 2))


_T, _OVERLAY, _PIP = None, False, None


def _init(T, overlay, pip=None):
    global _T, _OVERLAY, _PIP
    _T, _OVERLAY, _PIP = T, overlay, pip


def _frame(args):
    t, src, dst = args
    base = Image.open(src).convert("RGBA")
    W, H = base.size
    if _PIP is not None and (_PIP / src.name).exists():
        draw_pip(base, Image.open(_PIP / src.name))
    dock = draw_dock(_T, t, W, _OVERLAY)
    if _OVERLAY:                                   # over the bottom of the frame
        base.alpha_composite(dock, (0, H - dock.height))
        out = base
    else:                                          # below it: the scene stays whole
        out = Image.new("RGBA", (W, H + dock.height))
        out.paste(base, (0, 0))
        out.paste(dock, (0, H))
    out.convert("RGB").save(dst)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--overlay", action="store_true", help="draw the dock over the bottom of the frame, not below it")
    ap.add_argument("--pip", type=Path, default=None,
                    help="frames from blender/pov.py (same names) to inset top right: the head camera's view")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="just these frame numbers (1-based)")
    a = ap.parse_args()
    T = analyse(a.run)
    files = sorted(a.frames.glob("f*.png"))
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, f in enumerate(files):
        if a.only and i + 1 not in a.only:
            continue
        t = min(round(i * (T["F"] - 1) / max(len(files) - 1, 1)), T["F"] - 1)
        jobs.append((t, f, a.out / f.name))
    with Pool(a.jobs, initializer=_init, initargs=(T, a.overlay, a.pip)) as p:
        for k, _ in enumerate(p.imap_unordered(_frame, jobs, chunksize=8)):
            if k % 100 == 0:
                print(f"hud {k}/{len(jobs)}", flush=True)
    print(f"wrote {len(jobs)} frames -> {a.out}")
