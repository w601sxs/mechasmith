"""Draw an instrument HUD over rendered frames of a recorded jet-pack run.

usage: python hud.py run.npz FRAMES_DIR OUT_DIR [--jobs 8]

run.npz is the recording (`fly.py --record` / `rl_jetpack/play.py --record`); FRAMES_DIR holds
the rendered frames (blender/render.py's OUT/frames/fNNNN.png, one per recorded frame). Writes
OUT_DIR/fNNNN.png. The HUD shows:

  top left      the segment being flown and a course timeline with a playhead
  top right     ground speed against the commanded speed
  right         an altitude tape (trunk height above standing)
  bottom left   fan thrust bars (with the hover line and a peak marker) and each nacelle's
                thrust vector seen from the side
  bottom centre a scrolling gait strip: which sole is on the ground, with both fans' thrust traced
  bottom right  how much of the duck's weight the fans are carrying

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
PANEL = (14, 18, 26, 150)
TEAL = (64, 214, 196)
AMBER = (255, 184, 64)
RED = (255, 92, 80)
BLUE = (110, 168, 255)
MODE_COL = {"stand": (130, 140, 155), "walk": (92, 200, 120), "hop": (190, 130, 255),
            "moon run": (110, 168, 255), "take off": (255, 184, 64), "fly": (255, 140, 64),
            "hover": (255, 210, 90), "land": (64, 214, 196), "run and jump": (255, 120, 150)}


def font(px, bold=False):
    for p in (["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"]):
        try:
            return ImageFont.truetype(p, px * SS, index=1 if bold and p.endswith("ttc") else 0)
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


def panel(dr, box, r=14):
    x0, y0, x1, y1 = (c * SS for c in box)
    dr.rounded_rectangle((x0, y0, x1, y1), radius=r * SS, fill=PANEL, outline=(255, 255, 255, 28), width=SS)


def text(dr, xy, s, f, fill=INK, anchor="la"):
    dr.text((xy[0] * SS, xy[1] * SS), s, font=f, fill=fill, anchor=anchor)


def draw_hud(T, t, W, H, F_):
    f_s, f_m, f_l = F_
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    S = lambda *v: [c * SS for c in v]

    # ---- top left: mode chip + timeline
    mode = T["modes"][t]
    col = MODE_COL.get(mode, INK)
    panel(dr, (16, 16, 346, 78))
    dr.ellipse(S(30, 30, 42, 42), fill=col)
    text(dr, (52, 26), mode.upper(), f_l, fill=INK)
    sub = T["labels"][t]
    if sub.lower() != mode:
        text(dr, (52, 50), sub[:40], f_s, fill=DIM)
    x0, x1, y = 30, 332, 66
    for name, a, b in T["segs"]:
        xa, xb = x0 + (x1 - x0) * a / T["F"], x0 + (x1 - x0) * b / T["F"]
        c = MODE_COL.get(name, DIM)
        on = a <= t < b
        dr.rounded_rectangle(S(xa + 1, y, xb - 1, y + 6), radius=3 * SS, fill=c + ((255,) if on else (110,)))
    px = x0 + (x1 - x0) * t / T["F"]
    dr.polygon(S(px - 4, y - 5, px + 4, y - 5, px, y), fill=INK)

    # ---- top right: speed vs command
    panel(dr, (W - 236, 16, W - 16, 78))
    text(dr, (W - 222, 24), "GROUND SPEED", f_s, fill=DIM)
    sp = T["speed"][t]
    text(dr, (W - 30, 22), f"{sp:4.2f} m/s", f_m, fill=INK, anchor="ra")
    bx0, bx1, by = W - 222, W - 30, 58
    vmax = 1.0
    dr.rounded_rectangle(S(bx0, by, bx1, by + 10), radius=5 * SS, fill=(255, 255, 255, 30))
    w = (bx1 - bx0) * min(sp / vmax, 1)
    if w > 2:
        dr.rounded_rectangle(S(bx0, by, bx0 + w, by + 10), radius=5 * SS, fill=BLUE + (230,))
    if T["cmd"] is not None:
        c = math.hypot(*T["cmd"][t][:2])
        cx = bx0 + (bx1 - bx0) * min(c / vmax, 1)
        dr.line(S(cx, by - 4, cx, by + 14), fill=INK, width=2 * SS)
        text(dr, (cx, by - 6), "cmd", f_s, fill=DIM, anchor="md")

    # ---- right: altitude tape
    tx0, tx1, ty0, ty1 = W - 74, W - 16, 96, H - 200
    panel(dr, (tx0, ty0, tx1, ty1), r=12)
    text(dr, ((tx0 + tx1) / 2, ty0 + 10), "ALT", f_s, fill=DIM, anchor="ma")
    alt = T["alt"][t]
    span, mid = 0.8, (ty0 + 30 + ty1 - 10) / 2       # metres shown on the tape
    ppm = (ty1 - ty0 - 40) / span
    for k in range(-10, 30):
        a_ = k * 0.1
        yy = mid - (a_ - alt) * ppm
        if ty0 + 28 < yy < ty1 - 32:
            major = k % 2 == 0
            dr.line(S(tx0 + 8, yy, tx0 + (24 if major else 16), yy), fill=(255, 255, 255, 160 if major else 70),
                    width=SS)
            if major:
                text(dr, (tx0 + 28, yy), f"{a_:.1f}", f_s, fill=DIM, anchor="lm")
    gy = mid - (0 - alt) * ppm                       # standing height (the ground)
    if ty0 + 28 < gy < ty1 - 8:
        dr.line(S(tx0 + 6, gy, tx1 - 6, gy), fill=TEAL + (180,), width=SS)
    dr.polygon(S(tx0 + 2, mid - 7, tx0 + 2, mid + 7, tx0 + 12, mid), fill=AMBER)
    text(dr, ((tx0 + tx1) / 2, ty1 - 20), f"{100 * max(alt, 0):3.0f}cm", f_s, fill=INK, anchor="ma")

    # ---- bottom left: thrust bars + thrust vectors
    bx, byb = 16, H - 16
    panel(dr, (bx, byb - 196, bx + 230, byb))
    text(dr, (bx + 14, byb - 186), "FAN THRUST", f_s, fill=DIM)
    bar_top, bar_bot = byb - 146, byb - 36
    for j, name in enumerate(("L", "R")):
        u = float(T["thr"][t, j])
        peak = float(T["thr"][max(0, t - T["fps"]):t + 1, j].max())
        x = bx + 22 + j * 58
        dr.rounded_rectangle(S(x, bar_top, x + 30, bar_bot), radius=6 * SS, fill=(255, 255, 255, 26))
        h = (bar_bot - bar_top) * min(u, 1)
        if h > 1:
            # gradient fill: colour of each slice by its thrust level
            n = max(int(h), 1)
            for k in range(n):
                yy = bar_bot - k - 1
                dr.rectangle(S(x + 2, yy, x + 28, yy + 1), fill=heat(k / (bar_bot - bar_top)) + (235,))
        hy = bar_bot - (bar_bot - bar_top) * T["hover"][j]
        for xx in range(x - 4, x + 34, 6):
            dr.line(S(xx, hy, xx + 3, hy), fill=INK, width=SS)
        py = bar_bot - (bar_bot - bar_top) * min(peak, 1)
        dr.line(S(x + 2, py, x + 28, py), fill=INK + (200,), width=2 * SS)
        text(dr, (x + 15, bar_bot + 6), f"{100 * u:3.0f}%", f_s, fill=INK, anchor="ma")
        text(dr, (x + 15, bar_top - 4), name, f_s, fill=DIM, anchor="md")
    hy = bar_bot - (bar_bot - bar_top) * T["hover"][1]
    text(dr, (bx + 216, byb - 186), "┄ hover", f_s, fill=DIM, anchor="ra")
    # thrust vector dials (side view): arrow along the jet, opposite the thrust
    for j in range(2):
        cx, cy, R = bx + 160 + j * 46, byb - 66, 18
        dr.ellipse(S(cx - R, cy - R, cx + R, cy + R), outline=(255, 255, 255, 70), width=SS)
        dr.line(S(cx, cy - R, cx, cy - R + 4), fill=DIM, width=SS)
        a = T["tilt"][t, j]
        L = R * (0.35 + 0.65 * min(float(T["thr"][t, j]) / 0.6, 1))
        ex, ey = cx + L * math.sin(a), cy - L * math.cos(a)
        c = heat(float(T["thr"][t, j]))
        dr.line(S(cx, cy, ex, ey), fill=c, width=3 * SS)
        hd = 6
        dr.polygon(S(ex + hd * math.sin(a), ey - hd * math.cos(a),
                     ex + hd * 0.6 * math.cos(a), ey + hd * 0.6 * math.sin(a),
                     ex - hd * 0.6 * math.cos(a), ey - hd * 0.6 * math.sin(a)), fill=c)
        text(dr, (cx, cy + R + 6), f"{math.degrees(a):+3.0f}°", f_s, fill=DIM, anchor="ma")
    text(dr, (bx + 183, byb - 104), "VECTOR", f_s, fill=DIM, anchor="ma")

    # ---- bottom centre: gait strip + thrust traces
    gx0, gx1, gy0, gy1 = 262, W - 196, H - 112, H - 16
    panel(dr, (gx0, gy0, gx1, gy1))
    text(dr, (gx0 + 14, gy0 + 8), "GAIT", f_s, fill=DIM)
    text(dr, (gx1 - 14, gy0 + 8), "3 s", f_s, fill=DIM, anchor="ra")
    win = 3 * T["fps"]
    lx0, lx1 = gx0 + 36, gx1 - 14
    rows = [(gy0 + 32, "L"), (gy0 + 50, "R")]
    for k, (ry, nm) in enumerate(rows):
        text(dr, (gx0 + 14, ry + 5), nm, f_s, fill=DIM, anchor="lm")
        dr.rounded_rectangle(S(lx0, ry, lx1, ry + 10), radius=3 * SS, fill=(255, 255, 255, 18))
        run, a0 = False, 0
        for i in range(win + 1):
            tt = t - win + i
            on = 0 <= tt < T["F"] and T["feet"][tt, k]
            if on and not run:
                run, a0 = True, i
            if run and (not on or i == win):
                xa = lx0 + (lx1 - lx0) * a0 / win
                xb = lx0 + (lx1 - lx0) * i / win
                dr.rounded_rectangle(S(xa, ry, max(xb, xa + 2), ry + 10), radius=3 * SS, fill=TEAL + (230,))
                run = False
    ty_b, ty_t = gy1 - 8, gy0 + 66
    for j, c in enumerate((AMBER, RED)):
        pts = []
        for i in range(win + 1):
            tt = t - win + i
            if tt < 0:
                continue
            u = float(T["thr"][tt, j])
            pts.append((lx0 + (lx1 - lx0) * i / win, ty_b - (ty_b - ty_t) * min(u, 1)))
        if len(pts) > 1:
            dr.line([c * SS for p in pts for c in p], fill=c + (220,), width=2 * SS, joint="curve")
    text(dr, (gx0 + 14, (ty_b + ty_t) / 2), "fan", f_s, fill=DIM, anchor="lm")

    # ---- bottom right: lift ring
    cx, cy, R = W - 104, H - 104, 62
    panel(dr, (W - 180, H - 180, W - 16, H - 16))
    frac = float(T["lift"][t])
    a0, a1 = 135, 405                                # 270° arc: 0 .. 150 % of weight
    dr.arc(S(cx - R, cy - R, cx + R, cy + R), a0, a1, fill=(255, 255, 255, 34), width=10 * SS)
    end = a0 + (a1 - a0) * min(max(frac, 0) / 1.5, 1)
    if end > a0 + 1:
        dr.arc(S(cx - R, cy - R, cx + R, cy + R), a0, end, fill=heat(min(frac, 1.2) / 1.4) + (240,), width=10 * SS)
    tick = math.radians(a0 + (a1 - a0) / 1.5)        # 100 %: the fans alone hold it up
    dr.line(S(cx + (R - 14) * math.cos(tick), cy + (R - 14) * math.sin(tick),
              cx + (R + 7) * math.cos(tick), cy + (R + 7) * math.sin(tick)), fill=INK, width=2 * SS)
    text(dr, (cx, cy - 12), f"{100 * max(frac, 0):3.0f}%", f_l, fill=INK, anchor="mm")
    text(dr, (cx, cy + 12), "of weight", f_s, fill=DIM, anchor="mm")
    text(dr, (cx, cy + 28), "on the fans", f_s, fill=DIM, anchor="mm")

    return im.resize((W, H), Image.LANCZOS)


_T = None


def _init(T):
    global _T
    _T = T


def _frame(args):
    t, src, dst = args
    base = Image.open(src).convert("RGBA")
    W, H = base.size
    hud = draw_hud(_T, t, W, H, (font(11), font(14), font(18, bold=True)))
    Image.alpha_composite(base, hud).convert("RGB").save(dst)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
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
    with Pool(a.jobs, initializer=_init, initargs=(T,)) as p:
        for k, _ in enumerate(p.imap_unordered(_frame, jobs, chunksize=8)):
            if k % 100 == 0:
                print(f"hud {k}/{len(jobs)}", flush=True)
    print(f"wrote {len(jobs)} frames -> {a.out}")
