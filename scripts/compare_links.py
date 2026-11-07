"""Body-by-body side-by-side renders: each reference body next to ours, same cameras,
everything else hidden. Shows which part's shape is off (agent tool).

usage: python scripts/compare_links.py <reference.xml> build/<design>/scene.xml out.png [--map ours=theirs ...]
Bodies are matched by name (our bodies that the reference lacks are skipped).
"""
import argparse
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("reference"); ap.add_argument("ours"); ap.add_argument("out")
ap.add_argument("--map", action="append", default=[], help="body name pairs, ours=theirs")
args = ap.parse_args()
ref, ours, out = args.reference, args.ours, args.out
names = dict(pair.split("=", 1) for pair in args.map)
_m, _r = mujoco.MjModel.from_xml_path(ours), mujoco.MjModel.from_xml_path(ref)
_rn = {_r.body(i).name for i in range(_r.nbody)}
BODIES = [_m.body(i).name for i in range(1, _m.nbody) if names.get(_m.body(i).name, _m.body(i).name) in _rn]
VIEWS = [(90, 0), (0, 0), (180, -89), (135, -30)]   # from +x, from +y... azimuth, elevation
W = H = 260

def shots(path, body):
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    bid = m.body(body).id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_group[g] == 3: m.geom_rgba[g, 3] = 0
        elif m.geom_matid[g] < 0 and m.geom_rgba[g, 3] > 0: pass
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1; opt.geomgroup[3] = 0
    # frame the body's visible geoms
    pts = [d.geom_xpos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    c = np.mean(pts, 0) if pts else d.xpos[bid]
    cam = mujoco.MjvCamera(); cam.lookat[:] = c; cam.distance = 0.32
    ims = []
    for az, el in VIEWS:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam, opt); ims.append(Image.fromarray(r.render()))
    return ims

rows = []
for b in BODIES:
    rr = shots(ref, names.get(b, b)); oo = shots(ours, b)
    row = Image.new("RGB", (W * 8 + 10, H), "white")
    for i, im in enumerate(rr + oo): row.paste(im, (i * W + (10 if i >= 4 else 0), 0))
    ImageDraw.Draw(row).text((4, 4), f"{b}: reference | ours", fill="red"); rows.append(row)
sheet = Image.new("RGB", (rows[0].width, H * len(rows)), "white")
for i, r in enumerate(rows): sheet.paste(r, (0, i * H))
sheet.save(out); print(out)
