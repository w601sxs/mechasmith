"""How much a built design looks like a reference model: silhouette overlap from three views.

The reference is posed at its `--ref-key` keyframe (else its first, else zero) and ours at the
same joint angles wherever the joint names match (else its `home`), both with the root at the
origin and level; ours is turned about z by the quarter turn that overlaps best (or `--yaw`). Every visual mesh is projected orthographically onto the front (y-z),
side (x-z) and top (x-y) planes and filled in at `--px` mm per pixel. The score per view is the
intersection over union of the two silhouettes (1.0 = identical outlines); `--bodies` also scores
each body that both models share by name, which shows where the outline differs.

usage: python scripts/compare_looks.py <reference.xml> build/<design>/robot.xml [--png out.png] [--bodies]
"""
import argparse
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw

VIEWS = {"front": (1, 2), "side": (0, 2), "top": (0, 1)}       # world axes kept in each view


def joint_angles(m, d):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(d.qpos[m.jnt_qposadr[j]])
            for j in range(m.njnt) if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)}


def posed(xml, key, angles=None):
    """Model posed at keyframe `key` (else the first one, else zero), then at `angles` by joint name."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key) if key else -1
    if k < 0 and m.nkey:
        k = 0
    if k >= 0:
        d.qpos[:] = m.key_qpos[k]
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if angles and n in angles:
            d.qpos[m.jnt_qposadr[j]] = angles[n]
    if m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    return m, d


def visual_triangles(m, d):
    """{body name: (N, 3, 3) triangles in mm, in the root body's frame} for every visual mesh geom."""
    out = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g]:
            continue
        if m.geom_rgba[g][3] == 0:
            continue
        mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
        f = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + m.mesh_facenum[mi]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        w = (w - d.xpos[1]) @ d.xmat[1].reshape(3, 3) * 1000.0      # in the root body's frame
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        out.setdefault(body, []).append(w[f])
    return {b: np.concatenate(t) for b, t in out.items()}


def silhouette(tris, axes, lo, shape, px):
    img = Image.new("1", shape, 0)
    dr = ImageDraw.Draw(img)
    p = (tris[:, :, list(axes)] - lo) / px
    p[:, :, 1] = shape[1] - 1 - p[:, :, 1]
    for t in p:
        dr.polygon([tuple(q) for q in t], fill=1)
    return np.asarray(img, bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--ref-key", default="home")
    ap.add_argument("--px", type=float, default=2.0)
    ap.add_argument("--png")
    ap.add_argument("--bodies", action="store_true")
    ap.add_argument("--yaw", type=float, help="turn our model about z by this many degrees (default: best of 0/90/180/270)")
    a = ap.parse_args()
    mr, dr_ = posed(a.ref, a.ref_key)
    ref = visual_triangles(mr, dr_)
    # Our model takes the reference's joint angles wherever the joint names match.
    ours = visual_triangles(*posed(a.ours, "home", joint_angles(mr, dr_)))
    # Base frames may differ by a quarter turn about z: use the yaw whose top views overlap most.
    def rot(tris, deg):
        c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        return {b: t @ R.T for b, t in tris.items()}

    def top_iou(o):
        allp = np.concatenate([t.reshape(-1, 3) for t in (*ref.values(), *o.values())])
        l2 = allp.min(0)[:2] - 10
        shape = tuple(int(np.ceil(x)) for x in (allp.max(0)[:2] + 10 - l2) / (2 * a.px))
        return iou(silhouette(np.concatenate(list(ref.values())), (0, 1), l2, shape, 2 * a.px),
                   silhouette(np.concatenate(list(o.values())), (0, 1), l2, shape, 2 * a.px))
    yaw = a.yaw if a.yaw is not None else max((0, 90, 180, 270), key=lambda y: top_iou(rot(ours, y)))
    ours = rot(ours, yaw)
    allp = np.concatenate([t.reshape(-1, 3) for t in (*ref.values(), *ours.values())])
    lo, hi = allp.min(0) - 10, allp.max(0) + 10
    result, panels = {"yaw_deg": yaw, "views": {}}, []
    for view, axes in VIEWS.items():
        l2 = lo[list(axes)]
        shape = tuple(int(np.ceil(s)) for s in (hi[list(axes)] - l2) / a.px)
        sr = silhouette(np.concatenate(list(ref.values())), axes, l2, shape, a.px)
        so = silhouette(np.concatenate(list(ours.values())), axes, l2, shape, a.px)
        result["views"][view] = round(iou(sr, so), 3)
        rgb = np.full(sr.shape + (3,), 255, np.uint8)
        rgb[sr & ~so] = (220, 60, 60)          # reference only: red
        rgb[so & ~sr] = (60, 110, 220)         # ours only: blue
        rgb[sr & so] = (120, 120, 120)         # both: grey
        panels.append(Image.fromarray(rgb))
        if a.bodies:
            for b in sorted(set(ref) & set(ours)):
                s = iou(silhouette(ref[b], axes, l2, shape, a.px), silhouette(ours[b], axes, l2, shape, a.px))
                result.setdefault("bodies", {}).setdefault(b, {})[view] = round(s, 3)
    result["mean"] = round(float(np.mean(list(result["views"].values()))), 3)
    if a.bodies:
        for b, v in result["bodies"].items():
            v["mean"] = round(float(np.mean(list(v.values()))), 3)
    if a.png:
        W = sum(p.width for p in panels) + 20 * (len(panels) - 1)
        H = max(p.height for p in panels) + 24
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        dr, x = ImageDraw.Draw(sheet), 0
        for (view, _), p in zip(VIEWS.items(), panels):
            sheet.paste(p, (x, 24))
            dr.text((x + 4, 4), f"{view}: IoU {result['views'][view]}  (grey both, red reference only, blue ours only)",
                    fill=(0, 0, 0))
            x += p.width + 20
        sheet.save(a.png)
        result["png"] = a.png
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
