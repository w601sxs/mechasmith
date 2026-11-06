"""Headless renders the agent can look at (and humans can skim)."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

VIEWS = {  # label: (azimuth, elevation)
    "front": (180, -10),
    "left side": (270, -10),
    "iso": (215, -25),
    "back-iso": (35, -25),
}


def _camera(model, data, azimuth, elevation, zoom=1.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Frame the robot from its geometry, not the model statistic (which includes the floor).
    pts = []
    for g in range(model.ngeom):
        if model.geom_group[g] == 2:
            pts.append(data.geom_xpos[g])
    pts = np.array(pts) if pts else np.zeros((1, 3))
    lo, hi = pts.min(0), pts.max(0)
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = max(0.12, 2.2 * np.linalg.norm(hi - lo)) / zoom
    cam.azimuth, cam.elevation = azimuth, elevation
    return cam


def _rest_on_floor(model, data):
    """Drop a floating robot so its lowest visible geometry touches z = 0."""
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_group[g] != 2 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        zmin = min(zmin, float((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()))
    data.qpos[2] -= zmin
    mujoco.mj_forward(model, data)


def contact_sheet(scene_xml, out_png, qpos: dict | None = None, size=(560, 480), title: str = "",
                  views=VIEWS, show_collision=False, zoom=1.0, full_qpos=None) -> Path:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if full_qpos is not None:
        data.qpos[:] = full_qpos
    for name, val in (qpos or {}).items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = val
    mujoco.mj_forward(model, data)
    if full_qpos is None and model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        _rest_on_floor(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    if show_collision:
        opt.geomgroup[3] = 1
    w, h = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, h, w)
    tiles = []
    for label, (az, el) in views.items():
        renderer.update_scene(data, camera=_camera(model, data, az, el, zoom), scene_option=opt)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((10, 8), label, fill=(40, 40, 40))
        tiles.append(img)
    renderer.close()
    cols = 2
    rows = (len(tiles) + 1) // 2
    top = 28 if title else 0
    sheet = Image.new("RGB", (cols * w, rows * h + top), (255, 255, 255))
    if title:
        ImageDraw.Draw(sheet).text((10, 8), title, fill=(20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, top + (i // cols) * h))
    out_png = Path(out_png)
    sheet.save(out_png)
    return out_png
