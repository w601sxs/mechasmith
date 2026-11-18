"""Render what the duck's head camera sees, from a scene render.py already built and baked.

    blender -b OUT/scene.blend -P pov.py -- EXPORT_DIR [--size 480 360] [--fov 100] [--samples 16] [--frames A B]

EXPORT_DIR is export.py's output (it must have eye.npy, the head_camera site per frame). Adds a
camera at the lens, looking along the head's x axis with its z axis up, so it nods, turns and
rolls with the head, and renders OUT/pov/fNNNN.png with the same smoke bake, lights and
obstacles. The duck's camera module has no lens spec in the design, so --fov (horizontal,
degrees) is a choice; 100 is a typical wide hobby-camera lens. hud.py --pip puts these frames
in a corner of the main view.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("export", type=Path)
ap.add_argument("--size", type=int, nargs=2, default=(480, 360))
ap.add_argument("--fov", type=float, default=100.0)
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--frames", type=int, nargs=2, default=None, help="render only this range (1-based)")
a = ap.parse_args(argv)

eye = np.load(a.export / "eye.npy")
sc = bpy.context.scene
out = Path(bpy.data.filepath).parent

cam = bpy.data.objects.new("eye", bpy.data.cameras.new("eye"))
cam.data.sensor_fit = "HORIZONTAL"
cam.data.angle = math.radians(a.fov)
cam.data.clip_start = 0.012                   # just past the lens, so the face plate stays out
cam.data.clip_end = 40
sc.collection.objects.link(cam)
cam.rotation_mode = "QUATERNION"
# site frame (x forward, z up) -> Blender camera (looks down -z, y up)
site_to_cam = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0))).to_quaternion()   # columns: cam x, y, z in site axes
for t in range(sc.frame_start, sc.frame_end + 1):
    p = eye[min(t - 1, len(eye) - 1)]
    cam.location = Vector(p[:3])
    cam.rotation_quaternion = Quaternion(p[3:]) @ site_to_cam
    cam.keyframe_insert("location", frame=t)
    cam.keyframe_insert("rotation_quaternion", frame=t)
sc.camera = cam
if a.frames:
    sc.frame_start, sc.frame_end = a.frames

r = sc.render
r.resolution_x, r.resolution_y = a.size
r.filepath = str(out / "pov" / "f")
if hasattr(sc.eevee, "taa_render_samples"):
    sc.eevee.taa_render_samples = a.samples
print(f"rendering head camera {a.size[0]}x{a.size[1]}, {a.fov:.0f} deg, frames {sc.frame_start}-{sc.frame_end}",
      flush=True)
bpy.ops.render.render(animation=True)
print("done", out / "pov")
