"""Render an exported jet-pack run in Blender with Mantaflow smoke from the fans.

usage (Blender in background):
  blender -b -P render.py -- EXPORT_DIR OUT_DIR [--res 160] [--noise 2] [--frames 1 500]
                                                [--size 960 720] [--samples 64] [--skip-bake]

EXPORT_DIR comes from export.py. The duck's geoms are loaded as they are and keyframed from the
recorded poses; each nozzle exit is a small disc that is a Mantaflow smoke inflow, its exit speed
and density keyframed from the fan's throttle and its normal along the exhaust, so the jet tilts
with the nacelle. The floor is the bottom wall of the smoke domain (the jets spread along it), a
box is a collision object, and the camera follows the duck like fly.py's. Frames go to
OUT_DIR/frames, the bake to OUT_DIR/cache.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("export", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--res", type=int, default=160, help="smoke grid cells along the domain's longest side")
ap.add_argument("--noise", type=int, default=0, help="noise upres factor (0 = off)")
ap.add_argument("--frames", type=int, nargs=2, default=None)
ap.add_argument("--size", type=int, nargs=2, default=(960, 720))
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--vmax", type=float, default=9.0, help="exhaust speed at full throttle, m/s (drawn, not real)")
ap.add_argument("--skip-bake", action="store_true")
a = ap.parse_args(argv)
a.out = a.out.resolve()
(a.out / "frames").mkdir(parents=True, exist_ok=True)

meta = json.load(open(a.export / "scene.json"))
pose = np.load(a.export / "pose.npy")          # frames x geoms x (pos, quat wxyz)
noz = np.load(a.export / "nozzles.npy")        # frames x fans x (exit pos, thrust axis, throttle)
F = meta["frames"]
f0, f1 = a.frames or (1, F)

# ---- empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = meta["fps"]
sc.frame_start, sc.frame_end = f0, f1
sc.unit_settings.system = "METRIC"


def material(name, rgba, rough=0.45, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgba[:3], 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return v, f


def prim(kind, size):
    """A primitive geom as a mesh, in MuJoCo's size convention."""
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=2)
        o = bpy.context.object
        o.scale = size[:3]
    elif kind in ("sphere", "ellipsoid"):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=24, ring_count=12)
        o = bpy.context.object
        o.scale = (size[0],) * 3 if kind == "sphere" else size[:3]
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=24)
        o = bpy.context.object
    else:                                       # capsule: a cylinder with sphere caps, near enough
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1] + 2 * size[0], vertices=24)
        o = bpy.context.object
    bpy.ops.object.transform_apply(scale=True)
    return o


# ---- the duck
mats, meshes, objs = {}, {}, []
duck = bpy.data.collections.new("duck")
sc.collection.children.link(duck)
for j, g in enumerate(meta["geoms"]):
    key = tuple(round(x, 3) for x in g["rgba"][:3])
    if key not in mats:
        bright = sum(key) / 3
        mats[key] = material(f"m{len(mats)}", g["rgba"], rough=0.35 if bright > 0.8 else 0.5)
    if g["type"] == "mesh":
        if g["mesh"] not in meshes:
            v, f = load_obj(a.export / g["mesh"])
            me = bpy.data.meshes.new(g["mesh"])
            me.from_pydata(v, [], f)
            me.update()
            try:
                me.shade_smooth()
                me.set_sharp_from_angle(angle=math.radians(35))
            except (AttributeError, TypeError):
                pass
            meshes[g["mesh"]] = me
        o = bpy.data.objects.new(g["name"], meshes[g["mesh"]])
        duck.objects.link(o)
    else:
        o = prim(g["type"], g["size"])
        o.name = g["name"]
        for c in o.users_collection:
            c.objects.unlink(o)
        duck.objects.link(o)
    o.data.materials.clear()
    o.data.materials.append(mats[key])
    o.rotation_mode = "QUATERNION"
    objs.append(o)

print("keyframing", len(objs), "geoms x", f1 - f0 + 1, "frames", flush=True)
for t in range(f0 - 1, f1):
    for o, p in zip(objs, pose[t]):
        o.location = p[:3]
        o.rotation_quaternion = p[3:]
        o.keyframe_insert("location", frame=t + 1)
        o.keyframe_insert("rotation_quaternion", frame=t + 1)

# ---- floor, box
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
floor = bpy.context.object
fm = material("floor", (0.3, 0.3, 0.32, 1), rough=0.7)
nt = fm.node_tree
chk = nt.nodes.new("ShaderNodeTexChecker")
chk.inputs["Scale"].default_value = 120
chk.inputs["Color1"].default_value = (0.30, 0.30, 0.32, 1)
chk.inputs["Color2"].default_value = (0.26, 0.26, 0.28, 1)
nt.links.new(chk.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
floor.data.materials.append(fm)

if meta.get("box"):
    bx, top = meta["box"]
    bpy.ops.mesh.primitive_cube_add(size=1, location=(bx, 0, top / 2))
    box = bpy.context.object
    box.scale = (0.4, 1.2, top)
    bpy.ops.object.transform_apply(scale=True)
    box.data.materials.append(material("box", (0.55, 0.6, 0.7, 1)))
    mod = box.modifiers.new("Fluid", "FLUID")
    mod.fluid_type = "EFFECTOR"
if meta.get("puddle"):
    px, half = meta["puddle"]
    bpy.ops.mesh.primitive_plane_add(size=1, location=(px, 0, 0.001))
    pud = bpy.context.object
    pud.scale = (2 * half, 1.2, 1)
    pm = material("water", (0.2, 0.45, 0.8, 1), rough=0.05)
    pud.data.materials.append(pm)

# ---- smoke domain around everything the duck and its jets reach
lo = np.minimum(pose[f0 - 1:f1, :, :3].min(axis=(0, 1)), noz[f0 - 1:f1, :, :3].min(axis=(0, 1)))
hi = np.maximum(pose[f0 - 1:f1, :, :3].max(axis=(0, 1)), noz[f0 - 1:f1, :, :3].max(axis=(0, 1)))
lo = np.array([lo[0] - 0.6, lo[1] - 0.6, 0.0])
hi = np.array([hi[0] + 0.6, hi[1] + 0.6, hi[2] + 0.3])
vs = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
fs = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
dme = bpy.data.meshes.new("domain")
dme.from_pydata(vs, [], fs)
dom = bpy.data.objects.new("smoke_domain", dme)
sc.collection.objects.link(dom)
dmod = dom.modifiers.new("Fluid", "FLUID")
dmod.fluid_type = "DOMAIN"
ds = dmod.domain_settings
ds.domain_type = "GAS"
ds.resolution_max = a.res
for side in ("front", "back", "left", "right", "top"):
    setattr(ds, f"use_collision_border_{side}", False)      # smoke leaves the box ...
ds.use_collision_border_bottom = True                       # ... but the floor is a wall
ds.alpha, ds.beta = 0.0, 0.4             # no density buoyancy; warm exhaust rises once it slows
ds.vorticity = 0.15
ds.use_dissolve_smoke = True
ds.dissolve_speed = 70
ds.use_dissolve_smoke_log = True
ds.use_adaptive_timesteps = True
ds.timesteps_max = 12
ds.cfl_condition = 3.0
ds.cache_type = "ALL"
ds.cache_directory = str(a.out / "cache")
ds.cache_frame_start, ds.cache_frame_end = f0, f1
if a.noise:
    ds.use_noise = True
    ds.noise_scale = a.noise
    ds.noise_strength = 1.2
vm = bpy.data.materials.new("smoke")
vm.use_nodes = True
vn = vm.node_tree.nodes
vn.remove(vn["Principled BSDF"])
pv = vn.new("ShaderNodeVolumePrincipled")
pv.inputs["Color"].default_value = (0.92, 0.92, 0.94, 1)
pv.inputs["Density"].default_value = 60.0
pv.inputs["Anisotropy"].default_value = 0.2
vm.node_tree.links.new(pv.outputs["Volume"], vn["Material Output"].inputs["Volume"])
dom.data.materials.append(vm)

# ---- the jets: an inflow disc at each nozzle exit, normal along the exhaust
for k in range(noz.shape[1]):
    n_seg = 16
    r = 0.026
    v = [(0, 0, 0)] + [(r * math.cos(2 * math.pi * i / n_seg), r * math.sin(2 * math.pi * i / n_seg), 0)
                       for i in range(n_seg)]
    f = [(0, 1 + i, 1 + (i + 1) % n_seg) for i in range(n_seg)]
    me = bpy.data.meshes.new(f"jet{k}")
    me.from_pydata(v, [], f)
    e = bpy.data.objects.new(f"jet{k}", me)
    sc.collection.objects.link(e)
    e.hide_render = True
    e.rotation_mode = "QUATERNION"
    fmod = e.modifiers.new("Fluid", "FLUID")
    fmod.fluid_type = "FLOW"
    fl = fmod.flow_settings
    fl.flow_type, fl.flow_behavior, fl.flow_source = "SMOKE", "INFLOW", "MESH"
    fl.use_plane_init = True
    fl.surface_distance = 2.0
    fl.use_initial_velocity = True
    fl.smoke_color = (0.9, 0.9, 0.92)
    fl.temperature = 1.0
    fl.subframes = 2
    for t in range(f0 - 1, f1):
        p, axis, u = noz[t, k, :3], Vector(noz[t, k, 3:6]), float(noz[t, k, 6])
        e.location = p
        e.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(-axis)   # disc normal = exhaust
        e.keyframe_insert("location", frame=t + 1)
        e.keyframe_insert("rotation_quaternion", frame=t + 1)
        fl.velocity_normal = a.vmax * u
        fl.density = 1.0 if u > 0.02 else 0.0
        fl.keyframe_insert("velocity_normal", frame=t + 1)
        fl.keyframe_insert("density", frame=t + 1)

# ---- light, world, camera
world = bpy.data.worlds.new("sky")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.32, 0.36, 0.44, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4.5
sun.data.angle = math.radians(4)
sun.rotation_euler = (math.radians(35), 0, math.radians(-40))
sc.collection.objects.link(sun)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.sensor_fit = "VERTICAL"
cam.data.angle = math.radians(42)
cam.data.clip_start = 0.02
sc.collection.objects.link(cam)
sc.camera = cam
centre = pose[:, :, :3].mean(axis=1)
look = centre[f0 - 1].copy()
fps = meta["fps"]
for t in range(f0 - 1, f1):
    look += (1 - 0.7 ** (15 / fps)) * (centre[t] - look)     # fly.py's camera smoothing, per frame
    az = math.radians(140 + 25 * math.sin(0.25 * t / fps))
    el = math.radians(-14)
    fwd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    tgt = Vector(look) + Vector((0, 0, 0.02))
    cam.location = tgt - 1.05 * fwd
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = fwd.to_track_quat("-Z", "Y")
    cam.keyframe_insert("location", frame=t + 1)
    cam.keyframe_insert("rotation_quaternion", frame=t + 1)

# ---- render settings
r = sc.render
r.engine = "BLENDER_EEVEE"
r.resolution_x, r.resolution_y = a.size
r.resolution_percentage = 100
r.image_settings.file_format = "PNG"
r.filepath = str(a.out / "frames" / "f")
ee = sc.eevee
for name, val in (("taa_render_samples", a.samples), ("volumetric_tile_size", "2"),
                  ("volumetric_samples", 96), ("volumetric_start", 0.05), ("volumetric_end", 6.0),
                  ("use_volumetric_shadows", True), ("volumetric_shadow_samples", 32),
                  ("use_shadows", True), ("use_raytracing", True)):
    if hasattr(ee, name):
        try:
            setattr(ee, name, val)
        except (TypeError, ValueError):
            pass
try:
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
except TypeError:
    pass

bpy.ops.wm.save_as_mainfile(filepath=str(a.out / "scene.blend"))
if not a.skip_bake:
    print(f"baking smoke: grid {list(ds.domain_resolution)} frames {f0}-{f1}", flush=True)
    with bpy.context.temp_override(scene=sc, active_object=dom, object=dom, selected_objects=[dom]):
        bpy.ops.fluid.bake_all()
    bpy.ops.wm.save_mainfile()
print("rendering", flush=True)
bpy.ops.render.render(animation=True)
print("done", a.out / "frames")
