"""Export a Robot to MJCF using MuJoCo's own MjSpec builder.

Body frames: every body has identity orientation and sits at its parent
joint's anchor, so world-frame authoring maps 1:1 onto the model at qpos=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from build123d import export_stl

from .core import Link, Robot

VISUAL_GROUP, COLLISION_GROUP = 2, 3


def _export_part_mesh(part, origin_mm, path: Path):
    from build123d import Pos
    shape = Pos(*(-origin_mm)) * part.shape
    export_stl(shape, str(path), tolerance=0.05, angular_tolerance=0.15)


def build_spec(robot: Robot, out_dir: Path, base_height: float = 0.0) -> mujoco.MjSpec:
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec()
    spec.modelname = robot.name
    spec.compiler.degree = False
    spec.meshdir = "meshes"
    spec.option.timestep = 0.002

    manifest = {"robot": robot.name, "bodies": {}}

    def add_link(link: Link, parent_body, parent_origin_mm):
        origin = robot.body_origin(link)
        pos = (origin - parent_origin_mm) * 1e-3
        if link is robot.root:
            pos = pos + np.array([0, 0, base_height])
        body = parent_body.add_body(name=link.name, pos=pos)
        if link is robot.root and robot.floating:
            body.add_freejoint(name=f"{link.name}_freejoint")

        # Inertial from CAD solids.
        m, com, inertia = link.mass_properties()
        if m > 0:
            body.explicitinertial = True
            body.mass = m
            body.ipos = com - origin * 1e-3
            body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                inertia[0, 1], inertia[0, 2], inertia[1, 2]]

        pj = robot.parent_joint(link)
        if pj is not None:
            a = pj.actuator
            body.add_joint(
                name=pj.name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=pj.axis,
                range=pj.range, limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                damping=a.damping if a else 0.0, frictionloss=a.frictionloss if a else 0.0,
                armature=a.armature if a else 0.0,
            )

        parts_meta = []
        for part in link.parts:
            fname = f"{part.name}.stl"
            _export_part_mesh(part, origin, mesh_dir / fname)
            mesh = spec.add_mesh(name=part.name, file=fname)
            mesh.scale = [1e-3] * 3
            body.add_geom(name=f"{part.name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=part.name, rgba=part.color, contype=0, conaffinity=0,
                          group=VISUAL_GROUP, density=0)
            if part.collision:
                body.add_geom(name=f"{part.name}_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=part.name, group=COLLISION_GROUP, density=0,
                              rgba=(0.9, 0.2, 0.2, 0.4))
            pm, _, _ = part.mass_properties()
            parts_meta.append({"name": part.name, "mesh": f"meshes/{fname}", "mass": pm,
                               "material": part.material.name, "collision": part.collision})

        for sname, spos in link.sites.items():
            body.add_site(name=sname, pos=(np.asarray(spos) - origin) * 1e-3, group=3)

        manifest["bodies"][link.name] = {"origin_mm": origin.tolist(), "mass": m, "parts": parts_meta}
        for j in robot.children(link):
            add_link(j.child, body, origin)

    add_link(robot.root, spec.worldbody, np.zeros(3))

    for j in robot.joints:
        a = j.actuator
        if a is None:
            continue
        act = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=a.kp, kv=a.kv)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-a.forcerange, a.forcerange]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = list(j.range)

    manifest["joints"] = [{"name": j.name, "parent": j.parent.name, "child": j.child.name,
                           "anchor_mm": j.anchor.tolist(), "axis": j.axis.tolist(),
                           "range": list(j.range),
                           "actuator": (j.actuator.source if j.actuator else None)} for j in robot.joints]
    manifest["total_mass"] = robot.total_mass()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return spec


SCENE = """<mujoco model="{name}_scene">
  <include file="robot.xml"/>
  <statistic center="0 0 0.1" extent="0.4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="150" elevation="-20" offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.96 0.96 0.97" rgb2="0.85 0.87 0.9" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.92 0.92 0.93" rgb2="0.88 0.88 0.9" markrgb="0.8 0.8 0.82" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.25 0.25 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid" condim="3" friction="0.8 0.02 0.01"/>
  </worldbody>
</mujoco>
"""


def lowest_point_mm(robot: Robot) -> float:
    return min(p.shape.bounding_box().min.Z for l in robot.links.values() for p in l.parts)


def export(robot: Robot, out_dir, base_height: float | str = "auto") -> Path:
    """base_height: metres to lift the root, or "auto" to rest the lowest geometry on z=0."""
    out_dir = Path(out_dir)
    if base_height == "auto":
        base_height = -lowest_point_mm(robot) * 1e-3
    spec = build_spec(robot, out_dir, base_height)
    abs_mesh = str((out_dir / "meshes").resolve())
    spec.meshdir = abs_mesh
    spec.compile()  # raises with a clear message if the model is invalid
    xml = spec.to_xml().replace(f'meshdir="{abs_mesh}', 'meshdir="meshes')
    (out_dir / "robot.xml").write_text(xml)
    (out_dir / "scene.xml").write_text(SCENE.format(name=robot.name))
    mujoco.MjModel.from_xml_path(str(out_dir / "scene.xml"))  # round-trip check
    return out_dir / "scene.xml"
