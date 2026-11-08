"""Export a Robot to URDF (for Isaac Sim / Isaac Lab, ROS, PyBullet, Drake, ...).

Same tree, frames and meshes as the MJCF export: one URDF link per mechasmith link,
link frames with identity orientation at the parent joint's anchor, so the joint
origins are pure translations and the zero pose is the design's home pose.

* Meshes are the STL files the MJCF export wrote (mm, link frame); URDF scales them to m.
* Inertials come from the same CAD solids (mass, centre of mass, full inertia).
* Joints are `revolute` with the design's limits; effort and velocity come from the
  servo's actuator model, damping and friction too. Position-servo gains have no URDF
  field: they stay in the MJCF (and the manifest).
* A fixed-base robot is bolted to a `world` link; a floating one is left free.
* Sites (tool tips, feet, IMU, camera) become massless `<site>_frame` links on fixed joints.
* Coupled joints get `<mimic>`. URDF cannot express closed loops: the tree is exported and
  each loop pin is listed in a comment (the MJCF keeps them as constraints).
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .core import Link, Robot


def _f(c: float) -> str:
    c = float(c)
    return "0" if abs(c) < 1e-12 else f"{c:.6g}"


def _v(x) -> str:
    return " ".join(_f(c) for c in x)


def to_urdf(robot: Robot, mesh_dir: str = "meshes", base_height: float = 0.0) -> str:
    out = [f'<?xml version="1.0"?>', f"<robot name={quoteattr(robot.name)}>"]
    if not robot.floating:
        out += ['  <link name="world"/>',
                f'  <joint name="{robot.root.name}_to_world" type="fixed">',
                '    <parent link="world"/>', f'    <child link="{robot.root.name}"/>',
                f'    <origin xyz="0 0 {base_height:.6g}" rpy="0 0 0"/>', "  </joint>"]

    def link_xml(link: Link):
        origin = robot.body_origin(link)
        m, com, inertia = link.mass_properties()
        s = [f"  <link name={quoteattr(link.name)}>"]
        if m > 0:
            c = com - origin * 1e-3
            I = inertia
            s += ["    <inertial>", f'      <origin xyz="{_v(c)}" rpy="0 0 0"/>', f'      <mass value="{m:.6g}"/>',
                  f'      <inertia ixx="{_f(I[0, 0])}" ixy="{_f(I[0, 1])}" ixz="{_f(I[0, 2])}" '
                  f'iyy="{_f(I[1, 1])}" iyz="{_f(I[1, 2])}" izz="{_f(I[2, 2])}"/>', "    </inertial>"]
        for part in link.parts:
            mesh = f'<mesh filename="{mesh_dir}/{part.name}.stl" scale="0.001 0.001 0.001"/>'
            s += [f"    <visual name={quoteattr(part.name)}>", '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                  f"      <geometry>{mesh}</geometry>",
                  f'      <material name={quoteattr(part.name + "_color")}><color rgba="{_v(part.color)}"/></material>',
                  "    </visual>"]
            if part.collision:
                s += [f"    <collision name={quoteattr(part.name)}>", '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                      f"      <geometry>{mesh}</geometry>", "    </collision>"]
        s.append("  </link>")
        for site, pos in link.sites.items():
            s += [f'  <link name="{site}_frame"/>', f'  <joint name="{site}_frame_joint" type="fixed">',
                  f'    <parent link={quoteattr(link.name)}/>', f'    <child link="{site}_frame"/>',
                  f'    <origin xyz="{_v((np.asarray(pos) - origin) * 1e-3)}" rpy="0 0 0"/>', "  </joint>"]
        return s

    mimic = {f: (l, r) for f, l, r in robot.couplings}
    for a, b, p in robot.loops:
        out.append(f"  <!-- closed loop (not representable in URDF): {a.name} and {b.name} are pinned at "
                   f"{_v(np.asarray(p) * 1e-3)} m -->")

    def walk(link: Link):
        out.extend(link_xml(link))
        for j in robot.children(link):
            a = j.actuator
            lo, hi = j.range
            out.extend([
                f"  <joint name={quoteattr(j.name)} type=\"revolute\">",
                f"    <parent link={quoteattr(link.name)}/>", f"    <child link={quoteattr(j.child.name)}/>",
                f'    <origin xyz="{_v((j.anchor - robot.body_origin(link)) * 1e-3)}" rpy="0 0 0"/>',
                f'    <axis xyz="{_v(j.axis)}"/>',
                f'    <limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{(a.forcerange if a else 0):.6g}" '
                f'velocity="{(a.max_velocity if a else 0):.6g}"/>',
                f'    <dynamics damping="{(j.damping if j.damping is not None else (a.damping if a else 0)):.6g}" '
                f'friction="{(a.frictionloss if a else 0):.6g}"/>']
                + ([f'    <mimic joint="{mimic[j.name][0]}" multiplier="{mimic[j.name][1]:.6g}" offset="0"/>']
                   if j.name in mimic else []) + ["  </joint>"])
            walk(j.child)

    walk(robot.root)
    out.append("</robot>")
    return "\n".join(out) + "\n"


def export(robot: Robot, out_dir, base_height: float = 0.0) -> Path:
    """Write <out_dir>/robot.urdf next to the meshes the MJCF export wrote (run that first).
    `base_height` (m) lifts a fixed-base robot off its world link."""
    out_dir = Path(out_dir)
    missing = [p.name for l in robot.links.values() for p in l.parts if not (out_dir / "meshes" / f"{p.name}.stl").exists()]
    if missing:
        raise FileNotFoundError(f"urdf.export: meshes missing ({missing[:3]}...); run mjcf.export first")
    path = out_dir / "robot.urdf"
    path.write_text(to_urdf(robot, "meshes", base_height))
    return path
