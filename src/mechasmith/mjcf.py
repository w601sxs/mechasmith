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
EQ_SOLREF, EQ_SOLIMP = [0.005, 1.0], [0.95, 0.99, 0.001, 0.5, 2.0]   # stiff loop and coupling constraints


def _export_part_mesh(part, origin_mm, path: Path):
    from build123d import Pos
    shape = Pos(*(-origin_mm)) * part.shape
    export_stl(shape, str(path), tolerance=0.05, angular_tolerance=0.15)


def _primitive(part, origin_mm):
    """A primitive collision geom fitted to a part's box in its body frame:
    (type, size, pos, quat) in metres."""
    bb = part.shape.bounding_box()
    lo, hi = np.array([bb.min.X, bb.min.Y, bb.min.Z]), np.array([bb.max.X, bb.max.Y, bb.max.Z])
    c, h = ((lo + hi) / 2 - origin_mm) * 1e-3, (hi - lo) / 2 * 1e-3
    kind = part.collision
    if kind == "box":
        return mujoco.mjtGeom.mjGEOM_BOX, h, c, [1, 0, 0, 0]
    if kind == "sphere":
        return mujoco.mjtGeom.mjGEOM_SPHERE, [h.max(), 0, 0], c, [1, 0, 0, 0]
    if kind in ("cylinder", "capsule"):
        k = int(np.argmax(h))
        r = max(h[i] for i in range(3) if i != k)
        half = h[k] - (r if kind == "capsule" else 0.0)
        quat = {0: [0.70710678, 0, 0.70710678, 0], 1: [0.70710678, -0.70710678, 0, 0], 2: [1, 0, 0, 0]}[k]
        t = mujoco.mjtGeom.mjGEOM_CYLINDER if kind == "cylinder" else mujoco.mjtGeom.mjGEOM_CAPSULE
        return t, [r, max(half, 1e-4), 0], c, quat
    raise ValueError(f"part {part.name}: collision must be True, 'mesh', 'box', 'sphere', 'cylinder' or 'capsule'")


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
            jnt = body.add_joint(
                name=pj.name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=pj.axis,
                range=pj.range, limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                damping=pj.damping if pj.damping is not None else (a.damping if a else 0.0),
                frictionloss=a.frictionloss if a else 0.0,
                armature=pj.armature if pj.armature is not None else (a.armature if a else 0.0),
            )
            if pj.stiffness:
                jnt.stiffness, jnt.springref = [pj.stiffness, 0.0, 0.0], pj.springref   # linear spring

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
                if part.collision in (True, "mesh"):
                    geom = body.add_geom(name=f"{part.name}_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                                         meshname=part.name, group=COLLISION_GROUP, density=0,
                                         rgba=(0.9, 0.2, 0.2, 0.4))
                else:
                    t, size, pos, quat = _primitive(part, origin)
                    geom = body.add_geom(name=f"{part.name}_collision", type=t, size=list(size), pos=list(pos),
                                         quat=quat, group=COLLISION_GROUP, density=0, rgba=(0.9, 0.2, 0.2, 0.4))
                for k, v in (part.contact or {}).items():
                    setattr(geom, k, v)
            pm, _, _ = part.mass_properties()
            parts_meta.append({"name": part.name, "mesh": f"meshes/{fname}", "mass": pm,
                               "material": part.material.name, "collision": part.collision,
                               "color": list(part.color)})

        for sname, spos in link.sites.items():
            r = getattr(robot, "site_radius", {}).get(sname)
            body.add_site(name=sname, pos=(np.asarray(spos) - origin) * 1e-3, group=3,
                          **({"size": [r * 1e-3, 0, 0]} if r else {}))

        manifest["bodies"][link.name] = {"origin_mm": origin.tolist(), "mass": m, "parts": parts_meta,
                                         "parent": pj.parent.name if pj else None,
                                         "joint": None if pj is None else {"name": pj.name, "axis": pj.axis.tolist(),
                                                                          "range": list(pj.range)}}
        for j in robot.children(link):
            add_link(j.child, body, origin)

    add_link(robot.root, spec.worldbody, np.zeros(3))

    # Closed loops: each pin becomes a connect constraint (its anchor in the first body's frame;
    # MuJoCo finds the matching point in the second at the home pose). Pinned bodies do not collide.
    for la, lb, p in robot.loops:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=la.name,
                          name2=lb.name, data=list((p - robot.body_origin(la)) * 1e-3) + [0.0] * 9,
                          solref=EQ_SOLREF, solimp=EQ_SOLIMP)
        spec.add_exclude(bodyname1=la.name, bodyname2=lb.name)
    for f, l, r in robot.couplings:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f, name2=l,
                          data=[0.0, r, 0.0, 0.0, 0.0] + [0.0] * 7, solref=EQ_SOLREF, solimp=EQ_SOLIMP)
    # Rotors: a site actuator each, thrust along the axis and the reaction torque about it.
    for r in robot.rotors:
        act = spec.add_actuator(name=r.name, target=r.name, trntype=mujoco.mjtTrn.mjTRN_SITE)
        act.gear = list(r.axis) + list(-r.spin * r.torque_ratio * r.axis)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [0.0, r.thrust_max]
    kinds = {"gyro": mujoco.mjtSensor.mjSENS_GYRO, "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
             "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT, "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
             "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL, "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
             "touch": mujoco.mjtSensor.mjSENS_TOUCH}
    for kind, site in robot.sensors:
        spec.add_sensor(name=f"{site}_{kind}", type=kinds[kind], objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
    for j in robot.joints:
        a = j.actuator
        if a is None:
            continue
        act = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-a.forcerange, a.forcerange]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        if a.mode == "torque":
            act.set_to_motor()
            act.ctrlrange = [-a.forcerange, a.forcerange]
        else:
            act.set_to_position(kp=a.kp, kv=a.kv)
            act.ctrlrange = list(j.range)

    manifest["joints"] = [{"name": j.name, "parent": j.parent.name, "child": j.child.name,
                           "anchor_mm": j.anchor.tolist(), "axis": j.axis.tolist(),
                           "range": list(j.range),
                           "actuator": (j.actuator.source if j.actuator else None)} for j in robot.joints]
    manifest["actuators"] = {j.name: {"mode": j.actuator.mode, "kp": j.actuator.kp, "kv": j.actuator.kv,
                                      "effort": j.actuator.forcerange, "velocity": j.actuator.max_velocity,
                                      "armature": j.armature if j.armature is not None else j.actuator.armature,
                                      "source": j.actuator.source}
                             for j in robot.joints if j.actuator is not None}
    manifest["rotors"] = [{"name": r.name, "link": r.link.name, "pos_mm": r.pos.tolist(), "axis": r.axis.tolist(),
                           "spin": r.spin, "thrust_max_n": r.thrust_max, "torque_ratio_m": r.torque_ratio,
                           "source": r.source} for r in robot.rotors]
    manifest["loops"] = [{"a": a.name, "b": b.name, "anchor_mm": p.tolist()} for a, b, p in robot.loops]
    manifest["couplings"] = [{"follower": f, "leader": l, "ratio": r} for f, l, r in robot.couplings]
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


def allocation(model, data=None):
    """Rotor allocation: the 4 x n matrix taking rotor thrusts to [thrust, roll, pitch, yaw]
    moments about the centre of mass (body frame), and the rotor limits."""
    import numpy as np
    d = data if data is not None else mujoco.MjData(model)
    if data is None:
        mujoco.mj_forward(model, d)
    root = 1
    com = d.subtree_com[root]
    cols, hi = [], []
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_SITE:
            continue
        s = model.actuator_trnid[i, 0]
        R = d.site_xmat[s].reshape(3, 3)
        f, tau = R @ model.actuator_gear[i, :3], R @ model.actuator_gear[i, 3:]
        m = np.cross(d.site_xpos[s] - com, f) + tau
        Rb = d.xmat[root].reshape(3, 3)
        cols.append(np.r_[Rb.T @ f, Rb.T @ m][[2, 3, 4, 5]])
        hi.append(model.actuator_ctrlrange[i, 1])
    return (np.array(cols).T, np.array(hi)) if cols else (None, None)


def hover_thrusts(model):
    """Rotor thrusts that hold the robot still in the air (None if out of range)."""
    import numpy as np
    A, hi = allocation(model)
    if A is None:
        return None
    w = np.array([model.body_subtreemass[1] * -model.opt.gravity[2], 0, 0, 0])
    u = np.linalg.lstsq(A, w, rcond=None)[0]
    return u if np.all(u >= 0) and np.all(u <= hi) else None


def lowest_point_mm(robot: Robot) -> float:
    return min(p.shape.bounding_box().min.Z for l in robot.links.values() for p in l.parts)


def export(robot: Robot, out_dir, base_height: float | str = "auto", keyframes: dict | None = None) -> Path:
    """base_height: metres to lift the root, or "auto" to rest the lowest geometry on z=0.
    keyframes: {name: joint pose}; each becomes an MJCF keyframe (a floating robot is set down
    on the floor; position actuators hold the pose; torque motors start at zero)."""
    out_dir = Path(out_dir)
    if base_height == "auto":
        base_height = -lowest_point_mm(robot) * 1e-3
    spec = build_spec(robot, out_dir, base_height)
    abs_mesh = str((out_dir / "meshes").resolve())
    spec.meshdir = abs_mesh
    model = spec.compile()  # raises with a clear message if the model is invalid
    if robot.rotors and robot.floating:
        u = hover_thrusts(model)
        if u is not None:
            key = spec.add_key(name="hover")
            key.qpos = list(model.qpos0)
            key.ctrl = list(u)
            model = spec.compile()
    if keyframes and not robot.rotors and not model.neq:
        from .sim import place_on_floor
        d = mujoco.MjData(model)
        for name, pose in keyframes.items():
            if robot.floating:
                place_on_floor(model, d, pose)
            else:
                mujoco.mj_resetData(model, d)
                for jn, v in pose.items():
                    d.qpos[model.jnt_qposadr[model.joint(jn).id]] = v
            ctrl = []
            for i in range(model.nu):
                jn = model.joint(model.actuator_trnid[i, 0]).name
                ctrl.append(float(d.qpos[model.jnt_qposadr[model.joint(jn).id]])
                            if model.actuator_biastype[i] != 0 else 0.0)
            key = spec.add_key(name=name)
            key.qpos, key.ctrl = list(d.qpos), ctrl
        spec.compile()
    xml = spec.to_xml().replace(f'meshdir="{abs_mesh}', 'meshdir="meshes')
    (out_dir / "robot.xml").write_text(xml)
    (out_dir / "scene.xml").write_text(SCENE.format(name=robot.name))
    mujoco.MjModel.from_xml_path(str(out_dir / "scene.xml"))  # round-trip check
    return out_dir / "scene.xml"
