"""Simulation checks: does the robot stand, and how hard are the servos working?"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def _set_joints(m, d, q: dict):
    for name, val in q.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = val


def _lowest_collision_z(m, d) -> float:
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        if m.geom_bodyid[g] == 0:
            continue  # the floor
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            zmin = min(zmin, w[:, 2].min())
        else:
            zmin = min(zmin, d.geom_xpos[g][2] - m.geom_rbound[g])
    return float(zmin)


def _support_margin(points_xy: np.ndarray, com_xy: np.ndarray) -> float | None:
    """Signed distance (m) from the CoM to the edge of the support polygon (+ inside)."""
    if len(points_xy) < 3:
        return None
    try:
        hull = ConvexHull(points_xy)
    except Exception:
        return None
    # hull.equations: a*x + b*y + c <= 0 inside, with (a, b) unit normals
    return float(-(hull.equations[:, :2] @ com_xy + hull.equations[:, 2]).max())


def constrained(m) -> bool:
    """True when the model has loop or coupling constraints (poses must be solved, not set)."""
    return m.neq > 0


def driven_joints(m) -> list[str]:
    """Joints you set directly: every hinge in an unconstrained model; otherwise only the
    actuated ones (passive pins in loops and coupled followers are worked out)."""
    hinges = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if not constrained(m):
        return hinges
    acted = {m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu) if m.actuator_trntype[i] == 0}
    return [n for n in hinges if n in acted]


def solve_pose(m, d, q: dict, max_steps: int = 4000) -> np.ndarray:
    """Full qpos for a pose given by joint values. Unconstrained models: set and done.
    Constrained ones (loops, couplings): actuators hold their joints at the requested values
    and the rest settles under the constraints and joint springs, with gravity and contacts
    off: the pose the mechanism takes with nothing touching it."""
    mujoco.mj_resetData(m, d)
    _set_joints(m, d, q)
    if not constrained(m):
        mujoco.mj_kinematics(m, d)
        return d.qpos.copy()
    g, flags = m.opt.gravity.copy(), m.opt.disableflags
    m.opt.gravity[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    try:
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            d.ctrl[i] = d.qpos[m.jnt_qposadr[j]]
        for k in range(max_steps):
            mujoco.mj_step(m, d)
            if k > 200 and np.abs(d.qvel).max() < 1e-5:
                break
    finally:
        m.opt.gravity[:], m.opt.disableflags = g, flags
    return d.qpos.copy()


def place_on_floor(m, d, q: dict, clearance: float = 0.0005):
    """Set the joint pose and drop the floating base so the feet just touch the floor."""
    mujoco.mj_resetData(m, d)
    _set_joints(m, d, q)
    mujoco.mj_kinematics(m, d)
    d.qpos[2] -= _lowest_collision_z(m, d) - clearance
    mujoco.mj_forward(m, d)


def _hold(m, d, pose: dict, targets: np.ndarray, seconds: float) -> dict:
    place_on_floor(m, d, pose)
    d.ctrl[:] = targets
    root = m.jnt_bodyid[0]
    z0 = float(d.qpos[2])
    n = int(seconds / m.opt.timestep)
    tau = np.zeros((min(int(1.0 / m.opt.timestep), n), m.nu))
    tilt_max = 0.0
    for k in range(n):
        mujoco.mj_step(m, d)
        up = d.xmat[root].reshape(3, 3)[:, 2]
        tilt_max = max(tilt_max, float(np.arccos(np.clip(up[2], -1, 1))))
        if k >= n - len(tau):
            tau[k - (n - len(tau))] = d.actuator_force
    R = d.xmat[root].reshape(3, 3)
    pts = [d.contact[i].pos[:2].copy() for i in range(d.ncon)
           if 0 in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])]
    margin = _support_margin(np.array(pts), d.subtree_com[root][:2]) if pts else None
    q = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] for i in range(m.nu)])
    return {
        "fallen": tilt_max > 0.6 or d.qpos[2] < 0.5 * z0,
        "tilt_max": tilt_max,
        "pitch_deg": float(np.degrees(np.arctan2(-R[2, 0], R[2, 2]))),  # + = nose down
        "roll_deg": float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))),
        "drop_mm": (z0 - float(d.qpos[2])) * 1000,
        "margin": margin,
        "tau": tau,
        "q": q,
        "qpos": d.qpos.copy(),
    }


def stand_test(scene_xml: Path, pose: dict, seconds: float = 3.0) -> dict:
    """Hold `pose` with the position servos alone and report whether the robot stays
    up, where it settles, how much torque each servo uses, and whether the servos
    are stiff enough to hold the pose upright without an active balance controller."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return {"skipped": "fixed-base model"}
    names = [m.actuator(i).name for i in range(m.nu)]
    goal = np.array([pose.get(n, 0.0) for n in names])
    h = _hold(m, d, pose, goal, seconds)

    # Passive stiffness check (sagittal plane): each leg's pitch chain is springs in
    # series; two legs in parallel; compared with gravity's toppling stiffness m*g*h.
    place_on_floor(m, d, pose)
    root = m.jnt_bodyid[0]
    mgh = m.body_subtreemass[root] * 9.81 * d.subtree_com[root][2]
    kp = m.actuator_gainprm[:, 0]
    chain = [i for i, n in enumerate(names) if any(k in n for k in ("hip_pitch", "knee", "ankle"))]
    per_leg = len(chain) // 2 or 1
    k_leg = 1 / sum(1 / kp[i] for i in chain[:per_leg]) if chain else 0.0
    ratio = 2 * k_leg / mgh if mgh > 0 else float("inf")

    frange = m.actuator_forcerange[:, 1]
    load = np.abs(h["tau"]).mean(0) / frange
    peak = np.abs(h["tau"]).max(0) / frange
    worst = int(np.argmax(load))
    r1 = lambda x: None if x is None else round(float(x), 1)
    return {
        "stands": not h["fallen"],
        "seconds": seconds,
        "settled_pitch_deg": r1(h["pitch_deg"]),   # + = nose down
        "settled_roll_deg": r1(h["roll_deg"]),
        "height_drop_mm": r1(h["drop_mm"]),
        "support_margin_mm": None if h["margin"] is None else r1(h["margin"] * 1000),
        "sag_rad": {n: round(float(v), 3) for n, v in zip(names, h["q"] - goal)},
        "servo_load": {n: round(float(v), 2) for n, v in zip(names, load)},
        "servo_peak": {n: round(float(v), 2) for n, v in zip(names, peak)},
        "worst_servo": {"name": names[worst], "load": round(float(load[worst]), 2)},
        "pd_stiffness_ratio": round(float(ratio), 2),
        "note": ("servo stiffness alone cannot hold this pose upright (ratio < 1): it leans until "
                 "the feet stop it; an active balance controller (e.g. an RL policy) is required")
                if ratio < 1 else "servo stiffness alone can hold this pose upright",
        "final_qpos": h["qpos"].tolist(),
    }


def arm_test(scene_xml: Path, pose: dict, tool_site: str, payload_kg: float = 0.0) -> dict:
    """Static check for a fixed-base arm: the torque each servo needs to hold `pose`
    (its own weight plus `payload_kg` at the tool site), how far the pose sags under
    the servos' position gain, and the reach from the first joint to the tool."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    _set_joints(m, d, pose)
    mujoco.mj_forward(m, d)                  # qvel = 0, so qfrc_bias is gravity only
    sid = m.site(tool_site).id
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, sid)
    f_payload = np.array([0.0, 0.0, -payload_kg * 9.81])
    tau = d.qfrc_bias - jacp.T @ f_payload   # what the actuators must supply
    out = {"tool_site": tool_site, "payload_kg": payload_kg, "torque_nm": {}, "load": {}, "sag_rad": {}}
    worst = (None, -1.0)
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        t = float(tau[m.jnt_dofadr[j]])
        load = abs(t) / m.actuator_forcerange[i, 1]
        name = m.actuator(i).name
        out["torque_nm"][name] = round(t, 3)
        out["load"][name] = round(load, 2)
        out["sag_rad"][name] = round(abs(t) / max(m.actuator_gainprm[i, 0], 1e-9), 3)
        if load > worst[1]:
            worst = (name, load)
    first = m.jnt_bodyid[0]
    out["reach_mm"] = round(float(np.linalg.norm(d.site_xpos[sid] - d.xanchor[0])) * 1000, 1)
    out["tool_mm"] = [round(v * 1000, 1) for v in d.site_xpos[sid]]
    out["worst_servo"] = {"name": worst[0], "load": round(worst[1], 2)}
    out["holds"] = bool(worst[1] <= 1.0)
    return out
