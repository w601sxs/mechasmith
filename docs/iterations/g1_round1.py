"""Unitree G1 humanoid (29 DOF), ROUND 1: the library as it was after the Go2.

Pelvis, a 3-joint waist, torso with head, 6-joint legs and 7-joint arms. Joint anchors, axes
(the hip roll, hip yaw and shoulder pitch are tilted), ranges, torque limits and link masses
measured from the Menagerie `unitree_g1` model; the parts are our own.

Workarounds the API forced:
  * every link is drawn by hand as a motor hub on its own joint axis plus a strut to the next
    joint, so neighbouring links overlap at every joint;
  * Unitree's mirror convention is per joint (pitch axes keep their direction and range, roll
    and yaw axes keep their direction and mirror the range), and no mirror mode does that;
  * the waist's roll and pitch axes cross at one point and need a hand-made cross piece;
  * position servos need kv by hand (Menagerie uses dampratio=1); sensor noise has no field.
"""
import numpy as np

from mechasmith import (ALUMINIUM, BOUGHT, ActuatorModel, Assembly, Component, box, color, cyl, design, param)

DARK, METAL = color("#333333"), color("#b3b3b3")
X, Y, Z = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)


def servo_class(force, kv):
    return ActuatorModel(kp=500.0, kv=kv, forcerange=force, damping=0.0, frictionloss=0.3, armature=0.01,
                         max_velocity=20.0, source="Menagerie unitree_g1 (kp 500, forcerange per motor class)")


# name, parent, anchor (mm, pelvis frame), axis, range, (torque, kv), link mass kg, next anchor (strut end)
LEG = [
    ("left_hip_pitch", "pelvis", (0, 64.5, -102.7), Y, (-2.5307, 2.8798), (88, 10.3), 1.35, (0, 116.5, -133.2)),
    ("left_hip_roll", "left_hip_pitch", (0, 116.5, -133.2), (0.985, 0, 0.174), (-0.5236, 2.9671), (139, 15.8), 1.52,
     (46.2, 116.5, -251.0)),
    ("left_hip_yaw", "left_hip_roll", (46.2, 116.5, -251.0), (-0.174, 0, 0.985), (-2.7576, 2.7576), (88, 5.8), 1.702,
     (0, 118.6, -439.3)),
    ("left_knee", "left_hip_yaw", (0, 118.6, -439.3), Y, (-0.087267, 2.8798), (139, 9.1), 1.932, (0, 118.5, -739.3)),
    ("left_ankle_pitch", "left_knee", (0, 118.5, -739.3), Y, (-0.87267, 0.5236), (50, 4.9), 0.074, (0, 118.5, -756.9)),
    ("left_ankle_roll", "left_ankle_pitch", (0, 118.5, -756.9), X, (-0.2618, 0.2618), (50, 4.5), 0.608, None),
]
WAIST = [
    ("waist_yaw", "pelvis", (0, 0, 0), Z, (-2.618, 2.618), (88, 10.3), 0.214, (-4, 0, 44)),
    ("waist_roll", "waist_yaw", (-4, 0, 44), X, (-0.52, 0.52), (50, 16.2), 0.086, None),
    ("waist_pitch", "waist_roll", (-4, 0, 44), Y, (-0.52, 0.52), (50, 9.3), 7.818, None),
]
ARM = [
    ("left_shoulder_pitch", "waist_pitch", (0, 100.2, 291.8), (0, 0.961, 0.276), (-3.0892, 2.6704), (25, 10.8), 0.718,
     (0, 140.6, 289.0)),
    ("left_shoulder_roll", "left_shoulder_pitch", (0, 140.6, 289.0), X, (-1.5882, 2.2515), (25, 10.0), 0.643,
     (0, 146.8, 185.8)),
    ("left_shoulder_yaw", "left_shoulder_roll", (0, 146.8, 185.8), Z, (-2.618, 2.618), (25, 7.3), 0.734,
     (15.8, 146.8, 105.2)),
    ("left_elbow", "left_shoulder_yaw", (15.8, 146.8, 105.2), Y, (-1.0472, 2.0944), (25, 7.1), 0.6, (115.8, 148.7, 95.2)),
    ("left_wrist_roll", "left_elbow", (115.8, 148.7, 95.2), X, (-1.9722, 1.9722), (25, 4.6), 0.085,
     (153.8, 148.7, 95.2)),
    ("left_wrist_pitch", "left_wrist_roll", (153.8, 148.7, 95.2), Y, (-1.6144, 1.6144), (5, 5.0), 0.484,
     (199.8, 148.7, 95.2)),
    ("left_wrist_yaw", "left_wrist_pitch", (199.8, 148.7, 95.2), Z, (-1.6144, 1.6144), (5, 4.7), 0.255, None),
]
STAND = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28,
         "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28}


def strut(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return cyl((p0 + p1) / 2, p1 - p0, r, float(np.linalg.norm(p1 - p0)))


@design(params=dict(pelvis_g=param(3813.0, (2000, 6000), "pelvis with its electronics")), home=STAND)
def build(p):
    pelvis = Component.world("pelvis", box((-55, -66, -150), (58, 66, -28)) + cyl((0, 0, -14), Z, 30, 28), BOUGHT,
                             DARK, collision="box", mass_override=p.pelvis_g / 1000)
    asm = Assembly("g1_29dof", root=pelvis, floating=True, root_link="pelvis")
    pelvis.sites["imu_in_pelvis"] = (45.2, 0.0, -83.4)
    parts = {"pelvis": pelvis}
    for name, parent, at, axis, rng, (force, kv), mass, nxt in LEG + WAIST + ARM:
        shape = cyl(at, axis, 28 if force > 30 else 20, 40)                       # the motor hub
        if nxt is not None and np.linalg.norm(np.subtract(nxt, at)) > 1:
            shape = shape + strut(at, nxt, 16 if force > 30 else 11)
        if name == "waist_pitch":                                                 # torso and head
            shape = shape + box((-70, -105, 40), (78, 105, 330)) + box((-60, -70, 330), (70, 70, 520))
        if name == "left_ankle_roll":                                             # foot
            shape = shape + box((-60, 85, -790), (135, 152, -765))
        if name == "left_wrist_yaw":                                              # hand
            shape = shape + box((200, 110, 60), (370, 168, 150))
        child = Component.world(f"{name}_part", shape, BOUGHT, METAL if "hip" in name else DARK,
                                collision="capsule" if nxt else "box", mass_override=mass)
        asm.hinge(f"{name}_joint", parts[parent], child, at, axis, rng, actuator=servo_class(force, kv),
                  child_link=f"{name}_link")
        parts[name] = child
    parts["left_ankle_roll"].sites["left_foot"] = (0.0, 118.5, -756.9)
    parts["waist_pitch"].sites["imu_in_torso"] = (-43.6, -2.2, 191.9)
    asm.imu("imu_in_pelvis", ("gyro", "accelerometer"))
    asm.imu("imu_in_torso", ("gyro", "accelerometer"))
    return asm.robot().mirror("left_", "right_", axes="same")
