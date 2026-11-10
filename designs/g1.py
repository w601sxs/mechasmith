"""Unitree G1 humanoid (29 DOF) as mechasmith code.

Pelvis, a 3-joint waist, torso with head, 6-joint legs and 7-joint arms, written as joint
tables: `asm.chain` puts each joint's motor on the parent side and its output flange on the
child side and runs struts between them, so nothing overlaps by construction. Motors come in
the G1's five torque classes (139, 88, 50, 25 and 5 N*m); the waist roll and pitch and the
ankle pitch and roll are cross pieces, driven by motors elsewhere (on the real robot, through
linkages). Each link's measured mass is its budget: its motors take their share and the
structure the rest. Position servos, kp 500 with a damping ratio of 1, as in Menagerie; two IMUs
with Menagerie's noise and cutoffs; the left side is mirrored with Unitree's convention.

Joint anchors, axes (hip roll, hip yaw and shoulder pitch are tilted), ranges, torque limits and
link masses measured from the Menagerie `unitree_g1` model; the parts are our own.
"""
from mechasmith import (BOUGHT, ActuatorModel, Assembly, ChainJoint, Component, box, color, cyl, design, param)

DARK, METAL = color("#333333"), color("#b3b3b3")
X, Y, Z = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)
# Motor classes: peak torque N*m -> (hub radius, length mm, motor mass kg). Sizes and masses are
# estimates for Unitree's G1 joint motors (verify); torques from Menagerie.
MOTOR = {139: (32, 40, 1.0, 16), 88: (30, 38, 0.7, 16), 50: (26, 34, 0.45, 14), 25: (24, 28, 0.35, 11),
         5: (16, 26, 0.15, 9)}                 # radius, length, mass, strut radius


def servo(force):
    return ActuatorModel(kp=500.0, kv=0.0, forcerange=force, damping=0.0, frictionloss=0.3, armature=0.01,
                         max_velocity=20.0, dampratio=1.0, source="Menagerie unitree_g1 (kp 500, dampratio 1)")


def J(name, at, axis, rng, force, mass, hub=True, link=None, **kw):
    r, L, m, strut = MOTOR[force]
    return ChainJoint(f"{name}_joint", at, axis, rng, servo(force), hub=(r, L) if hub else None,
                      hub_mass=m if hub else 0.0, mass=mass, strut=strut, child_link=link or f"{name}_link", **kw)


FOOT = box((-60, 85, -790), (135, 152, -765))
HAND = box((240, 108, 55), (370, 168, 155))
TORSO = box((-70, -105, 60), (78, 105, 330)) + box((-60, -70, 330), (70, 70, 520))
FOOT_CONTACT = dict(friction=[0.6, 0.005, 0.0001], priority=1)
STAND = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28,
         "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28}


@design(params=dict(pelvis_g=param(3813.0, (2000, 6000), "pelvis with its electronics"),
                    torso_g=param(7818.0, (5000, 11000), "torso, head, battery and computer")), home=STAND)
def build(p):
    pelvis = Component.world("pelvis", box((-55, -66, -150), (58, 66, -28)) + cyl((0, 0, -20), Z, 30, 16), BOUGHT,
                             DARK, collision="box")
    pelvis.mass_budget = p.pelvis_g / 1000
    pelvis.sites["imu_in_pelvis"] = (45.2, 0.0, -83.4)
    asm = Assembly("g1_29dof", root=pelvis, floating=True, root_link="pelvis")
    asm.chain(pelvis, [
        J("left_hip_pitch", (0, 64.5, -102.7), Y, (-2.5307, 2.8798), 88, 1.35),
        J("left_hip_roll", (0, 116.5, -133.2), (0.985, 0, 0.174), (-0.5236, 2.9671), 139, 1.52),
        J("left_hip_yaw", (46.2, 116.5, -251.0), (-0.174, 0, 0.985), (-2.7576, 2.7576), 88, 1.702),
        J("left_knee", (0, 118.6, -439.3), Y, (-0.087267, 2.8798), 139, 1.932),
        J("left_ankle_pitch", (0, 118.5, -739.3), Y, (-0.87267, 0.5236), 50, 0.074, hub=False),
        J("left_ankle_roll", (0, 118.5, -756.9), X, (-0.2618, 0.2618), 50, 0.608, hub=False, add=(FOOT,),
          collision="box", contact=FOOT_CONTACT),
    ], color=METAL, motor_color=DARK)
    waist = asm.chain(pelvis, [
        J("waist_yaw", (0, 0, 0), Z, (-2.618, 2.618), 88, 0.214),
        J("waist_roll", (-4, 0, 44), X, (-0.52, 0.52), 50, 0.086, hub=False),
        J("waist_pitch", (-4, 0, 44), Y, (-0.52, 0.52), 50, p.torso_g / 1000, hub=False, add=(TORSO,),
          collision="box", link="torso_link"),
    ], color=DARK, motor_color=DARK)
    torso = waist["waist_pitch_joint"]
    torso.sites["imu_in_torso"] = (-43.6, -2.2, 191.9)
    arm = asm.chain(torso, [
        J("left_shoulder_pitch", (0, 100.2, 291.8), (0, 0.961, 0.276), (-3.0892, 2.6704), 25, 0.718),
        J("left_shoulder_roll", (0, 140.6, 289.0), X, (-1.5882, 2.2515), 25, 0.643),
        J("left_shoulder_yaw", (0, 146.8, 185.8), Z, (-2.618, 2.618), 25, 0.734),
        J("left_elbow", (15.8, 146.8, 105.2), Y, (-1.0472, 2.0944), 25, 0.6),
        J("left_wrist_roll", (115.8, 148.7, 95.2), X, (-1.9722, 1.9722), 25, 0.085),
        J("left_wrist_pitch", (153.8, 148.7, 95.2), Y, (-1.6144, 1.6144), 5, 0.484, hub=False),
        J("left_wrist_yaw", (199.8, 148.7, 95.2), Z, (-1.6144, 1.6144), 5, 0.255, add=(HAND,), collision="box"),
    ], color=METAL, motor_color=DARK)
    asm.components["left_ankle_roll_structure"].sites["left_foot"] = (0.0, 118.5, -756.9)
    noise, cutoff = {"gyro": 0.0005, "accelerometer": 0.01}, {"gyro": 34.9, "accelerometer": 157.0}
    asm.imu("imu_in_pelvis", ("gyro", "accelerometer"), noise, cutoff)
    asm.imu("imu_in_torso", ("gyro", "accelerometer"), noise, cutoff)
    return asm.robot().mirror("left_", "right_", axes="closest")
