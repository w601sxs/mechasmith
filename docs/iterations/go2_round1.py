"""Unitree Go2 quadruped, ROUND 1: the library as it was.

Torso, four legs (hip, thigh, calf), cylindrical joint actuators drawn by hand. Layout and
masses measured from the Menagerie `unitree_go2` model; the parts are our own.

Workarounds the API forced:
  * the only actuator is a position servo (kp), but the Go2's motors are torque-controlled,
    and there is no way to derive armature or torque limits from a motor's gear ratio;
  * each motor is a cylinder with a hand-set mass (no spec for a pancake actuator);
  * mirroring has two conventions and Unitree uses neither (same world axes on both sides);
  * collision is mesh-based, with no way to set foot friction or contact softness;
  * only an IMU exists for sensing; no home keyframe.
"""
import numpy as np

from mechasmith import (ALUMINIUM, BOUGHT, ActuatorModel, Assembly, Component, box, color, cyl, design, param,
                        symmetric)

BLACK, METAL, GRAY = color("#000000"), color("#e6f2f2"), color("#abb1c5")
X, Y = (1.0, 0, 0), (0, 1.0, 0)
# Unitree GO-M8010-6 (hip/thigh) and the knee's extra belt reduction, as Isaac Lab-style PD servos.
HIP = ActuatorModel(kp=25.0, kv=0.5, forcerange=23.7, damping=2.0, frictionloss=0.2, armature=0.01,
                    max_velocity=30.1, source="Menagerie unitree_go2 (torque limits); PD gains as Isaac Lab")
KNEE = ActuatorModel(kp=25.0, kv=0.5, forcerange=45.43, damping=2.0, frictionloss=0.2, armature=0.01,
                     max_velocity=20.1, source="Menagerie unitree_go2 (torque limits); PD gains as Isaac Lab")
STAND = {"FL_thigh_joint": 0.9, "FL_calf_joint": -1.8, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8}


@design(
    params=dict(
        hip_xy=param((193.4, 46.5), doc="abduction axis position (x fore/aft, y from the centre line)"),
        thigh_out=param(95.5, (80, 110), "abduction axis to the thigh: lateral offset"),
        thigh_len=param(213.0, (180, 260), "thigh joint to knee"),
        calf_len=param(213.0, (180, 260), "knee to foot centre"),
        torso_g=param(4800.0, (3000, 7000), "frame, battery and computer"),
        motor_g=param(530.0, (400, 700), "one GO-M8010-6 joint motor"),
    ),
    home=symmetric(STAND, "FL_", "FR_", axes="pseudo") | symmetric(STAND, "RL_", "RR_", axes="pseudo"),
)
def build(p):
    hx, hy = p.hip_xy
    torso = box((-188, -47, -57), (188, 47, 57)) + cyl((285, 0, 10), (0, 0, 1), 50, 90) + cyl((293, 0, -60), X, 47, 60)
    base = Component.world("base", torso, BOUGHT, BLACK, collision=True, mass_override=p.torso_g / 1000)
    asm = Assembly("go2", root=base, floating=True, root_link="base")
    base.sites["imu"] = (-25.6, 0.0, 42.3)
    for leg, x, rng in (("FL", hx, (-1.5708, 3.4907)), ("RL", -hx, (-0.5236, 4.5379))):
        # abduction motor in the torso, axis along x
        asm.attach(base, Component.world(f"{leg}_abduction_motor", cyl((x, hy, 0), X, 46, 40), BOUGHT, GRAY,
                                         mass_override=p.motor_g / 1000))
        hip = Component.world(f"{leg}_hip_part", cyl((x, hy + 20, 0), Y, 48, 76) + box((x - 20, hy - 20, -20),
                              (x + 20, hy + 20, 20)), BOUGHT, METAL, mass_override=0.678)
        asm.hinge(f"{leg}_hip_joint", base, hip, (x, hy, 0), X, (-1.0472, 1.0472), actuator=HIP,
                  child_link=f"{leg}_hip")
        ty = hy + p.thigh_out
        thigh = cyl((x, ty - 20, 0), Y, 48, 74) + box((x - 17, ty - 12, -p.thigh_len), (x + 17, ty + 12, 0))
        thigh = Component.world(f"{leg}_thigh_part", thigh, ALUMINIUM, METAL, collision=True)
        asm.hinge(f"{leg}_thigh_joint", hip, thigh, (x, ty, 0), Y, rng, actuator=HIP, child_link=f"{leg}_thigh")
        asm.attach(thigh, Component.world(f"{leg}_knee_motor", cyl((x, ty - 35, 0), Y, 40, 30), BOUGHT, GRAY,
                                          mass_override=p.motor_g / 1000))
        kz = -p.thigh_len
        calf = cyl((x, ty, kz - p.calf_len / 2), (0, 0, 1), 12, p.calf_len) + cyl((x, ty, kz), Y, 15, 24)
        calf = Component.world(f"{leg}_calf_part", calf, ALUMINIUM, GRAY)
        asm.hinge(f"{leg}_calf_joint", thigh, calf, (x, ty, kz), Y, (-2.7227, -0.83776), actuator=KNEE,
                  child_link=f"{leg}_calf")
        foot = Component.world(f"{leg}_foot", cyl((x, ty, kz - p.calf_len), Y, 22, 30), BOUGHT, BLACK,
                               collision=True, mass_override=0.03)
        asm.attach(calf, foot).sites[leg] = (x, ty, kz - p.calf_len)
    asm.imu("imu")
    robot = asm.robot()
    return robot.mirror("FL_", "FR_", axes="pseudo").mirror("RL_", "RR_", axes="pseudo")
