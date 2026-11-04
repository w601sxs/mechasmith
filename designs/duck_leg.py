"""Day 1: one duck-class leg (5 DOF) on a fixed test trunk.

Kinematics follow the common small-biped template:
hip yaw -> hip roll -> hip pitch -> knee -> ankle.
World frame: x forward, y left, z up, mm, home pose = leg straight.
"""
from __future__ import annotations

import numpy as np
from build123d import Box, Pos

from mechasmith.brackets import horn_plate, u_bracket
from mechasmith.components.servo import XL330, Servo
from mechasmith.core import Link, Part, Robot
from mechasmith.materials import PLA, TPU

# ---- design parameters (the knobs an agent iterates on) ----------------------
P = dict(
    hip_y=17.5,          # lateral offset of the hip yaw axis from the centre line
    thigh_reach=27.5,    # hip-pitch U-bracket bridge distance (sets thigh length)
    shin_reach=27.5,     # knee U-bracket bridge distance (sets shin length)
    roll_reach=16.0,     # hip-roll U-bracket bridge; 16 gives full ±0.38 rad (grid-verified)
    roll_body="out",     # hip-roll servo case points outward: compact hip, no clash with the other leg
    foot_reach=14.5,     # ankle U-bracket bridge; >= ~14 needed for full ankle range (sweep-verified)
    foot_size=(72.0, 36.0, 3.0),
    foot_forward=8.0,    # foot centre ahead of the ankle axis
)

UP, DOWN = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])
FWD, BACK = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
LEFT = np.array([0, 1.0, 0])

SHELL = (0.93, 0.91, 0.86, 1.0)   # cream
ACCENT = (0.96, 0.55, 0.16, 1.0)  # orange


def build(side: str = "left", p: dict = P) -> Robot:
    s = 1 if side == "left" else -1
    out = LEFT * s
    pre = f"{side}_"

    trunk = Link("trunk")
    robot = Robot("duck_leg_rig", trunk, floating=False)

    # Hip yaw: axis vertical, horn facing down, case lying backwards inside the trunk.
    yaw = Servo(pre + "hip_yaw", XL330, anchor=(0, s * p["hip_y"], 0), axis=DOWN, body_dir=BACK)
    trunk.add(yaw.case_part(),
              Part("trunk_block", Pos(-10, 0, 20) * Box(70, 70, 16), PLA, SHELL))

    # Yaw link: horn plate + the hip-roll servo hanging under it (axis forward).
    yaw_link = Link(pre + "hip_yaw_link")
    plate, face = horn_plate(yaw, size=(24, 34), offset=(0, 0))
    roll_body = {"down": DOWN, "out": out, "in": -out}[p.get("roll_body", "down")]
    roll = Servo.mount_on(pre + "hip_roll", XL330, face, axis=FWD, body_dir=roll_body)
    yaw_link.add(yaw.horn_part(), plate, roll.case_part())
    robot.joint(pre + "hip_yaw", trunk, yaw_link, yaw.anchor, yaw.axis,
                (-0.436, 0.524) if s > 0 else (-0.524, 0.436), XL330.actuator)

    # Roll link: U-bracket around the roll servo, hip-pitch servo below (axis outward).
    roll_link = Link(pre + "hip_roll_link")
    ub, face = u_bracket(roll, DOWN, p["roll_reach"], name=pre + "hip_roll_bracket")
    pitch = Servo.mount_on(pre + "hip_pitch", XL330, face, axis=out, body_dir=DOWN)
    roll_link.add(roll.horn_part(), ub, pitch.case_part())
    robot.joint(pre + "hip_roll", yaw_link, roll_link, roll.anchor, roll.axis, (-0.384, 0.384), XL330.actuator)

    # Thigh: U-bracket around the hip-pitch servo, knee servo below.
    thigh = Link(pre + "hip_pitch_link")
    ub, face = u_bracket(pitch, DOWN, p["thigh_reach"], name=pre + "thigh_bracket", color=SHELL)
    knee = Servo.mount_on(pre + "knee", XL330, face, axis=out, body_dir=DOWN)
    thigh.add(pitch.horn_part(), ub, knee.case_part())
    robot.joint(pre + "hip_pitch", roll_link, thigh, pitch.anchor, pitch.axis, (-1.571, 1.571), XL330.actuator)

    # Shin: U-bracket around the knee servo, ankle servo below (case pointing up into the shin).
    shin = Link(pre + "knee_link")
    ub, face = u_bracket(knee, DOWN, p["shin_reach"], name=pre + "shin_bracket", color=SHELL)
    ankle = Servo.mount_on(pre + "ankle", XL330, face, axis=out, body_dir=UP)
    shin.add(knee.horn_part(), ub, ankle.case_part())
    robot.joint(pre + "knee", thigh, shin, knee.anchor, knee.axis, (-1.571, 1.571), XL330.actuator)

    # Foot: U-bracket around the ankle servo + sole plate (the only ground-contact geometry).
    foot = Link(pre + "ankle_link")
    ub, face = u_bracket(ankle, DOWN, p["foot_reach"], name=pre + "ankle_bracket", color=ACCENT)
    fl, fw, ft = p["foot_size"]
    sole_c = face.point + DOWN * (ft / 2) + FWD * p["foot_forward"]
    sole = Part(pre + "sole", Pos(*sole_c) * Box(fl, fw, ft), TPU, ACCENT, collision=True)
    foot.add(ankle.horn_part(), ub, sole)
    foot.sites[pre + "foot"] = tuple(face.point + DOWN * ft)
    robot.joint(pre + "ankle", shin, foot, ankle.anchor, ankle.axis, (-1.571, 1.571), XL330.actuator)

    return robot


HOME: dict[str, float] = {}
POSES = {
    "crouch": {"left_hip_pitch": -0.6, "left_knee": 1.2, "left_ankle": -0.6},
    "kick": {"left_hip_yaw": 0.3, "left_hip_roll": 0.3, "left_hip_pitch": -1.0, "left_knee": 0.4},
}
