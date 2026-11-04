"""A 14-DOF duck-class biped (XL330 servos), designed from scratch in code.

Topology learned from Pollen's Microduck (the servo layout, not their parts):
  * hip: yaw servo in the trunk, roll servo hanging under it, and the hip-pitch
    servo BESIDE the roll servo at the same height (not stacked below it);
  * "inverted" mounting: the hip-pitch and knee cases ride on the thigh, the
    ankle case on the shin, so servos cluster and links stay short;
  * neck: two servos back to back (neck pitch below, head pitch above);
  * head: yaw servo, and the roll servo directly behind it at the same height.
Every bracket is generated from the servos' own frames, and every number below
is a knob the sweep tools can move.

Frame: mm, x forward, y left, z up, origin midway between the hip-yaw axes.
Joint names and axis signs follow microduck_rl so policies/configs can map over.
"""
from __future__ import annotations

import numpy as np
from build123d import Box, Cylinder, Pos

from mechasmith.brackets import cover_plate, horn_plate, l_bracket, spacer, strap, u_bracket
from mechasmith.components.servo import XL330, MountFace, Servo
from mechasmith.core import Link, Part, Robot
from mechasmith.materials import BOUGHT, PLA, TPU
from mechasmith.shells import ellipsoid, ellipsoid_shell, rounded_box_shell

P = dict(
    # legs
    hip_y=17.5,          # hip-yaw axis offset from the centre line
    roll_gap=15.0,       # roll axis -> roll-link side plate (sets roll clearance and stance width)
    pitch_x=-2.0,        # hip-pitch axis, fore/aft
    knee_gap=1.0,        # gap between the hip-pitch and knee cases on the thigh
    knee_drop=22.0,      # knee axis below the hip-pitch axis
    shin_len=42.0,       # knee axis -> ankle axis (knee case points up, so >= ~39 frees the knee)
    foot_clear=14.0,     # ankle axis -> top of the foot plate (>=13.8 frees the full ankle range)
    foot_size=(70.0, 34.0, 3.0),
    foot_forward=8.0,
    # trunk
    trunk_x=(-80.0, 54.0),
    trunk_half_w=35.0,
    trunk_z=(-8.0, 21.0),
    trunk_wall=1.6,
    battery_g=50.0,      # 2S LiPo, lying across the chest in front of the hips
    battery_x=32.0,      # the balance knob: puts the CoM over the ankles at HOME
    electronics_g=25.0,  # SBC + servo bus board + IMU
    # neck and head
    neck_x=20.0,
    neck_z=40.0,
    neck_reach=17.0,     # trunk U-bracket under the neck servo (> neck-plate corner radius)
    head_pitch_reach=17.0,
    roll_spacer=8.0,     # head-yaw case -> head-roll horn
    head_radii=(48.0, 38.0, 36.0),
    head_wall=1.6,
    head_hole_r=26.0,    # opening around the head-yaw axis
    head_cut=14.0,       # flat underside, below the head-roll axis
    head_payload_g=20.0, # camera + speaker
)

UP, DOWN = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])
FWD, BACK = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
LEFT = np.array([0, 1.0, 0])

CREAM = (0.95, 0.93, 0.86, 1.0)
WHITE = (0.97, 0.97, 0.96, 1.0)
ORANGE = (0.96, 0.55, 0.16, 1.0)
GREY = (0.55, 0.57, 0.6, 1.0)
INK = (0.08, 0.08, 0.09, 1.0)

T = 2.5  # bracket plate thickness


def add_leg(robot: Robot, trunk: Link, side: str, p: dict) -> None:
    s = 1 if side == "left" else -1
    out = LEFT * s
    pre = f"{side}_"
    hf = XL330.horn_face

    # Hip yaw: case in the trunk, horn facing down.
    yaw = Servo(pre + "hip_yaw", XL330, anchor=(0, s * p["hip_y"], 0), axis=DOWN, body_dir=BACK)
    trunk.add(yaw.case_part())

    # Yaw link: horn plate + roll servo hanging under it (axis forward).
    yaw_link = Link(pre + "hip_yaw_link")
    plate, face = horn_plate(yaw, size=(20, 26), name=pre + "hip_yaw_plate", color=GREY)
    roll = Servo.mount_on(pre + "hip_roll", XL330, face, axis=FWD, body_dir=DOWN)
    yaw_link.add(yaw.horn_part(), plate, roll.case_part())
    robot.joint(pre + "hip_yaw", trunk, yaw_link, yaw.anchor, DOWN,
                (-0.436, 0.524) if s > 0 else (-0.524, 0.436), XL330.actuator)

    # Hip-pitch servo beside the roll servo, horn facing inward, case lying backwards.
    y_horn = p["hip_y"] + p["roll_gap"] + T
    pitch = Servo(pre + "hip_pitch", XL330, anchor=(p["pitch_x"], s * (y_horn + hf), roll.anchor[2]),
                  axis=-out, body_dir=BACK)

    # Roll link: L-bracket from the roll horn (front) to the pitch horn (inner side).
    roll_link = Link(pre + "hip_roll_link")
    lb = l_bracket(roll.horn_face(), pitch.horn_face(), name=pre + "hip_roll_bracket", color=GREY)
    roll_link.add(roll.horn_part(), lb, pitch.horn_part())
    robot.joint(pre + "hip_roll", yaw_link, roll_link, roll.anchor, FWD, (-0.384, 0.384), XL330.actuator)

    # Thigh: hip-pitch case + knee case (behind and below), tied by an outer plate.
    knee_x = p["pitch_x"] - XL330.case_below - p["knee_gap"] - XL330.case_width / 2
    knee = Servo(pre + "knee", XL330, anchor=(knee_x, pitch.anchor[1], pitch.anchor[2] - p["knee_drop"]),
                 axis=-out, body_dir=UP)
    thigh = Link(pre + "hip_pitch_link")
    pc, kc = pitch.case_part(), knee.case_part()
    thigh.add(pc, kc, cover_plate([pc, kc], out, name=pre + "thigh_plate", color=CREAM))
    robot.joint(pre + "hip_pitch", roll_link, thigh, pitch.anchor, out, (-1.571, 1.571), XL330.actuator)

    # Shin: plate from the knee horn down to the ankle case (ankle horn faces out).
    ankle = Servo(pre + "ankle", XL330, anchor=(knee_x, knee.anchor[1], knee.anchor[2] - p["shin_len"]),
                  axis=out, body_dir=UP)
    shin = Link(pre + "knee_link")
    shin.add(knee.horn_part(), strap(knee.horn_face(), ankle.boss_face(), name=pre + "shin_plate", color=CREAM),
             ankle.case_part())
    robot.joint(pre + "knee", thigh, shin, knee.anchor, -out, (-1.571, 1.571), XL330.actuator)

    # Foot: L-bracket from the ankle horn to a plate under the ankle, plus a TPU sole.
    foot = Link(pre + "ankle_link")
    top = ankle.anchor + DOWN * p["foot_clear"]
    fb = l_bracket(ankle.horn_face(), MountFace(top, DOWN), name=pre + "foot_bracket", color=GREY)
    fl, fw, ft = p["foot_size"]
    sole_c = top + DOWN * (T + ft / 2) + FWD * p["foot_forward"]
    sole = Part(pre + "sole", Pos(*sole_c) * Box(fl, fw, ft), TPU, ORANGE, collision=True)
    foot.add(ankle.horn_part(), fb, sole)
    foot.sites[pre + "foot"] = tuple(top + DOWN * (T + ft))
    robot.joint(pre + "ankle", shin, foot, ankle.anchor, out, (-1.571, 1.571), XL330.actuator)


def add_trunk_parts(trunk: Link, p: dict) -> None:
    (x0, x1), hw, (z0, z1) = p["trunk_x"], p["trunk_half_w"], p["trunk_z"]
    shell = rounded_box_shell((x0, -hw, z0), (x1, hw, z1), radius=10, wall=p["trunk_wall"])
    for s in (1, -1):  # openings for the hip-yaw servos to pass through the floor
        lo = np.array([-26.0, min(s * 6.0, s * 29.0), z0 - 1])
        hi = np.array([11.0, max(s * 6.0, s * 29.0), z0 + p["trunk_wall"] + 1])
        shell = shell - Pos(*((lo + hi) / 2)) * Box(*(hi - lo))
    trunk.add(Part("trunk_shell", shell, PLA, WHITE, collision=True))
    trunk.add(Part("battery", Pos(p["battery_x"], 0, 0) * Box(34, 60, 14), BOUGHT, (0.2, 0.3, 0.6, 1.0),
                   mass_override=p["battery_g"] / 1000))
    trunk.add(Part("electronics", Pos(-50, 0, 4) * Box(50, 30, 8), BOUGHT, (0.1, 0.45, 0.2, 1.0),
                   mass_override=p["electronics_g"] / 1000))
    trunk.sites["imu"] = (-50.0, 0.0, 4.0)


def add_neck_and_head(robot: Robot, trunk: Link, p: dict) -> None:
    hf = XL330.horn_face
    # Neck pitch: case on the neck (pointing up), U-bracket on the trunk around it.
    np_servo = Servo("neck_pitch", XL330, anchor=(p["neck_x"], 0, p["neck_z"]), axis=LEFT, body_dir=UP)
    ub, _ = u_bracket(np_servo, DOWN, p["neck_reach"], name="neck_mount", color=GREY)
    trunk.add(np_servo.horn_part(), ub)

    # Neck: neck-pitch case + head-pitch case back to back, front and back plates.
    hp_z = p["neck_z"] + XL330.case_below + 1.0 + XL330.case_below
    hp_servo = Servo("head_pitch", XL330, anchor=(p["neck_x"], 0, hp_z), axis=LEFT, body_dir=DOWN)
    neck = Link("neck_link")
    npc, hpc = np_servo.case_part(), hp_servo.case_part()
    neck.add(npc, hpc,
             cover_plate([npc, hpc], FWD, margin=0.0, name="neck_front", color=CREAM),
             cover_plate([npc, hpc], BACK, margin=0.0, name="neck_back", color=CREAM))
    robot.joint("neck_pitch", trunk, neck, np_servo.anchor, -LEFT, (-1.571, 1.047), XL330.actuator)

    # Head-pitch link: U-bracket over the head-pitch servo, carrying the head-yaw horn.
    ub, face = u_bracket(hp_servo, UP, p["head_pitch_reach"], name="head_pitch_bracket", color=GREY)
    yaw = Servo.horn_on("head_yaw", XL330, face, body_dir=-LEFT)
    hp_link = Link("head_pitch_link")
    hp_link.add(hp_servo.horn_part(), ub, yaw.horn_part())
    robot.joint("head_pitch", neck, hp_link, hp_servo.anchor, LEFT, (-1.571, 1.571), XL330.actuator)

    # Head-yaw link: yaw case + spacer to the roll horn behind it.
    roll_anchor = yaw.case_face(BACK).point + BACK * (p["roll_spacer"] + hf)
    roll_anchor[1] = 0.0
    roll = Servo("head_roll", XL330, anchor=roll_anchor, axis=FWD, body_dir=-LEFT)
    yaw_link = Link("head_yaw_link")
    yaw_link.add(yaw.case_part(), spacer(roll.horn_face(), yaw.case_face(BACK), name="head_roll_spacer",
                                         color=GREY), roll.horn_part())
    robot.joint("head_yaw", hp_link, yaw_link, yaw.anchor, UP, (-2.967, 2.967), XL330.actuator)

    # Head: roll case + shell (open underneath so the yaw bracket can turn) + beak + eyes.
    zc = roll.anchor[2]
    hx = p["neck_x"] - 5.0
    ax, ay, az = p["head_radii"]
    hc = (hx, 0.0, zc + 4.0)
    shell = ellipsoid_shell(hc, p["head_radii"], p["head_wall"])
    shell = shell - Pos(p["neck_x"], 0, zc - 12 - 50) * Cylinder(p["head_hole_r"], 100)  # yaw clearance
    shell = shell - Pos(hx, 0, zc - p["head_cut"] - 50) * Box(4 * ax, 4 * ay, 100)     # flat underside
    head = Link("head")
    head.add(roll.case_part(), Part("head_shell", shell, PLA, WHITE, collision=True),
             Part("beak", ellipsoid((hx + ax - 4, 0, zc - 4), (24, 17, 6)), PLA, ORANGE),
             Part("head_electronics", Pos(hx, 0, zc + 21) * Box(20, 20, 6), BOUGHT, (0.1, 0.45, 0.2, 1.0),
                  mass_override=p["head_payload_g"] / 1000))
    for s in (1, -1):
        eye_c = (hx + 0.5 * ax, s * 0.78 * ay, zc + 4 + 0.35 * az)
        head.add(Part(f"eye_{'left' if s > 0 else 'right'}", ellipsoid(eye_c, (6, 6, 6)), PLA, INK))
    head.sites["head_camera"] = (hx + ax, 0.0, zc + 10)
    robot.joint("head_roll", yaw_link, head, roll.anchor, BACK, (-0.436, 0.436), XL330.actuator)


def build(p: dict = P) -> Robot:
    trunk = Link("trunk")
    robot = Robot("duck", trunk, floating=True)
    add_trunk_parts(trunk, p)
    add_leg(robot, trunk, "left", p)
    add_leg(robot, trunk, "right", p)
    add_neck_and_head(robot, trunk, p)
    return robot


# Microduck's standing pose (microduck_rl HOME_FRAME); joint names and signs match.
HOME = {
    "left_hip_roll": -0.0873, "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
    "left_knee": -0.0049, "right_knee": 0.0049,
    "left_ankle": 0.4530, "right_ankle": -0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491,
}
POSES = {
    "look_left": {**HOME, "head_yaw": 0.8, "head_roll": 0.3},
    "crouch": {**HOME, "left_hip_pitch": -0.9, "right_hip_pitch": 0.9, "left_knee": -0.6, "right_knee": 0.6,
               "left_ankle": 1.0, "right_ankle": -1.0},
}
