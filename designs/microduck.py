"""Microduck, reverse-engineered: Pollen Robotics' Microduck rebuilt as mechasmith code.

Nothing from Pollen's model is imported. Dimensions were measured from the public
microduck_rl MJCF (part bounding boxes and section profiles, see benchmarks/) and
every part is rebuilt here as parametric CAD: the loaf-shaped head with its face
plate and lens ring, the compact trunk with the upright battery behind the hips,
the teardrop thigh covers and the orange feet.

Frame: mm, x forward, y left, z up; origin at the left/right hip-yaw servo centres.
Joint names, axis signs and HOME match microduck_rl.
"""
from __future__ import annotations

import numpy as np
from build123d import Box, Cylinder, Plane, Pos, RectangleRounded, Rot, SlotOverall, extrude

from mechasmith.brackets import horn_plate, l_bracket, u_bracket
from mechasmith.components.servo import XL330, MountFace, Servo
from mechasmith.core import Link, Part, Robot
from mechasmith.materials import BOUGHT, PLA, TPU
from mechasmith.shells import arch, loft_shell, profile_plate, rounded_box_shell

P = dict(
    # legs (Microduck values, measured)
    hip_y=17.5,
    roll_gap=20.5,       # roll axis -> roll-link side plate; puts the hip-pitch servo centre 55 mm out
    pitch_x=-2.0,
    knee_gap=1.0,
    knee_drop=22.0,
    shin_len=42.0,
    ankle_in=5.0,        # ankle sits 5 mm inboard of the knee
    foot_clear=14.0,
    foot_size=(54.0, 40.0),
    foot_forward=6.8,
    sole_bottom=25.0,    # ankle axis -> bottom of the sole
    # trunk
    trunk_x=(-52.0, 29.0),
    trunk_half_w=32.0,
    trunk_z=(-8.5, 33.5),
    trunk_wall=1.6,
    battery_g=100.0,     # upright 2S pack behind the hips; 100 g holds HOME best on servos alone (4 deg, 22 mm margin)
    # neck and head
    neck_x=20.0,
    neck_z=23.4,
    neck_reach=17.0,
    head_pitch_reach=17.0,
    roll_spacer=8.0,
    head_wall=1.8,
    head_hole_r=26.0,
    electronics_g=25.0,  # Raspberry Pi Zero 2 W + servo hat, behind the face
    speaker_g=8.0,
    camera_g=5.0,
)

# Head loaf: (x from the head-yaw axis, half width, top above the head-roll axis), mm.
HEAD_STATIONS = [(-58, 22, 14), (-56, 29, 23), (-46, 35, 30), (-36, 38, 33), (-16, 40.6, 35.6),
                 (4, 42.3, 37.3), (24, 43, 39.5), (44, 44, 40.5), (64, 44, 41)]
HEAD_BOTTOM = -17.0   # below the head-roll axis
HEAD_SPLIT = -3.0     # white top shell above, orange bottom shell below

UP, DOWN = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])
FWD, BACK = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
LEFT = np.array([0, 1.0, 0])

# Colours measured from the reference model's materials.
SHELL = (0.85, 0.85, 0.84, 1.0)
SHELL_L = (0.80, 0.80, 0.80, 1.0)
TAUPE = (0.67, 0.64, 0.60, 1.0)
ORANGE = (1.0, 0.40, 0.12, 1.0)
YELLOW = (0.96, 0.69, 0.14, 1.0)
DARK = (0.13, 0.15, 0.13, 1.0)
SLATE = (0.35, 0.38, 0.40, 1.0)
LENS = (0.26, 0.28, 0.30, 1.0)
BATTERY = (0.30, 0.30, 0.30, 1.0)
PCB = (0.31, 0.49, 0.41, 1.0)
INK = (0.08, 0.08, 0.09, 1.0)

T = 2.5


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Pos(*((lo + hi) / 2)) * Box(*(hi - lo))


def add_leg(robot: Robot, trunk: Link, side: str, p: dict) -> None:
    s = 1 if side == "left" else -1
    out = LEFT * s
    pre = f"{side}_"
    hf = XL330.horn_face

    yaw = Servo(pre + "hip_yaw", XL330, anchor=(0, s * p["hip_y"], 0), axis=DOWN, body_dir=BACK)
    trunk.add(yaw.case_part())

    yaw_link = Link(pre + "hip_yaw_link")
    plate, face = horn_plate(yaw, size=(20, 26), name=pre + "hip_yaw_plate", color=DARK)
    roll = Servo.mount_on(pre + "hip_roll", XL330, face, axis=FWD, body_dir=DOWN)
    yaw_link.add(yaw.horn_part(), plate, roll.case_part())
    robot.joint(pre + "hip_yaw", trunk, yaw_link, yaw.anchor, DOWN,
                (-0.436, 0.524) if s > 0 else (-0.524, 0.436), XL330.actuator)

    y_horn = p["hip_y"] + p["roll_gap"] + T
    pitch = Servo(pre + "hip_pitch", XL330, anchor=(p["pitch_x"], s * (y_horn + hf), roll.anchor[2]),
                  axis=-out, body_dir=BACK)
    roll_link = Link(pre + "hip_roll_link")
    lb = l_bracket(roll.horn_face(), pitch.horn_face(), name=pre + "hip_roll_bracket", color=TAUPE)
    roll_link.add(roll.horn_part(), lb, pitch.horn_part())
    robot.joint(pre + "hip_roll", yaw_link, roll_link, roll.anchor, FWD, (-0.384, 0.384), XL330.actuator)

    # Thigh: hip-pitch and knee cases under a teardrop cover on the outside.
    knee_x = p["pitch_x"] - XL330.case_below - p["knee_gap"] - XL330.case_width / 2
    knee = Servo(pre + "knee", XL330, anchor=(knee_x, pitch.anchor[1], pitch.anchor[2] - p["knee_drop"]),
                 axis=-out, body_dir=UP)
    ya = pitch.anchor[1] + s * hf  # outer face of the back bosses
    y0, y1 = sorted((ya, ya + s * T))
    tear = [(pitch.anchor[0], pitch.anchor[2]), (knee_x - 1.5, pitch.anchor[2] + 2.5), (knee_x, knee.anchor[2])]
    cover = profile_plate(tear, y0, y1, round_r=12.0)
    thigh = Link(pre + "hip_pitch_link")
    thigh.add(pitch.case_part(), knee.case_part(), Part(pre + "thigh_cover", cover, PLA, SHELL))
    for nm, sv in (("pitch", pitch), ("knee", knee)):
        yb0, yb1 = sorted((y1 if s > 0 else y0, (y1 if s > 0 else y0) + s * 1.0))
        dot = Pos(sv.anchor[0], (yb0 + yb1) / 2, sv.anchor[2]) * Rot(90, 0, 0) * Cylinder(4.5, yb1 - yb0)
        thigh.add(Part(f"{pre}{nm}_bearing", dot, BOUGHT, DARK, mass_override=0.002))
    robot.joint(pre + "hip_pitch", roll_link, thigh, pitch.anchor, out, (-1.571, 1.571), XL330.actuator)

    # Shin: thick plate on the knee horn reaching the ankle case (ankle 5 mm inboard).
    ankle = Servo(pre + "ankle", XL330,
                  anchor=(knee_x, knee.anchor[1] - s * p["ankle_in"], knee.anchor[2] - p["shin_len"]),
                  axis=out, body_dir=UP)
    yk = knee.horn_face().point[1]
    yb = ankle.boss_face().point[1]
    ys0, ys1 = sorted((yk, yb - s * T))
    shin_plate = box((knee_x - 9.5, ys0, ankle.anchor[2] - 9.5), (knee_x + 9.5, ys1, knee.anchor[2] + 9.5))
    shin = Link(pre + "knee_link")
    shin.add(knee.horn_part(), Part(pre + "shin_plate", shin_plate, PLA, TAUPE), ankle.case_part())
    robot.joint(pre + "knee", thigh, shin, knee.anchor, -out, (-1.571, 1.571), XL330.actuator)

    # Foot: orange bracket from the ankle horn, orange foot, yellow TPU sole.
    za, xa, yc = ankle.anchor[2], ankle.anchor[0], ankle.anchor[1]
    top = ankle.anchor + DOWN * p["foot_clear"]
    fb = l_bracket(ankle.horn_face(), MountFace(top, DOWN), width=24, name=pre + "foot_bracket", color=ORANGE)
    fl, fw = p["foot_size"]
    fx = xa + p["foot_forward"]
    z_sole = za - p["sole_bottom"]
    slab = lambda z0, z1, r: Pos(fx, yc, z0) * extrude(RectangleRounded(fl, fw, r), amount=z1 - z0)
    foot = Link(pre + "ankle_link")
    # chunky cup: thick base plus raised rims along both long sides (clear of the ankle case)
    rim = lambda sy: Pos(fx, yc + sy * (fw / 2 - 1.2), top[2]) * extrude(RectangleRounded(fl - 6, 2.4, 1.1),
                                                                          amount=8.0)
    cup = slab(z_sole + 3.5, top[2], 8.0) + rim(1) + rim(-1)
    foot.add(ankle.horn_part(), fb,
             Part(pre + "foot", cup, PLA, ORANGE),
             Part(pre + "sole", slab(z_sole, z_sole + 3.5, 8.0), TPU, YELLOW, collision=True))
    foot.sites[pre + "foot"] = (xa, yc, z_sole)
    robot.joint(pre + "ankle", shin, foot, ankle.anchor, out, (-1.571, 1.571), XL330.actuator)


def add_trunk_parts(trunk: Link, p: dict) -> None:
    (x0, x1), hw, (z0, z1) = p["trunk_x"], p["trunk_half_w"], p["trunk_z"]
    w = p["trunk_wall"]
    shell = rounded_box_shell((x0, -hw, z0), (x1, hw, z1), radius=12, wall=w)
    for s in (1, -1):  # hip-yaw servos pass through the floor
        shell = shell - box((-26, min(s * 6, s * 29), z0 - 1), (11, max(s * 6, s * 29), z0 + w + 1))
    # neck notch through the top and the front wall (the neck servo drops into the chest)
    # reaches back far enough for the neck to tilt back its full 60 degrees
    shell = shell - box((p["neck_x"] - 28, -19, p["neck_z"] - 12), (x1 + 1, 19, z1 + 1))
    shell = shell - box((x0 - 1, -20, z0 - 1), (-30, 20, z0 + w + 1))                          # battery slot
    left = shell & box((x0 - 1, 0, z0 - 1), (x1 + 1, hw + 1, z1 + 1))
    right = shell & box((x0 - 1, -hw - 1, z0 - 1), (x1 + 1, 0, z1 + 1))
    trunk.add(Part("trunk_shell_left", left, PLA, SHELL_L, collision=True),
              Part("trunk_shell_right", right, PLA, SHELL, collision=True))
    trunk.add(Part("battery", box((-50, -19, -48.5), (-31, 19, 22.5)), BOUGHT, BATTERY,
                   mass_override=p["battery_g"] / 1000),
              Part("power_support", box((-31, -27, -50), (-28.5, 27, 30)), PLA, TAUPE))
    trunk.sites["imu"] = (-10.0, 0.0, 20.0)


def add_neck_and_head(robot: Robot, trunk: Link, p: dict) -> None:
    hf = XL330.horn_face
    nx = p["neck_x"]
    np_servo = Servo("neck_pitch", XL330, anchor=(nx, 0, p["neck_z"]), axis=LEFT, body_dir=UP)
    ub, _ = u_bracket(np_servo, DOWN, p["neck_reach"], name="neck_mount", color=DARK)
    trunk.add(np_servo.horn_part(), ub)

    hp_z = p["neck_z"] + 2 * XL330.case_below + 1.0
    hp_servo = Servo("head_pitch", XL330, anchor=(nx, 0, hp_z), axis=LEFT, body_dir=DOWN)
    neck = Link("neck_link")
    zj = p["neck_z"] + XL330.case_below + 0.5   # joint line between the two cases
    neck.add(np_servo.case_part(), hp_servo.case_part(),
             Part("neck_plate_left", box((nx - 10, 11.5, zj - 5.5), (nx + 10, 14.0, zj + 5.5)), PLA, TAUPE),
             Part("neck_plate_right", box((nx - 10, -14.0, zj - 5.5), (nx + 10, -11.5, zj + 5.5)), PLA, TAUPE))
    robot.joint("neck_pitch", trunk, neck, np_servo.anchor, -LEFT, (-1.571, 1.047), XL330.actuator)

    ub, face = u_bracket(hp_servo, UP, p["head_pitch_reach"], name="head_pitch_bracket", color=SLATE)
    yaw = Servo.horn_on("head_yaw", XL330, face, body_dir=-LEFT)
    hp_link = Link("head_pitch_link")
    hp_link.add(hp_servo.horn_part(), ub, yaw.horn_part())
    robot.joint("head_pitch", neck, hp_link, hp_servo.anchor, LEFT, (-1.571, 1.571), XL330.actuator)

    from mechasmith.brackets import spacer
    roll_anchor = yaw.case_face(BACK).point + BACK * (p["roll_spacer"] + hf)
    roll_anchor[1] = 0.0
    roll = Servo("head_roll", XL330, anchor=roll_anchor, axis=FWD, body_dir=-LEFT)
    yaw_link = Link("head_yaw_link")
    yaw_link.add(yaw.case_part(), spacer(roll.horn_face(), yaw.case_face(BACK), name="head_roll_spacer", color=DARK),
                 roll.horn_part())
    robot.joint("head_yaw", hp_link, yaw_link, yaw.anchor, UP, (-2.967, 2.967), XL330.actuator)

    head = Link("head")
    head.add(roll.case_part())
    for part in head_parts(p, origin=(nx, 0.0, roll.anchor[2])):
        head.add(part)
    head.sites["head_camera"] = (nx + 64, 0.0, roll.anchor[2] + 16)
    robot.joint("head_roll", yaw_link, head, roll.anchor, BACK, (-0.436, 0.436), XL330.actuator)


def head_parts(p: dict, origin) -> list[Part]:
    """The Microduck head, built in a frame at the head-yaw axis / head-roll height."""
    O = Pos(*origin)
    st, wall = HEAD_STATIONS, p["head_wall"]
    shell = loft_shell(st, HEAD_BOTTOM, wall, radius=26.0, bottom_radius=6.0)
    shell = shell - Pos(0, 0, HEAD_BOTTOM - 40) * Cylinder(p["head_hole_r"], 80)           # neck opening
    fx = st[-1][0]
    face_c, face_w, face_h = (HEAD_SPLIT + 38) / 2, 82.0, 38 - HEAD_SPLIT
    recess = extrude(Plane.YZ.offset(fx - 10) * (Pos(0, face_c) * arch(face_w, face_h, 22, 3)), amount=20)
    shell = shell - recess
    top = shell & box((-80, -60, HEAD_SPLIT), (80, 60, 60))
    bottom = shell & box((-80, -60, HEAD_BOTTOM - 1), (80, 60, HEAD_SPLIT))
    face = extrude(Plane.YZ.offset(fx - 11) * (Pos(0, face_c) * arch(face_w + 2, face_h, 23, 3)), amount=1.5)
    # lens ring, lens and the pill-shaped window
    ring = (Pos(fx - 9.5, 0, 16) * Rot(0, 90, 0) * Cylinder(15, 8)) - (Pos(fx - 9.5, 0, 16) * Rot(0, 90, 0) * Cylinder(8.5, 10))
    lens = Pos(fx - 11, 0, 16) * Rot(0, 90, 0) * Cylinder(8.5, 6)
    pill = extrude(Plane.YZ.offset(fx - 9.6) * (Pos(24, 16) * SlotOverall(12, 6)), amount=1.2)
    parts = [
        Part("head_top_shell", O * top, PLA, SHELL, collision=True),
        Part("head_bottom_shell", O * bottom, PLA, ORANGE, collision=True),
        Part("face_plate", O * face, PLA, TAUPE),
        Part("lens_ring", O * ring, PLA, YELLOW),
        Part("lens", O * lens, BOUGHT, LENS, mass_override=p["camera_g"] / 1000),
        Part("face_window", O * pill, PLA, INK),
        Part("head_electronics", O * box((33, -32, 1), (43, 32, 31)), BOUGHT, PCB,
             mass_override=p["electronics_g"] / 1000),
        Part("speaker", O * box((-44, -17, 10), (-19, 17, 17)), BOUGHT, (0.29, 0.29, 0.29, 1),
             mass_override=p["speaker_g"] / 1000),
    ]
    # jaw hinge lobes on both sides, with the hinge horn
    for s, nm in ((1, "left"), (-1, "right")):
        hw = 42.6
        y0, y1 = sorted((s * hw, s * (hw + 2.5)))
        lobe = profile_plate([(-12, HEAD_SPLIT), (14, HEAD_SPLIT), (-1, 12)], y0, y1, round_r=4)
        horn = Pos(-1, s * (hw + 3.0), 8) * Rot(90, 0, 0) * Cylinder(5, 1.2)
        parts += [Part(f"jaw_lobe_{nm}", O * lobe, PLA, ORANGE),
                  Part(f"jaw_horn_{nm}", O * horn, PLA, DARK)]
    # Microduck's jaw servo (fixed, no joint in microduck_rl) is left out: at its reference
    # position it sits 1.5 mm from the head-yaw case and blocks head roll in the sweep.
    return parts


def build(p: dict = P) -> Robot:
    trunk = Link("trunk")
    robot = Robot("microduck", trunk, floating=True)
    add_trunk_parts(trunk, p)
    add_leg(robot, trunk, "left", p)
    add_leg(robot, trunk, "right", p)
    add_neck_and_head(robot, trunk, p)
    return robot


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
