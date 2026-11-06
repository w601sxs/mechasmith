"""One duck-class leg (5 DOF) on a fixed test block: the smallest complete mechasmith design.

Kinematics: hip yaw -> hip roll -> hip pitch -> knee -> ankle, leg straight at home.
Every servo hangs off the bracket before it; nothing is positioned by hand.
"""
from mechasmith import (BACK, DOWN, FWD, LEFT, PLA, TPU, XL330, Assembly, Component, box, color, design,
                        horn_plate, param, servo, u_bracket)

SHELL = color("#ede8db")
ACCENT = color("#f58c29")


@design(
    params=dict(
        hip_y=param(17.5, (12, 30), "hip-yaw axis offset from the centre line, mm"),
        thigh_reach=param(27.5, (20, 60), "hip-pitch U-bracket bridge distance (thigh length), mm"),
        shin_reach=param(27.5, (20, 60), "knee U-bracket bridge distance (shin length), mm"),
        roll_reach=param(16.0, (14, 40), "hip-roll U-bracket bridge; 16 gives the full roll range"),
        foot_reach=param(14.5, (10, 30), "ankle U-bracket bridge; >= ~14 frees the full ankle range"),
        foot_size=param((72.0, 36.0, 3.0), doc="sole length, width, thickness, mm"),
        foot_forward=param(8.0, (-20, 30), "sole centre ahead of the ankle axis, mm"),
    ),
    poses={
        "crouch": {"left_hip_pitch": -0.6, "left_knee": 1.2, "left_ankle": -0.6},
        "kick": {"left_hip_yaw": 0.3, "left_hip_roll": 0.3, "left_hip_pitch": -1.0, "left_knee": 0.4},
    },
)
def build(p):
    block = Component.world("trunk_block", box((-45, -35, 12), (25, 35, 28)), PLA, SHELL)
    block.connector("left_hip_seat", at=(0, p.hip_y, 12), normal=DOWN, x=FWD)
    asm = Assembly("duck_leg_rig", root=block, floating=False)

    yaw = servo("left_hip_yaw", XL330)
    asm.fasten(block.c("left_hip_seat"), yaw.case.c("back"), spin=(yaw.BODY, BACK))
    asm.joint("left_hip_yaw", yaw, (-0.436, 0.524))

    plate = horn_plate(XL330, "left_hip_yaw_plate", (24, 34))
    asm.fasten(yaw.horn.c("face"), plate.c("horn"))
    roll = servo("left_hip_roll", XL330)
    asm.fasten(plate.c("out"), roll.case.c("side_pos"), spin=(roll.AXIS, FWD))
    asm.joint("left_hip_roll", roll, (-0.384, 0.384))

    # Each U-bracket rides on a horn; the next servo stands on its bridge.
    rb = u_bracket(XL330, "left_hip_roll_bracket", p.roll_reach, toward="side_neg")
    asm.fasten(roll.horn.c("face"), rb.c("horn"))
    pitch = servo("left_hip_pitch", XL330)
    asm.fasten(rb.c("base"), pitch.case.c("short_end"), spin=(pitch.AXIS, LEFT))
    asm.joint("left_hip_pitch", pitch, (-1.571, 1.571))

    tb = u_bracket(XL330, "left_thigh_bracket", p.thigh_reach, toward="long_end", color=SHELL)
    asm.fasten(pitch.horn.c("face"), tb.c("horn"))
    knee = servo("left_knee", XL330)
    asm.fasten(tb.c("base"), knee.case.c("short_end"), spin=(knee.AXIS, LEFT))
    asm.joint("left_knee", knee, (-1.571, 1.571))

    sb = u_bracket(XL330, "left_shin_bracket", p.shin_reach, toward="long_end", color=SHELL)
    asm.fasten(knee.horn.c("face"), sb.c("horn"))
    ankle = servo("left_ankle", XL330)
    asm.fasten(sb.c("base"), ankle.case.c("long_end"), spin=(ankle.AXIS, LEFT))
    asm.joint("left_ankle", ankle, (-1.571, 1.571))

    ab = u_bracket(XL330, "left_ankle_bracket", p.foot_reach, toward="short_end", color=ACCENT)
    asm.fasten(ankle.horn.c("face"), ab.c("horn"))
    fl, fw, ft = p.foot_size
    x, y, z = ab.at("base") + (p.foot_forward, 0, 0)
    sole = asm.attach(ab, Component.world("left_sole", box((x - fl / 2, y - fw / 2, z - ft), (x + fl / 2, y + fw / 2, z)),
                                          TPU, ACCENT, collision=True))
    sole.sites["left_foot"] = (x - p.foot_forward, y, z - ft)
    return asm.robot()
