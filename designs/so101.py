"""SO-101 arm (The Robot Studio / Hugging Face LeRobot), rebuilt as mechasmith code.

Six Feetech STS3215 servos: shoulder pan, shoulder lift, elbow, wrist flex, wrist roll
and the gripper jaw. Joint positions and part proportions were measured from the
Apache-2.0 SO-ARM100 MJCF (`Simulation/SO101/so101_old_calib.xml`); every printed part is
our own, drawn with `asm.link`. Nothing from that repository is imported.

Each printed link is the SO-101 pattern: a fork that spans the servo it turns on, cut
clear of that servo over the joint range, and a walled pocket around the servo it carries.

Frame and zero pose follow the reference "old calibration": mm, base on the floor, the
arm stretched out horizontally along -y at zero. Joint names and axis signs match.
"""
import numpy as np

from mechasmith import (BACK, DOWN, FWD, LEFT, RIGHT, STS3215, UP, Assembly, Hold, box, color, design, extrude,
                        diamonds, param, polygon, printed, servo, swept_link)

YELLOW = color("#ffd11f")
S = STS3215


@design(
    params=dict(
        pan_xy=param((20.8, -23.1), doc="shoulder-pan axis position on the base"),
        base_top=param(58.0, (46, 80), "bottom of the pan servo case"),
        lift_z=param(149.1, (125, 180), "shoulder-lift axis height"),
        lift_back=param(30.4, (25, 45), "shoulder-lift axis behind the pan axis (along -y)"),
        upper_arm=param((112.5, -28.0), doc="shoulder lift -> elbow, (along -y, up)"),
        lower_arm=param((134.9, 5.2), doc="elbow -> wrist flex, (along -y, up)"),
        wrist_roll_y=param(43.1, (38, 70), "wrist flex axis -> wrist roll servo centre, along -y"),
        gripper_y=param(20.5, (15, 40), "wrist roll horn -> gripper axis, along -y"),
        gripper_up=param(20.2, (0, 35), "gripper axis above the wrist roll axis"),
        link_width=param(24.0, (18, 34), "width of the upper and lower arm in their plane of motion"),
        wall=param(2.5, (1.5, 2.9), "wall of the pockets that hold the servos"),
        holder_band=param(28.0, (16, 35), "axial width of the ring holders round the pan, lift and wrist servos"),
    ),
    base_height=0.0,
    tool_site="gripper",
    payload_kg=0.2,
    poses={
        "rest": {"shoulder_lift": -1.75, "elbow_flex": 1.6, "wrist_flex": 0.6},
        "reach_up": {"shoulder_lift": -2.6, "elbow_flex": 0.9, "wrist_flex": -0.3, "wrist_roll": 1.0},
        "open": {"gripper": 1.2},
    },
)
def build(p):
    px, py = p.pan_xy
    # Servo anchors (case centre on the output axis) at the zero pose.
    P = np.array([px, py, p.base_top + S.case_half_axial])           # shoulder pan, horn up
    L = np.array([px, py - p.lift_back, p.lift_z])                    # shoulder lift
    E = L + (0, -p.upper_arm[0], p.upper_arm[1])                      # elbow
    W = E + (0, -p.lower_arm[0], p.lower_arm[1])                      # wrist flex
    R = np.array([px, W[1] - p.wrist_roll_y, W[2]])                   # wrist roll, horn toward -y
    G = np.array([px, R[1] - S.horn_face - p.gripper_y, R[2] + p.gripper_up])  # gripper
    pan, lift, elbow, wrist, roll, grip = (servo(n, S) for n in (
        "shoulder_pan_motor", "shoulder_lift_motor", "elbow_motor", "wrist_flex_motor", "wrist_roll_motor",
        "gripper_motor"))
    kw = dict(wall=p.wall, color=YELLOW)

    feet = [diamonds((cx, py + 23, 33.5), UP, LEFT, 5, 3, size=7, pitch=11.5, depth=12) for cx in (px - 29, px + 29)]
    # The shoulder yoke, declared once: the base is carved wherever the yoke can swing.
    yoke_path, yoke_width = [P, P + (0, -32, 0)], 48.0
    yoke_span = (-(S.back_face + 0.8 + 4.0), L[2] + S.case_above + p.wall - P[2])
    room = swept_link(P, UP, yoke_path, yoke_width, (-1.92, 1.92), yoke_span, gap=1.5)
    base = printed("base", holds=Hold(pan, P, UP, LEFT, band=p.holder_band), add=base_shapes(P, p), cut=feet,
                   clear=[room], collision=True, **kw)
    asm = Assembly("so101", root=base, floating=False, root_link="base")
    asm.fasten(base.c("holds"), pan.case.c("back"))
    asm.joint("shoulder_pan", pan, (-1.92, 1.92), reverse=True, child_link="shoulder")

    # Shoulder: a yoke around the pan servo; a block on its back carries the lift servo.
    band = p.holder_band / 2 - 1
    riser = box((px - band, L[1] - 14, P[2] + 18), (px + band, L[1] + 14, L[2] - S.case_below))
    asm.link("shoulder_yoke", around=pan, path=yoke_path, width=yoke_width,
             holds=Hold(lift, L, FWD, DOWN, band=p.holder_band), add=[riser], **kw)
    asm.joint("shoulder_lift", lift, (-3.316, 0.175), child_link="upper_arm")

    # Upper arm: a bar at shoulder height; the elbow servo hangs from its far end.
    asm.link("upper_arm_link", around=lift, path=[L, (px, E[1], L[2])], width=p.link_width,
             holds=Hold(elbow, E, BACK, UP, wrap=26), cut=[diamonds((px, E[1], L[2]), FWD, UP, 2, 4, 5, 7.5)], **kw)
    asm.joint("elbow_flex", elbow, (-0.175, 3.142), reverse=True, child_link="lower_arm")

    # Lower arm: a bar ending in the pocket of the wrist servo, which faces back along it.
    asm.link("lower_arm_link", around=elbow, path=[E, (px, W[1] + S.case_below + p.wall, W[2])],
             width=p.link_width, holds=Hold(wrist, W, FWD, LEFT, band=p.holder_band),
             cut=[diamonds((px, W[1] + 12.5, W[2]), UP, LEFT, 4, 2, 5, 9)], **kw)
    asm.joint("wrist_flex", wrist, (-1.658, 1.658), child_link="wrist")

    # Wrist: a yoke around the wrist-flex servo holding the roll servo along the arm.
    asm.link("wrist_link", around=wrist, path=[W, R], width=34, holds=Hold(roll, R, RIGHT, FWD), **kw)
    asm.joint("wrist_roll", roll, (-2.793, 2.793), reverse=True, child_link="gripper")

    # Gripper body on the roll horn: the gripper servo's pocket and the fixed finger.
    face_y = R[1] - S.horn_face
    plate = box((px - 20, face_y, G[2] - 37), (px + 20, G[1] + 12, G[2] + 12))
    body = asm.link("gripper_body", on=roll.horn, holds=Hold(grip, G, BACK, DOWN),
                    add=[plate, fixed_finger(G)], collision=True, **kw)
    body.sites["gripper"] = (px, G[1] - 74.6, G[2] - 28.0)
    asm.joint("gripper", grip, (-0.175, 1.745), child_link="moving_jaw")

    # Moving jaw: a fork around the gripper servo and a tapered finger above the fixed one.
    asm.link("moving_jaw", around=grip, path=[G, G + (0, -28, -4)], width=20, add=[moving_finger(G)],
             collision=True, **kw)
    return asm.robot()


def tapered(side, top, x_span, z_span):
    """A finger: side outline (y, z) extruded across x, intersected with the top outline
    (x, y) extruded across z."""
    return (extrude(polygon(side), "YZ", at=x_span[0], depth=x_span[1] - x_span[0])
            & extrude(polygon(top), "XY", at=z_span[0], depth=z_span[1] - z_span[0]))


def fixed_finger(G):
    x, y, z = G
    side = [(y + 14, z - 36), (y + 14, z - 54), (y - 15, z - 54), (y - 80, z - 33), (y - 82, z - 28.4),
            (y - 15, z - 28.4), (y - 15, z - 36)]
    top = [(x - 15, y + 14), (x + 15, y + 14), (x + 15, y - 15), (x + 3, y - 82), (x - 3, y - 82), (x - 15, y - 15)]
    return tapered(side, top, (x - 15, x + 15), (z - 60, z - 20))


def moving_finger(G):
    x, y, z = G
    side = [(y - 12, z + 6), (y - 12, z - 10), (y - 80, z - 11), (y - 82, z - 7), (y - 22, z + 6)]
    top = [(x - 15, y - 10), (x + 15, y - 10), (x + 15, y - 22), (x + 2, y - 83), (x - 2, y - 83), (x - 15, y - 22)]
    return tapered(side, top, (x - 15, x + 15), (z - 20, z + 10))


def base_shapes(P, p):
    """Two hexagonal feet and a block that fills everything the shoulder yoke leaves free
    (the build carves the yoke's swing out of it, so the pan pocket is held from behind)."""
    px, py, pz = P
    foot_z = (30.0, 37.0)
    feet = []
    for cx in (px - 29, px + 29):
        hexa = [(cx, py + 68), (cx + 27, py + 35), (cx + 27, py + 11), (cx, py - 22), (cx - 27, py + 11),
                (cx - 27, py + 35)]
        feet.append(extrude(polygon(hexa, round=3), "XY", at=foot_z[0], depth=foot_z[1] - foot_z[0]))
    block = box((px - 23, py - 15, foot_z[0]), (px + 23, py + 68, pz + 22))
    board = box((px - 26, py + 68, pz - 22), (px + 26, py + 76, pz + 22))   # driver-board mounting plate
    return feet + [block, board]
