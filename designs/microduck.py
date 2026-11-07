"""Microduck, reverse-engineered: Pollen Robotics' Microduck rebuilt as mechasmith code.

Nothing from Pollen's model is imported. Dimensions were measured from the public
microduck_rl MJCF (part bounding boxes, section profiles, material colours).

Every part is connected, never positioned: servos and brackets are fastened face to
face, and the drawn parts (shells, covers, plates) are attached to the parts they touch,
which mechasmith checks as it goes. So nothing can float. The left leg is built once
and mirrored with microduck_rl's axis convention (right = -left for symmetric motion).

Frame: mm, x forward, y left, z up; origin between the hip-yaw servo centres.
"""
import numpy as np
from scipy.spatial import ConvexHull

from mechasmith import (BACK, BOUGHT, DOWN, FWD, LEFT, PLA, TPU, UP, XL330, Assembly, Component, MountFace,
                        arch, bearing, box, color, cyl, design, extrude, horn_plate, l_bracket, loft_cavity,
                        loft_shell, move, param, polygon, rect, rounded_box_cavity, rounded_box_shell, servo, slot,
                        spacer, symmetric, tray, u_bracket, Hold, RIGHT)

# Head loaf: (x from the head-yaw axis, half width, top above the head-roll axis), mm.
HEAD_STATIONS = [(-58, 22, 14), (-56, 29, 23), (-46, 35, 30), (-36, 38, 33), (-16, 40.6, 35.6),
                 (4, 42.3, 37.3), (24, 43, 39.5), (44, 44, 40.5), (64, 44, 41)]
HEAD_BOTTOM, HEAD_SPLIT = -17.0, 3.0   # below the roll axis; white above the split, orange below

# Colours measured from the reference model's materials.
SHELL, SHELL_L, TAUPE = color("#d9d9d6"), color("#cccccc"), color("#aba399")
ORANGE, YELLOW, DARK, SLATE = color("#ff661f"), color("#f5b024"), color("#212621"), color("#596166")
LENS, BATTERY, PCB, INK = color("#42474c"), color("#4d4d4d"), color("#4f7d69"), color("#141417")
T = 2.5   # bracket plate thickness
BIG_BEARING = (22.0, 16.0, 4.0)   # the 22 x 16 x 4 mm bearings at the hips and neck
SMALL_BEARING = (15.0, 9.0, 2.5)  # the small one in each foot

STAND = {"left_hip_roll": -0.0873, "left_hip_pitch": -0.4579, "left_knee": -0.0049, "left_ankle": 0.4530,
         "neck_pitch": 0.3491, "head_pitch": 0.3491}   # microduck_rl HOME_FRAME


@design(
    params=dict(
        hip_y=param(17.5, (14, 30), "hip-yaw axis offset from the centre line"),
        roll_gap=param(20.5, (15, 35), "roll axis -> roll-link side plate; sets stance width"),
        pitch_x=param(-2.0, (-15, 10), "hip-pitch axis, fore/aft of the roll servo"),
        knee_gap=param(1.0, (0.5, 10), "gap between the hip-pitch and knee cases on the thigh"),
        knee_drop=param(22.0, (10, 40), "knee axis below the hip-pitch axis"),
        shin_len=param(42.0, (39, 70), "knee axis -> ankle axis (>= 39 frees the knee)"),
        ankle_in=param(5.0, (0, 10), "ankle inboard of the knee"),
        tub_top=param(10.5, (10.0, 20), "ankle axis -> top of the foot tub's walls (>= 10 clears the shin)"),
        foot_size=param((54.0, 40.0), doc="foot length, width"),
        foot_forward=param(6.8, (-10, 25), "foot centre ahead of the ankle axis"),
        sole_bottom=param(25.0, (20, 40), "ankle axis -> bottom of the sole"),
        trunk_x=param((-52.0, 29.0), doc="trunk shell, back and front x"),
        trunk_half_w=param(32.0, (25, 40), "trunk shell half width"),
        trunk_z=param((-8.5, 33.5), doc="trunk shell, bottom and top z"),
        trunk_wall=param(1.6, (1.0, 3.0), "trunk shell wall"),
        battery_g=param(100.0, (20, 200), "upright battery pack behind the hips"),
        neck_x=param(20.0, (0, 30), "neck axis, fore/aft"),
        neck_z=param(23.4, (15, 40), "neck-pitch axis height; sets the neck mount's length"),
        head_pitch_reach=param(17.0, (16, 30), "U-bracket over the head-pitch servo"),
        roll_spacer=param(8.0, (2, 20), "head-yaw case -> head-roll horn"),
        head_wall=param(1.8, (1.0, 3.0), "head shell wall"),
        head_hole_r=param(29.0, (21, 32), "neck opening in the head floor, round the head-yaw axis"),
        electronics_g=param(25.0, (0, 60), "Raspberry Pi Zero 2 W + servo hat, behind the face"),
        speaker_g=param(8.0, (0, 30), "speaker at the back of the head"),
        camera_g=param(5.0, (0, 20), "camera module behind the lens"),
        infill=param(0.85, (0.1, 1.0), "PLA infill; 0.85 matches the reference's part masses (1.07 g/cm^3)"),
    ),
    home=symmetric(STAND),
    poses={
        "look_left": symmetric({**STAND, "head_yaw": 0.8, "head_roll": 0.3}),
        "crouch": symmetric({**STAND, "left_hip_pitch": -0.9, "left_knee": -0.6, "left_ankle": 1.0}),
    },
)
def build(p):
    asm = trunk(p)
    left_leg(asm, p)
    neck_and_head(asm, p)
    return asm.robot().mirror("left_", "right_").print_settings(infill=p.infill)


def trunk(p) -> Assembly:
    (x0, x1), hw, (z0, z1), w = p.trunk_x, p.trunk_half_w, p.trunk_z, p.trunk_wall
    shell = rounded_box_shell((x0, -hw, z0), (x1, hw, z1), radius=12, wall=w)
    cavity = rounded_box_cavity((x0, -hw, z0), (x1, hw, z1), radius=12, wall=w)
    for s in (1, -1):  # hip-yaw servos pass through the floor
        shell -= box((-26, s * 6, z0 - 1), (11, s * 29, z0 + w + 1))
    shell -= box((p.neck_x - 28, -19, p.neck_z - 12), (x1 + 1, 19, z1 + 1))   # neck notch, 60 deg tilt back
    shell -= box((x0 - 1, -20, z0 - 1), (-30, 20, z0 + w + 1))                # battery slot
    left = Component.world("trunk_shell_left", shell & box((x0 - 1, 0, z0 - 1), (x1 + 1, hw + 1, z1 + 1)),
                           PLA, SHELL_L, collision=True)
    left.connector("neck_seat", at=(p.neck_x, 0, z0 + w), normal=UP, x=FWD)
    left.sites["imu"] = (-10.0, 0.0, 20.0)
    asm = Assembly("microduck", root=left)
    asm.attach(left, Component.world("trunk_shell_right", shell & box((x0 - 1, -hw - 1, z0 - 1), (x1 + 1, 0, z1 + 1)),
                                     PLA, SHELL, collision=True))
    # plate across the top of the hip-yaw servos (they end at x = 9.5), filling the shell wall to wall;
    # it stops at x = 8 so the neck servo can tilt forward past it
    hip_plate = asm.attach(left, Component.world("hip_plate", box((-26, -50, 11.5), (8, 50, 14)) & cavity, PLA,
                                                 DARK))
    hip_plate.connector("left_hip_seat", at=(0, p.hip_y, 11.5), normal=DOWN, x=FWD)
    support = asm.attach(left, Component.world("power_support", box((-31, -27, -50), (-28.5, 27, z1 - w)),
                                               PLA, TAUPE))
    asm.attach(support, Component.world("battery", box((-50, -19, -48.5), (-31, 19, 22.5)), BOUGHT, BATTERY,
                                        mass_override=p.battery_g / 1000))
    return asm


def left_leg(asm: Assembly, p) -> None:
    hip_plate = asm.components["hip_plate"]
    yaw = servo("left_hip_yaw", XL330)
    asm.fasten(hip_plate.c("left_hip_seat"), yaw.case.c("back"), spin=(yaw.BODY, BACK))
    asm.joint("left_hip_yaw", yaw, (-0.436, 0.524))
    a = yaw.axis
    asm.attach(hip_plate, bearing("left_hip_yaw_bearing", yaw.anchor - a * (XL330.case_half_axial + 2.0), a,
                                  *BIG_BEARING))

    plate = horn_plate(XL330, "left_hip_yaw_plate", (20, 26), color=DARK)
    asm.fasten(yaw.horn.c("face"), plate.c("horn"))
    roll = servo("left_hip_roll", XL330)
    asm.fasten(plate.c("out"), roll.case.c("short_end"), spin=(roll.AXIS, FWD))
    asm.joint("left_hip_roll", roll, (-0.384, 0.384))

    # Hip-pitch servo beside the roll servo, horn facing in, held by an L-bracket off the roll horn.
    target = roll.anchor + (p.pitch_x, p.roll_gap + T, 0)
    lb = asm.attach(roll.horn, l_bracket("left_hip_roll_bracket", roll.face("face"), MountFace(target, -LEFT),
                                         b_x=FWD, width=22, color=TAUPE))
    a = roll.axis
    asm.attach(lb, bearing("left_hip_roll_bearing", roll.anchor + a * (XL330.case_half_axial + 1.5), a, 18.0,
                           16.0, 3.0))
    pitch = servo("left_hip_pitch", XL330)
    asm.fasten(lb.c("b"), pitch.horn.c("face"))
    asm.joint("left_hip_pitch", pitch, (-1.571, 1.571), reverse=True)

    # Thigh: a teardrop cover on the pitch servo's boss carries the knee servo's boss.
    px, py, pz = pitch.anchor
    knee_x, knee_z = px - XL330.case_below - p.knee_gap - XL330.case_width / 2, pz - p.knee_drop
    y_out = py + XL330.horn_face
    knee = servo("left_knee", XL330)
    cap_wall, cap_end = 1.6, y_out - 4.5           # a shallow skirt: the shin and ankle swing inside it
    knee_case = box((knee_x - 10, py - 1, knee_z - XL330.case_above), (knee_x + 10, py + 1, knee_z + XL330.case_below))
    outline = case_outline([pitch.case.world_shape, knee_case], pad=0.0)
    cap = tray(outline, "XZ", at=y_out + cap_wall, depth=-(y_out + cap_wall - cap_end), wall=cap_wall, round=3.0)
    seat = (knee_x, y_out - 0.4, knee_z)            # the knee bearing's ring is pressed into the cap
    cap -= cyl(seat, LEFT, BIG_BEARING[0] / 2, BIG_BEARING[2]) - cyl(seat, LEFT, BIG_BEARING[1] / 2, 9)
    cover = asm.attach(pitch.case, Component.world("left_thigh_cover", cap, PLA, SHELL))
    cover.connector("knee_seat", at=(knee_x, y_out, knee_z), normal=-LEFT, x=FWD)
    asm.fasten(cover.c("knee_seat"), knee.case.c("boss"), spin=(knee.BODY, UP))
    asm.attach(cover, bearing("left_knee_bearing", seat, LEFT, *BIG_BEARING))
    asm.joint("left_knee", knee, (-1.571, 1.571))

    # Shin: block on the knee horn, then a round-ended plate down the inside to the ankle
    # servo's boss (round, so it clears the foot tub at any ankle angle).
    kx, ky, kz = knee.anchor
    y_horn, y_boss, za = ky - XL330.horn_face, ky - p.ankle_in - XL330.horn_face, kz - p.shin_len
    shin = extrude(polygon([(kx, kz), (kx, za)], round=9.5), "XZ", at=y_boss - T, depth=T) \
        + box((kx - 9.5, y_boss, kz - 9.5), (kx + 9.5, y_horn, kz + 9.5))
    shin = asm.attach(knee.horn, Component.world("left_shin_plate", shin, PLA, TAUPE))
    shin.connector("ankle_seat", at=(kx, y_boss, za), normal=LEFT, x=FWD)
    ankle = servo("left_ankle", XL330)
    asm.fasten(shin.c("ankle_seat"), ankle.case.c("boss"), spin=(ankle.BODY, UP))
    asm.joint("left_ankle", ankle, (-1.571, 1.571))

    # Foot: an open orange tub with the fork rising out of it (one part; the ankle servo
    # swings down inside the tub, and the build carves the walls where it passes), in a
    # yellow TPU sole that wraps up its sides.
    xa, yc, za = ankle.anchor
    fl, fw = p.foot_size
    fx, z_sole = xa + p.foot_forward, za - p.sole_bottom
    core = [(fx + sx * (fl / 2 - 8), yc + sy * (fw / 2 - 8)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    tub = tray(core, "XY", at=z_sole + 2.0, depth=za - p.tub_top - z_sole - 2.0, wall=2.0, round=6.0)
    cup = asm.link("left_foot", around=ankle, path=[ankle.anchor, (xa, yc, z_sole + 12)], width=20, add=[tub],
                   cheek=2.5, gap=0.3, round=1.5, back_cheek=False, color=ORANGE)   # the shin holds the back
    sole = asm.attach(cup, Component.world("left_sole", tray(core, "XY", at=z_sole, depth=13.0, wall=2.0, round=8.0),
                                           TPU, YELLOW, collision=True))
    sole.sites["left_foot"] = (xa, yc, z_sole)


def case_outline(shapes, pad: float = 1.0):
    """Convex outline (x, z) round the side views of some servo cases, for a cap over them."""
    pts = []
    for sh in shapes:
        bb = sh.bounding_box()
        pts += [(x, z) for x in (bb.min.X - pad, bb.max.X + pad) for z in (bb.min.Z - pad, bb.max.Z + pad)]
    pts = np.array(pts)
    return [tuple(pts[i]) for i in ConvexHull(pts).vertices]


def neck_and_head(asm: Assembly, p) -> None:
    left = asm.root
    z_floor = p.trunk_z[0] + p.trunk_wall
    # Neck mount stands on the trunk floor; its length is whatever puts the neck axis at neck_z.
    mount = u_bracket(XL330, "neck_mount", reach=p.neck_z - z_floor - T, color=DARK)
    asm.fasten(left.c("neck_seat"), mount.c("base"), spin=(mount.AXIS, LEFT))
    neck = servo("neck_pitch", XL330)
    asm.fasten(mount.c("horn"), neck.horn.c("face"))
    asm.joint("neck_pitch", neck, (-1.571, 1.047), reverse=True, child_link="neck_link")

    # Second servo back to back with the first, through a 1 mm coupler; side plates for stiffness.
    coupler = spacer("neck_coupler", 1.0, width=23, height=20, color=TAUPE)
    asm.fasten(neck.case.c("long_end"), coupler.c("a"))
    head_pitch = servo("head_pitch", XL330)
    asm.fasten(coupler.c("b"), head_pitch.case.c("long_end"), spin=(head_pitch.AXIS, LEFT))
    nx, _, nz = neck.anchor
    zj = nz + XL330.case_below + 0.5
    for s, nm in ((1, "left"), (-1, "right")):
        asm.attach(neck.case, Component.world(f"neck_plate_{nm}", box((nx - 10, s * 11.5, zj - 5.5),
                                                                      (nx + 10, s * 14.0, zj + 5.5)), PLA, TAUPE))
    asm.joint("head_pitch", head_pitch, (-1.571, 1.571))

    bracket = u_bracket(XL330, "head_pitch_bracket", reach=p.head_pitch_reach, color=SLATE)
    asm.fasten(head_pitch.horn.c("face"), bracket.c("horn"))
    yaw = servo("head_yaw", XL330)
    asm.fasten(bracket.c("base"), yaw.horn.c("face"), spin=(yaw.BODY, -LEFT))
    asm.joint("head_yaw", yaw, (-2.967, 2.967), reverse=True)
    a = yaw.axis
    asm.attach(bracket, bearing("head_yaw_bearing", yaw.anchor + a * (XL330.case_half_axial + 1.5), a, 22.0, 16.0,
                                3.0))

    # Cradle round the head-yaw servo (a ring, so the head can roll round it), with a bearing on
    # the roll axis at its front; the head pivots there and on the roll servo.
    cradle = asm.link("head_yaw_cradle", on=yaw.case, holds=Hold(yaw, yaw.anchor, yaw.axis, RIGHT, band=18),
                      wall=2.0, color=SLATE)
    roll_spacer = spacer("head_roll_spacer", p.roll_spacer, color=DARK)
    asm.fasten(yaw.case.face(BACK), roll_spacer.c("a"))
    roll = servo("head_roll", XL330)
    asm.fasten(roll_spacer.c("b"), roll.horn.c("face"), spin=(roll.BODY, -LEFT))
    asm.joint("head_roll", roll, (-0.436, 0.436), reverse=True, child_link="head")
    a = roll.axis
    asm.attach(roll_spacer, bearing("head_roll_bearing", roll.anchor + a * (XL330.case_half_axial + 1.5), a, 22.0,
                                    16.0, 3.0))
    front = cradle.world_shape.bounding_box().max.X
    asm.attach(cradle, bearing("head_cradle_bearing", (front + 2.0, 0.0, roll.anchor[2]), FWD, *BIG_BEARING))
    head(asm, p, roll, origin=(yaw.anchor[0], 0.0, roll.anchor[2]))


def head(asm: Assembly, p, roll, origin) -> None:
    """The Microduck head, drawn in a frame at the head-yaw axis and head-roll height."""
    def at(shape):
        return move(shape, origin)

    ox, _, oz = origin
    fx = HEAD_STATIONS[-1][0]
    face_c, face_w, face_h = (HEAD_SPLIT + 38) / 2, 82.0, 38 - HEAD_SPLIT
    loft = dict(stations=HEAD_STATIONS, bottom=HEAD_BOTTOM, wall=p.head_wall, radius=26.0, bottom_radius=6.0,
                open_bottom=False)   # the bottom shell has a floor (with the neck opening)
    cavity = at(loft_cavity(**loft))
    shell = loft_shell(**loft) - cyl((0, 0, HEAD_BOTTOM), UP, p.head_hole_r, 80)
    shell -= extrude(arch(face_w, face_h, 22, 3), "YZ", at=fx - 10, center=(0, face_c), depth=20)

    # Motor support: a plate on top of the roll servo reaching out to the shell walls.
    rx, _, rz = roll.anchor
    top_of_roll = rz + XL330.case_width / 2
    support = asm.attach(roll.case, Component.world(
        "head_motor_support", box((rx - 11.5, -60, top_of_roll), (rx + 11.5, 60, top_of_roll + 2.5)) & cavity,
        PLA, DARK))
    top = asm.attach(support, Component.world("head_top_shell", at(shell & box((-80, -60, HEAD_SPLIT), (80, 60, 60))),
                                              PLA, SHELL, collision=True))
    asm.attach(top, Component.world("head_bottom_shell", at(shell & box((-80, -60, HEAD_BOTTOM - 1), (80, 60, HEAD_SPLIT))),
                                    PLA, ORANGE, collision=True))
    # Face plate fills the recess out to the shell walls.
    face = extrude(arch(face_w + 6, face_h, 23, 3), "YZ", at=fx - 11, center=(0, face_c), depth=1.5) & loft_cavity(
        **{**loft, "wall": 0.0})
    face = asm.attach(top, Component.world("face_plate", at(face), PLA, TAUPE))
    eye = (fx - 9.5, 0, 16)
    ring = asm.attach(face, Component.world("lens_ring", at(cyl(eye, FWD, 15, 8) - cyl(eye, FWD, 8.5, 10)), PLA, YELLOW))
    asm.attach(ring, Component.world("lens", at(cyl((fx - 11, 0, 16), FWD, 8.5, 6)), BOUGHT, LENS,
                                     mass_override=p.camera_g / 1000))
    asm.attach(face, Component.world("face_window", at(extrude(slot(12, 6), "YZ", at=fx - 9.6, center=(24, 16), depth=1.2)),
                                     PLA, INK))
    asm.attach(face, Component.world("head_electronics", at(box((fx - 21, -32, 1), (fx - 11, 32, 31))), BOUGHT, PCB,
                                     mass_override=p.electronics_g / 1000))
    asm.attach(support, Component.world("speaker", box((rx - 11.5, -17, top_of_roll + 2.5), (rx + 11.5, 17, top_of_roll + 9.5)),
                                        BOUGHT, color("#4a4a4a"), mass_override=p.speaker_g / 1000))
    # Jaw servo, upright just ahead of the head-yaw servo (head roll turns about x, so they
    # never meet), hanging from a plate across the shell. The jaw is not simulated as a joint:
    # the servo is locked and the jaw bolts to its horn. Then the soft TPU mouth.
    jx, jz, horn_y = 23.0, -0.5, 40.0
    mount = asm.attach(top, Component.world("jaw_servo_mount", at(box((13, 11, 24), (33, 60, 26.5))) & cavity,
                                            PLA, DARK))
    mount.connector("jaw_seat", at=(ox + jx, horn_y - XL330.horn_face, oz + 24), normal=DOWN, x=FWD)
    jaw_servo = servo("jaw_servo", XL330)
    asm.fasten(mount.c("jaw_seat"), jaw_servo.case.c("long_end"), spin=(jaw_servo.AXIS, LEFT))
    asm.lock(jaw_servo)
    arms = [extrude(polygon([(jx, jz), (62, -11.5)], round=6), "XZ", at=y0, depth=7) for y0 in (horn_y, -horn_y - 7)]
    lip = box((61, -horn_y - 7, HEAD_BOTTOM - 0.5), (66.5, horn_y + 7, -6))
    jaw = asm.attach(jaw_servo.horn, Component.world("jaw", at(arms[0] + arms[1] + lip), PLA, ORANGE))
    for s_ in (1, -1):
        asm.attach(jaw, Component.world(f"jaw_hinge_{'left' if s_ > 0 else 'right'}",
                                        at(cyl((jx, s_ * (horn_y + 7.6), jz), LEFT, 5, 1.2)), PLA, DARK))
    asm.attach(asm.components["head_bottom_shell"], Component.world("soft_mouth", at(box((25.6, -44, HEAD_BOTTOM), (57.8, 44, -7.3))) & cavity,
                                    TPU, ORANGE))
    top.sites["head_camera"] = (ox + fx, 0.0, oz + 16)
