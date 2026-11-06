"""Microduck, reverse-engineered: Pollen Robotics' Microduck rebuilt as mechasmith code.

Nothing from Pollen's model is imported. Dimensions were measured from the public
microduck_rl MJCF (part bounding boxes, section profiles, material colours).

Every part is connected, never positioned: servos and brackets are fastened face to
face, and the drawn parts (shells, covers, plates) are attached to the parts they touch,
which mechasmith checks as it goes. So nothing can float. The left leg is built once
and mirrored with microduck_rl's axis convention (right = -left for symmetric motion).

Frame: mm, x forward, y left, z up; origin between the hip-yaw servo centres.
"""
from mechasmith import (BACK, BOUGHT, DOWN, FWD, LEFT, PLA, TPU, UP, XL330, Assembly, Component, MountFace,
                        arch, box, color, cyl, design, extrude, horn_plate, l_bracket, loft_cavity, loft_shell, move,
                        param, profile_plate, rect, rounded_box_cavity, rounded_box_shell, servo, slot, spacer,
                        symmetric, u_bracket)

# Head loaf: (x from the head-yaw axis, half width, top above the head-roll axis), mm.
HEAD_STATIONS = [(-58, 22, 14), (-56, 29, 23), (-46, 35, 30), (-36, 38, 33), (-16, 40.6, 35.6),
                 (4, 42.3, 37.3), (24, 43, 39.5), (44, 44, 40.5), (64, 44, 41)]
HEAD_BOTTOM, HEAD_SPLIT = -17.0, -3.0   # below the roll axis; white above the split, orange below

# Colours measured from the reference model's materials.
SHELL, SHELL_L, TAUPE = color("#d9d9d6"), color("#cccccc"), color("#aba399")
ORANGE, YELLOW, DARK, SLATE = color("#ff661f"), color("#f5b024"), color("#212621"), color("#596166")
LENS, BATTERY, PCB, INK = color("#42474c"), color("#4d4d4d"), color("#4f7d69"), color("#141417")
T = 2.5   # bracket plate thickness

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
        foot_clear=param(14.0, (13.8, 25), "ankle axis -> top of the foot (>= 13.8 frees the ankle)"),
        foot_size=param((54.0, 40.0), doc="foot length, width"),
        foot_forward=param(6.8, (-10, 25), "foot centre ahead of the ankle axis"),
        sole_bottom=param(25.0, (20, 40), "ankle axis -> bottom of the sole"),
        trunk_x=param((-52.0, 29.0), doc="trunk shell, back and front x"),
        trunk_half_w=param(32.0, (25, 40), "trunk shell half width"),
        trunk_z=param((-8.5, 33.5), doc="trunk shell, bottom and top z"),
        trunk_wall=param(1.6, (1.0, 3.0), "trunk shell wall"),
        battery_g=param(100.0, (20, 200), "upright pack behind the hips; 100 g holds HOME best on servos alone"),
        neck_x=param(20.0, (0, 30), "neck axis, fore/aft"),
        neck_z=param(23.4, (15, 40), "neck-pitch axis height; sets the neck mount's length"),
        head_pitch_reach=param(17.0, (16, 30), "U-bracket over the head-pitch servo"),
        roll_spacer=param(8.0, (2, 20), "head-yaw case -> head-roll horn"),
        head_wall=param(1.8, (1.0, 3.0), "head shell wall"),
        head_hole_r=param(26.0, (21, 30), "opening around the head-yaw axis"),
        electronics_g=param(25.0, (0, 60), "Raspberry Pi Zero 2 W + servo hat, behind the face"),
        speaker_g=param(8.0, (0, 30), "speaker at the back of the head"),
        camera_g=param(5.0, (0, 20), "camera module behind the lens"),
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
    return asm.robot().mirror("left_", "right_")


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

    plate = horn_plate(XL330, "left_hip_yaw_plate", (20, 26), color=DARK)
    asm.fasten(yaw.horn.c("face"), plate.c("horn"))
    roll = servo("left_hip_roll", XL330)
    asm.fasten(plate.c("out"), roll.case.c("short_end"), spin=(roll.AXIS, FWD))
    asm.joint("left_hip_roll", roll, (-0.384, 0.384))

    # Hip-pitch servo beside the roll servo, horn facing in, held by an L-bracket off the roll horn.
    target = roll.anchor + (p.pitch_x, p.roll_gap + T, 0)
    lb = asm.attach(roll.horn, l_bracket("left_hip_roll_bracket", roll.face("face"), MountFace(target, -LEFT),
                                         b_x=FWD, color=TAUPE))
    pitch = servo("left_hip_pitch", XL330)
    asm.fasten(lb.c("b"), pitch.horn.c("face"))
    asm.joint("left_hip_pitch", pitch, (-1.571, 1.571), reverse=True)

    # Thigh: a teardrop cover on the pitch servo's boss carries the knee servo's boss.
    px, py, pz = pitch.anchor
    knee_x, knee_z = px - XL330.case_below - p.knee_gap - XL330.case_width / 2, pz - p.knee_drop
    y_out = py + XL330.horn_face
    tear = [(px, pz), (knee_x - 1.5, pz + 2.5), (knee_x, knee_z)]
    cover = asm.attach(pitch.case, Component.world("left_thigh_cover", profile_plate(tear, y_out, y_out + T, 12.0),
                                                   PLA, SHELL))
    cover.connector("knee_seat", at=(knee_x, y_out, knee_z), normal=-LEFT, x=FWD)
    knee = servo("left_knee", XL330)
    asm.fasten(cover.c("knee_seat"), knee.case.c("boss"), spin=(knee.BODY, UP))
    for nm, (x, z) in (("pitch", (px, pz)), ("knee", (knee_x, knee_z))):
        asm.attach(cover, Component.world(f"left_{nm}_bearing", cyl((x, y_out + T + 0.5, z), LEFT, 4.5, 1.0),
                                          BOUGHT, DARK, mass_override=0.002))
    asm.joint("left_knee", knee, (-1.571, 1.571))

    # Shin: block on the knee horn, then a plate down the inside to the ankle servo's boss.
    kx, ky, kz = knee.anchor
    y_horn, y_boss, za = ky - XL330.horn_face, ky - p.ankle_in - XL330.horn_face, kz - p.shin_len
    shin = box((kx - 9.5, y_boss - T, za - 9.5), (kx + 9.5, y_boss, kz + 9.5)) \
        + box((kx - 9.5, y_boss, kz - 9.5), (kx + 9.5, y_horn, kz + 9.5))
    shin = asm.attach(knee.horn, Component.world("left_shin_plate", shin, PLA, TAUPE))
    shin.connector("ankle_seat", at=(kx, y_boss, za), normal=LEFT, x=FWD)
    ankle = servo("left_ankle", XL330)
    asm.fasten(shin.c("ankle_seat"), ankle.case.c("boss"), spin=(ankle.BODY, UP))
    asm.joint("left_ankle", ankle, (-1.571, 1.571))

    # Foot: orange bracket off the ankle horn, an orange cup with side rims, a yellow TPU sole.
    xa, yc, za = ankle.anchor
    top = za - p.foot_clear
    fl, fw = p.foot_size
    fx, z_sole = xa + p.foot_forward, za - p.sole_bottom
    fb = asm.attach(ankle.horn, l_bracket("left_foot_bracket", ankle.face("face"), MountFace((xa, yc, top), DOWN),
                                          width=24, color=ORANGE))
    cup = extrude(rect(fl, fw, 8.0), "XY", at=z_sole + 3.5, center=(fx, yc), depth=top - z_sole - 3.5)
    for s in (1, -1):  # rims along both long sides, clear of the ankle case
        cup += extrude(rect(fl - 6, 2.4, 1.1), "XY", at=top, center=(fx, yc + s * (fw / 2 - 1.2)), depth=8.0)
    cup = asm.attach(fb, Component.world("left_foot", cup, PLA, ORANGE))
    sole = asm.attach(cup, Component.world("left_sole", extrude(rect(fl, fw, 8.0), "XY", at=z_sole, center=(fx, yc),
                                                                 depth=3.5), TPU, YELLOW, collision=True))
    sole.sites["left_foot"] = (xa, yc, z_sole)


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

    roll_spacer = spacer("head_roll_spacer", p.roll_spacer, color=DARK)
    asm.fasten(yaw.case.face(BACK), roll_spacer.c("a"))
    roll = servo("head_roll", XL330)
    asm.fasten(roll_spacer.c("b"), roll.horn.c("face"), spin=(roll.BODY, -LEFT))
    asm.joint("head_roll", roll, (-0.436, 0.436), reverse=True, child_link="head")
    head(asm, p, roll, origin=(yaw.anchor[0], 0.0, roll.anchor[2]))


def head(asm: Assembly, p, roll, origin) -> None:
    """The Microduck head, drawn in a frame at the head-yaw axis and head-roll height."""
    def at(shape):
        return move(shape, origin)

    ox, _, oz = origin
    fx = HEAD_STATIONS[-1][0]
    face_c, face_w, face_h = (HEAD_SPLIT + 38) / 2, 82.0, 38 - HEAD_SPLIT
    loft = dict(stations=HEAD_STATIONS, bottom=HEAD_BOTTOM, wall=p.head_wall, radius=26.0, bottom_radius=6.0)
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
    # Jaw hinge lobes, set into the shell, with the hinge horn on each.
    for s, nm in ((1, "left"), (-1, "right")):
        y0, y1 = sorted((s * 40.5, s * 45.1))
        lobe = asm.attach(top, Component.world(f"jaw_lobe_{nm}", at(profile_plate([(-12, HEAD_SPLIT), (14, HEAD_SPLIT),
                                                                                  (-1, 12)], y0, y1, 4)), PLA, ORANGE))
        asm.attach(lobe, Component.world(f"jaw_horn_{nm}", at(cyl((-1, s * 45.6, 8), LEFT, 5, 1.2)), PLA, DARK))
    top.sites["head_camera"] = (ox + fx, 0.0, oz + 16)
