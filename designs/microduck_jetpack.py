"""Microduck with a jet pack: the mechasmith Microduck carrying two tilting ducted fans.

A backpack on the duck's back holds a 4S LiPo, two ESCs and the flight controller. Two
shoulder pylons carry an XL330 tilt servo each, and on each servo horn hangs a 70 mm
electric ducted fan (EDF) in a printed nacelle: bell-mouth inlet lip, fan rotor, stator
vanes round the motor, exit nozzle. It flies like a tilt-birotor (V-Coptr style):

    thrust and roll   differential fan thrust (fans at +-y)
    pitch             both nacelles tilt the same way (the fans sit above the centre of mass)
    yaw               the nacelles tilt opposite ways (fans at +-y push opposite ways)

The fans counter-rotate, so their reaction torques cancel in a hover. The duck's 14 servos
hold the standing pose in flight; the legs are the landing gear.

Feasibility (the numbers this design is built with; verify against the parts you buy):
duck 692 g + pack and fans ~0.6 kg = ~1.3 kg. A 70 mm 12-blade EDF on 4S makes ~1.2 kgf
(12 N) at ~55 A, so two give a thrust-to-weight of ~1.9: it flies, but only for about a
minute on a 1300 mAh pack (hover ~55 % throttle, ~60 A). The thrust passes through the
tilt axis, so the XL330 holds the nacelle with almost no torque; its ~11 rad/s speed and
position loop (~3 Hz) are fast enough for pitch and yaw, while roll and height, which need
to be faster, come from the fans' throttle.

Frame: the Microduck's (mm, x forward, y left, z up, origin between the hip-yaw servos).
"""
from dataclasses import replace

import numpy as np
from build123d import Axis, Cone, Plane, Pos, Rot, revolve

import microduck as md
from build123d import Kind, offset

from mechasmith import (BOUGHT, FWD, LEFT, PLA, UP, XL330, Component, Material, box, color, cyl, design, diamonds,
                        horn_plate, param, polygon, rounded_box, rounded_box_shell, servo, symmetric)

WHITE, ORANGE, YELLOW, DARK = md.SHELL, md.ORANGE, md.YELLOW, md.DARK
SLATE, STEEL, LIPO = md.SLATE, color("#b8bcc2"), color("#2a3b8f")

# Nacelle section (radius, height above the duct centre), mm: inner wall, bell-mouth lip,
# fairing, nozzle. Revolved about the fan axis.
DUCT = [(35.5, -28), (35.5, 26), (37.0, 33), (40.5, 36.5), (44.0, 35.5), (45.0, 31), (44.5, 0), (43.0, -26),
        (38.5, -40), (33.5, -40), (34.0, -34)]
LIP_Z, NOZZLE_Z = 29.0, -27.0        # orange above, dark below
# Foaming PLA (LW-PLA), the usual print material for RC ducted-fan jets: ~0.6 g/cm^3 as printed.
# Tilt servos: the same XL330, with its Position P gain raised (the duck's leg tuning, kp 0.55 N*m/rad,
# has a ~0.5 deg friction deadband, as big as the corrections a hover needs: the nacelles stick-slip).
TILT_SERVO = replace(XL330.actuator, kp=3.0, kv=0.05,
                     source="XL330 (microduck_rl model) with Position P gain raised ~5x for the tilt loop (verify)")
LW_PLA = Material("LW-PLA (foamed)", 600, printed=True, infill=0.0, wall_mm=1.0)


@design(
    params=dict(
        duct_x=param(-23.0, (-40, 10), "fan axes, fore/aft (put them over the centre of mass)"),
        duct_z=param(48.0, (30, 70), "fan centres above the trunk origin (sets the pitch lever)"),
        tilt_y=param(38.0, (34, 50), "tilt servo axis point, out from the centre line"),
        tilt_range=param(0.7, (0.2, 1.2), "nacelle tilt either way, rad"),
        thrust_max=param(12.0, (5, 20), "max thrust per fan, N (70 mm 12-blade EDF on 4S, ~1.2 kgf)"),
        torque_ratio=param(0.004, (0.0, 0.02), "fan reaction torque per newton (stator vanes cancel most swirl), m"),
        motor_g=param(62.0, (30, 120), "EDF motor (2700 kV inrunner class)"),
        fan_g=param(12.0, (5, 30), "12-blade fan rotor"),
        housing_g=param(28.0, (10, 60), "EDF housing tube with its stator vanes and tail cone"),
        lipo_g=param(160.0, (80, 300), "4S 1300 mAh LiPo in the pack"),
        esc_g=param(28.0, (10, 60), "60 A ESC, each"),
        fc_g=param(12.0, (0, 40), "flight controller + receiver"),
        duck_battery_g=param(100.0, (20, 200), "the duck's own servo battery (power it from the LiPo via a BEC and "
                                               "set this to ~20)"),
        pack_wall=param(1.8, (1.2, 3.0), "pack shell wall"),
    ),
    home=symmetric(md.STAND),
    poses={
        "tilt_forward": symmetric({**md.STAND}) | {"jet_left_tilt": 0.5, "jet_right_tilt": 0.5},
        "yaw_left": symmetric({**md.STAND}) | {"jet_left_tilt": -0.4, "jet_right_tilt": 0.4},
    },
)
def build(p):
    dp = md.build.resolve(battery_g=p.duck_battery_g)
    asm = md.trunk(dp)
    md.left_leg(asm, dp)
    md.neck_and_head(asm, dp)
    pack = jetpack(asm, p)
    for side in (1, -1):
        jet(asm, p, pack, side)
    robot = asm.robot().mirror("left_", "right_").print_settings(infill=dp.infill)
    robot.name = "microduck_jetpack"
    return robot


def jetpack(asm, p) -> Component:
    """The backpack: a rounded shell on the trunk's back with the LiPo, ESCs and FC inside."""
    trunk = asm.root
    x0, x1, hw, z0, z1, w = -92.0, -52.0, 48.0, -12.0, 64.0, p.pack_wall
    shell = rounded_box_shell((x0, -hw, z0), (x1, hw, z1), radius=14, wall=w)
    shell -= diamonds((x0, 0, 30), (-1, 0, 0), (0, 0, 1.0), rows=3, cols=5, size=7, pitch=10, depth=8)  # vents
    pack = asm.attach(trunk, Component.world("jetpack_shell", shell, LW_PLA, WHITE, collision="box"))
    # orange band round the pack, standing just proud of the shell
    band = (rounded_box((x0 - 0.8, -hw - 0.8, z0), (x1, hw + 0.8, z1 + 0.8), 14.8)
            - rounded_box((x0, -hw, z0 - 1), (x1 + 1, hw, z1), 14)) & box((x0 - 2, -60, 2), (x1, 60, 10))
    asm.attach(pack, Component.world("jetpack_band", band, PLA, ORANGE))
    asm.attach(pack, Component.world("jetpack_button", cyl((x0 - 1.2, 0, 50), FWD, 6, 2.4), PLA, YELLOW))
    floor = z0 + w
    lipo = asm.attach(pack, Component.world("jetpack_lipo", box((-86, -37, floor), (-58, 37, floor + 35)), BOUGHT,
                                            LIPO, mass_override=p.lipo_g / 1000))
    top = floor + 35
    for s, nm in ((1, "a"), (-1, "b")):
        asm.attach(lipo, Component.world(f"jetpack_esc_{nm}", box((-85, s * 3, top), (-59, s * 33, top + 8)), BOUGHT,
                                         color("#1f5fbf"), mass_override=p.esc_g / 1000))
    fc = asm.attach(asm.components["jetpack_esc_a"], Component.world(
        "jetpack_fc", box((-80, -15, top + 8), (-64, 15, top + 13)), BOUGHT, md.PCB, mass_override=p.fc_g / 1000))
    fc.sites["fc_imu"] = (-72.0, 0.0, top + 13)
    asm.imu("fc_imu", kinds=("gyro", "accelerometer", "framequat", "framelinvel", "frameangvel", "framepos"))
    return pack


def ring(profile, z_lo: float, z_hi: float, centre, wall: float = 1.0):
    """The duct section between two heights, revolved about the fan axis at `centre`: a
    hollow moulding, `wall` mm thick."""
    outer = polygon([(r, z) for r, z in profile], round=1.0)
    face = Plane.XZ * (outer - offset(outer, amount=-wall, kind=Kind.ARC))
    solid = revolve(face, Axis.Z, 360) & box((-60, -60, z_lo), (60, 60, z_hi))
    return Pos(*centre) * solid


def jet(asm, p, pack: Component, side: int) -> None:
    """One shoulder pylon, tilt servo and ducted fan; side = +1 left, -1 right."""
    nm = "left" if side > 0 else "right"
    ys, zc, xd = side * p.tilt_y, p.duct_z, p.duct_x
    seat_x = xd - XL330.case_below                     # the servo's long end, facing back to the pack
    pylon = rounded_box((-62, ys - 11, zc - 11), (seat_x, ys + 11, zc + 11), 4)
    pylon = asm.attach(pack, Component.world(f"jet_{nm}_pylon", pylon, PLA, DARK, collision="box"))
    pylon.connector("servo_seat", at=(seat_x, ys, zc), normal=FWD, x=UP)
    tilt = servo(f"jet_{nm}_tilt", XL330)
    asm.fasten(pylon.c("servo_seat"), tilt.case.c("long_end"), spin=(tilt.AXIS, side * LEFT))
    # +angle tilts the thrust forward on both sides
    asm.joint(f"jet_{nm}_tilt", tilt, (-p.tilt_range, p.tilt_range), reverse=side < 0,
              child_link=f"jet_{nm}_nacelle", actuator=TILT_SERVO)
    plate = horn_plate(XL330, f"jet_{nm}_yoke", (22, 30), thickness=3.0, color=DARK)
    asm.fasten(tilt.horn.c("face"), plate.c("horn"))
    y_plate = tilt.anchor[1] + side * (XL330.horn_face + 3.0)
    yd = y_plate + side * 47.5                         # duct axis: its fairing just clears the yoke
    C = np.array([xd, yd, zc])

    # Fairing (white), with a mount block reaching in to the yoke; orange lip; dark nozzle.
    lo_y, hi_y = sorted((y_plate, yd - side * 40))
    mount = box((xd - 11, lo_y, zc - 14), (xd + 11, hi_y, zc + 14)) - cyl(C, UP, 35.6, 80)
    duct = asm.attach(plate, Component.world(f"jet_{nm}_duct", ring(DUCT, NOZZLE_Z, LIP_Z, C) + mount, LW_PLA, WHITE,
                                             collision="box"))
    asm.attach(duct, Component.world(f"jet_{nm}_lip", ring(DUCT, LIP_Z, 40, C), LW_PLA, ORANGE))
    asm.attach(duct, Component.world(f"jet_{nm}_nozzle", ring(DUCT, -41, NOZZLE_Z, C), LW_PLA, SLATE))

    # EDF unit: stator vanes and tail cone (moulded), motor can, fan rotor and spinner.
    vanes = None
    for k in range(6):
        v = Pos(*C) * Rot(0, 0, 30 + 60 * k) * box((12, -0.9, -24), (36, 0.9, -8))
        vanes = v if vanes is None else vanes + v
    stator = vanes + Pos(*C) * Pos(0, 0, -38) * Cone(5, 14.5, 10, align=None) + cyl(C + (0, 0, -18), UP, 14.5, 20)
    stator = asm.attach(duct, Component.world(f"jet_{nm}_stator", stator, BOUGHT, color("#3a3d42"),
                                                mass_override=p.housing_g / 1000))
    motor = asm.attach(stator, Component.world(f"jet_{nm}_motor", cyl(C + (0, 0, -3), UP, 13.5, 10), BOUGHT, STEEL,
                                               mass_override=p.motor_g / 1000))
    blades = None
    for k in range(12):
        b = Pos(*C) * Pos(0, 0, 6) * Rot(0, 0, 30 * k) * Rot(side * 32, 0, 0) * box((11, -5, -0.6), (35, 5, 0.6))
        blades = b if blades is None else blades + b
    fan = asm.attach(motor, Component.world(f"jet_{nm}_fan", cyl(C + (0, 0, 6), UP, 12.5, 8) + blades, BOUGHT, YELLOW,
                                            mass_override=p.fan_g / 1000))
    asm.attach(fan, Component.world(f"jet_{nm}_spinner", Pos(*C) * Pos(0, 0, 10) * Cone(12.5, 2.5, 10, align=None),
                                    PLA, ORANGE))
    asm.rotor(f"jet_{nm}_fan", fan, C + (0, 0, 6), spin=side, thrust_max=p.thrust_max,
              torque_ratio=p.torque_ratio, source="70 mm 12-blade EDF on 4S, ~1.2 kgf static (estimate; verify)")
