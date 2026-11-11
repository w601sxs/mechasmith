"""Unitree Go2 quadruped as mechasmith code.

Torso, four legs (hip, thigh, calf), and the twelve GO-M8010-6 joint motors built from the
datasheet: armature, torque and speed limits follow from the gear ratio and rotor inertia, and
the motors are torque-controlled (an RL policy runs the PD). The abduction motors sit in the
torso and the thigh motors in the hips; each knee motor sits at the top of its thigh and drives
the knee through a rod with an extra 1.917:1 reduction. The torso is carved wherever a hip can
swing. Feet are rubber spheres with their own contact settings and touch sensors; the IMU reports
orientation, rates, accelerations and the body's velocity.

Layout, joint ranges and masses measured from the Menagerie `unitree_go2` model; the parts are
our own. Joint and body names match the reference, and so does Unitree's axis convention (both
sides share the same world axes).
"""
from mechasmith import (ABS, ALUMINIUM, BOUGHT, GO_M8010_6, TPU, Assembly, Component, box, color, cyl, design, enclose,
                        extrude, motor, param, polygon, swept_link)

BLACK, METAL, GRAY, WHITE = color("#000000"), color("#e6f2f2"), color("#abb1c5"), color("#ffffff")
X, Y, Z = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)
M = GO_M8010_6
FOOT_CONTACT = dict(friction=[0.8, 0.02, 0.01], condim=6, priority=1, solimp=[0.015, 1, 0.022, 0.5, 2.0])
# Cover sections measured from the Menagerie Go2 meshes (t along the axis, then up lo/hi and
# across lo/hi, mm; across = axis x up): the torso along +x from the body origin (up +z,
# across -y), the thigh and calf down -z from the thigh joint and the knee (up +x, across -y).
TORSO = [(-128, -77, 63, -97, 97), (-86, -77, 63, -97, 97), (-44, -57, 60, -97, 97), (-3, -57, 59, -97, 97),
         (39, -57, 65, -97, 97), (81, -96, 80, -97, 97), (123, -96, 82, -97, 97), (165, -84, 89, -97, 97),
         (207, -59, 89, -64, 65), (248, -97, 89, -64, 65), (290, -97, 81, -64, 65), (332, -97, 73, -45, 45)]
THIGH = [(-51, -48, 46, -15, 57), (-16, -48, 48, -17, 57), (19, -48, 48, -17, 57), (54, -44, 44, -12, 57),
         (88, -39, 19, -11, 13), (123, -37, 9, -12, 14), (158, -36, 7, -15, 19), (193, -33, 15, -16, 19),
         (227, -29, 15, -16, 19)]
CALF = [(-31, -20, 15, -7, 9), (2, -20, 16, -15, 15), (36, -20, 22, -17, 17), (69, -13, 28, -17, 17),
        (102, -2, 32, -17, 17), (136, 2, 32, -15, 15), (169, -21, 32, -18, 18), (203, -26, 31, -20, 20),
        (236, -26, 25, -20, 20)]
STAND = {f"{leg}_{j}": v for leg in ("FL", "FR", "RL", "RR") for j, v in (("thigh_joint", 0.9), ("calf_joint", -1.8))}


@design(
    params=dict(
        hip_xy=param((193.4, 46.5), doc="abduction axis (x fore/aft of the centre, y from the centre line)"),
        thigh_motor_y=param(72.0, (50, 90), "thigh motor centre from the centre line, outboard of the torso"),
        thigh_len=param(213.0, (180, 260), "thigh joint to knee"),
        calf_len=param(213.0, (180, 260), "knee to foot centre"),
        knee_reduction=param(1.917, (1.0, 3.0), "rod reduction from the knee motor to the knee"),
        leg_y=param(142.0, (110, 170), "leg plane (thigh to foot) from the centre line"),
        battery_g=param(2100.0, (800, 2500), "battery pack, low in the middle of the torso"),
        computer_g=param(1900.0, (500, 2500), "computer and boards, in the middle of the torso"),
        shell_g=param(1300.0, (400, 2500), "torso shell and frame"),
        torso_half_width=param(47.0, (47, 90), "half width of the torso between the hip modules (wider costs abduction range)"),
        thigh_wall=param(5.0, (2, 8), "wall of the hollow thigh casting"),
        covers=param(False, doc="moulded covers over the torso, thighs and calves, shaped to the Go2's measured sections"),
        cover_wall=param(2.0, (1.0, 4.0), "wall of the moulded covers"),
    ),
    home=STAND,
)
def build(p):
    hx, hy = p.hip_xy
    my = p.thigh_motor_y
    # The hip modules sit at the torso's corners; whatever they sweep about the abduction axes is
    # carved out of the torso, and a bridge over the top carries the head.
    hip_reach = my + M.length / 2 + 2
    rooms = [swept_link((sx * hx, sy * hy, 0), X, [(sx * hx, sy * hy, 0), (sx * hx, sy * hip_reach, 0)],
                        M.diameter + 6, (-1.0472, 1.0472), (-58, 58), gap=2.0)
             for sx in (1, -1) for sy in (1, -1)]
    shell = (box((-188, -47, -57), (188, 47, 57)) + box((-135, -p.torso_half_width, -57), (135, p.torso_half_width, 57))) \
        - box((-185, -44, -54), (185, 44, 54)) - box((-132, -p.torso_half_width + 3, -54), (132, p.torso_half_width - 3, 54))
    bridge = box((-250, -30, 55), (250, 30, 62))
    head = cyl((285, 0, 10), Z, 50, 90) + cyl((280, 0, -50), X, 40, 50)
    torso = shell + bridge + head
    for r in rooms:
        torso = torso - r
    frame_g = p.shell_g / 2 if p.covers else p.shell_g
    base = Component.world("base", torso, BOUGHT, BLACK, collision="box", mass_override=frame_g / 1000)
    asm = Assembly("go2", root=base, floating=True, root_link="base")
    battery = asm.attach(base, Component.world("battery", box((-80, -43, -54), (80, 43, -14)), BOUGHT, GRAY,
                                               mass_override=p.battery_g / 1000))
    asm.attach(battery, Component.world("computer", box((-70, -43, -14), (70, 43, 30)), BOUGHT, GRAY,
                                     mass_override=p.computer_g / 1000))
    base.sites["imu"] = (-25.6, 0.0, 42.3)
    asm.imu("imu", ("gyro", "accelerometer", "framequat", "framelinvel", "frameangvel", "framepos"))

    for leg, sx in (("FL", 1), ("RL", -1)):
        x = sx * hx
        thigh_range = (-1.5708, 3.4907) if sx > 0 else (-0.5236, 4.5379)
        # Abduction motor inside the torso, just behind the hip module; its flange faces the hip.
        abd = motor(f"{leg}_abduction_motor", M)
        flange_x = x - sx * (M.diameter / 2 + 3)
        base.connector(f"{leg}_abd_seat", at=(flange_x - sx * (M.flange_thickness + M.length), hy, 0),
                       normal=(sx, 0, 0), x=Z, check=False)
        asm.fasten(base.c(f"{leg}_abd_seat"), abd.case.c("back"))
        asm.joint(f"{leg}_hip_joint", abd, (-1.0472, 1.0472), reverse=sx < 0, child_link=f"{leg}_hip")
        # Hip: a housing round the thigh motor, bolted to the abduction flange.
        fx = abd.face("face").point[0]
        housing = cyl((x, my, 0), Y, M.diameter / 2 + 3.5, M.length + 2) - cyl((x, my, 0), Y, M.diameter / 2, M.length + 4)
        arm = box((min(fx, fx + sx * 8), hy - 28, -28), (max(fx, fx + sx * 8), my - M.diameter / 2 + 1, 28))
        hip = asm.attach(abd.horn, Component.world(f"{leg}_hip_housing", housing + arm, ALUMINIUM, METAL,
                                                   collision="cylinder"))
        thigh_motor = motor(f"{leg}_thigh_motor", M)
        hip.connector("thigh_motor_seat", at=(x, my - M.length / 2, 0), normal=Y, x=Z, check=False)
        asm.fasten(hip.c("thigh_motor_seat"), thigh_motor.case.c("back"))
        asm.joint(f"{leg}_thigh_joint", thigh_motor, thigh_range, child_link=f"{leg}_thigh")
        # Thigh: a hollow cast beam from the thigh flange out to the leg plane and down to the knee.
        # The knee motor sits coaxially outboard at its top and drives the knee through a rod.
        ty, ly, kz = thigh_motor.face("face").point[1], p.leg_y - 12, -p.thigh_len
        beam = extrude(polygon([(x, 0), (x, kz)], round=20), "XZ", at=ty, depth=ly - ty)
        w = p.thigh_wall
        hollow = extrude(polygon([(x, 0), (x, kz)], round=20 - w), "XZ", at=ty + w, depth=ly - ty - 2 * w)
        thigh = asm.attach(thigh_motor.horn, Component.world(f"{leg}_thigh_link", beam - hollow, ALUMINIUM, METAL,
                                                             collision="box"))
        knee_motor = motor(f"{leg}_knee_motor", M, reduction=p.knee_reduction)
        thigh.connector("knee_motor_seat", at=(x, ly, 0), normal=Y, x=Z, check=False)
        asm.fasten(thigh.c("knee_motor_seat"), knee_motor.case.c("back"))
        asm.lock(knee_motor)
        cy = p.leg_y
        # Calf in the leg plane, beside the thigh, and a rubber foot.
        tube = cyl((x, cy, kz - p.calf_len / 2 + 10), Z, 12, p.calf_len - 20) - cyl(
            (x, cy, kz - p.calf_len / 2 + 10), Z, 6, p.calf_len - 30)
        calf_shape = tube + cyl((x, cy, kz), Y, 15, 24)
        foot_shape = cyl((x, cy, kz - p.calf_len), Y, 22, 24) - cyl((x, cy, kz - p.calf_len), Y, 8, 30)
        if p.covers:
            # Moulded covers shaped to the Go2's own sections (measured from Menagerie), grown
            # wherever our parts would poke through, each with a web resting on its parts. At export
            # each is cut clear of everything that moves past it (clear_motion), so the covers
            # leave every joint its range.
            calf_cover = enclose([calf_shape], (0, 0, -1.0), X, wall=p.cover_wall, radius=12, sections=CALF,
                                 origin=(x, cy, kz), web=True, clear=[box((x - 60, cy - 40, kz - p.calf_len - 80),
                                                                          (x + 60, cy + 40, kz - p.calf_len + 30))])
            thigh_cover = enclose([beam - hollow, knee_motor.case.world_shape], (0, 0, -1.0), X, wall=p.cover_wall,
                                  radius=18, sections=THIGH, origin=(x, cy, 0), web=True)
            asm.attach(thigh, Component.world(f"{leg}_thigh_cover", thigh_cover, ABS, METAL, clear_motion=True))
        calf = Component.world(f"{leg}_calf_link", calf_shape, ALUMINIUM, GRAY, collision="capsule")
        asm.hinge(f"{leg}_calf_joint", thigh, calf, (x, cy, kz), Y, (-2.7227, -0.83776),
                  actuator=knee_motor.drive(), child_link=f"{leg}_calf")
        if p.covers:
            asm.attach(calf, Component.world(f"{leg}_calf_cover", calf_cover, ABS, METAL, clear_motion=True))
        foot = Component.world(f"{leg}_foot", foot_shape, TPU, BLACK, collision="sphere", contact=FOOT_CONTACT)
        asm.attach(calf, foot).sites[f"{leg}_foot"] = (x, cy, kz - p.calf_len)
        asm.touch(f"{leg}_foot", radius=23.0)
    if p.covers:
        # One cover over the torso and the four abduction motors, cut at export where the hips
        # and legs swing past; it carries the other half of the shell's mass.
        abd_x = hx - (M.diameter / 2 + 3) - M.flange_thickness - M.length / 2
        abd = [cyl((sx * abd_x, sy * hy, 0), X, M.diameter / 2, M.length) for sx in (1, -1) for sy in (1, -1)]
        cover = enclose([torso, *abd], X, Z, wall=p.cover_wall, radius=25, sections=TORSO, origin=(0, 0, 0), web=True)
        asm.attach(base, Component.world("torso_cover", cover, BOUGHT, WHITE, mass_override=p.shell_g / 2000,
                                         clear_motion=True))
    robot = asm.robot()
    return robot.mirror("FL_", "FR_", axes="same").mirror("RL_", "RR_", axes="same")
