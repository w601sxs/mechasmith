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
from mechasmith import (ALUMINIUM, BOUGHT, GO_M8010_6, TPU, Assembly, Component, box, color, cyl, design, extrude,
                        motor, param, polygon, swept_link)

BLACK, METAL, GRAY, WHITE = color("#000000"), color("#e6f2f2"), color("#abb1c5"), color("#ffffff")
X, Y, Z = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)
M = GO_M8010_6
FOOT_CONTACT = dict(friction=[0.8, 0.02, 0.01], condim=6, priority=1, solimp=[0.015, 1, 0.022, 0.5, 2.0])
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
    base = Component.world("base", torso, BOUGHT, BLACK, collision="box", mass_override=p.shell_g / 1000)
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
        # Calf in the leg plane, beside the thigh, and a rubber foot.
        cy = p.leg_y
        tube = cyl((x, cy, kz - p.calf_len / 2 + 10), Z, 12, p.calf_len - 20) - cyl(
            (x, cy, kz - p.calf_len / 2 + 10), Z, 6, p.calf_len - 30)
        calf = tube + cyl((x, cy, kz), Y, 15, 24)
        calf = Component.world(f"{leg}_calf_link", calf, ALUMINIUM, GRAY, collision="capsule")
        asm.hinge(f"{leg}_calf_joint", thigh, calf, (x, cy, kz), Y, (-2.7227, -0.83776),
                  actuator=knee_motor.drive(), child_link=f"{leg}_calf")
        foot = Component.world(f"{leg}_foot", cyl((x, cy, kz - p.calf_len), Y, 22, 24) - cyl(
            (x, cy, kz - p.calf_len), Y, 8, 30), TPU, BLACK, collision="sphere", contact=FOOT_CONTACT)
        asm.attach(calf, foot).sites[f"{leg}_foot"] = (x, cy, kz - p.calf_len)
        asm.touch(f"{leg}_foot", radius=23.0)
    robot = asm.robot()
    return robot.mirror("FL_", "FR_", axes="same").mirror("RL_", "RR_", axes="same")
