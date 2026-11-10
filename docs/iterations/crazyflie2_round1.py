"""Bitcraze Crazyflie 2.1 nano quadcopter, ROUND 1: the library as it was.

The parts are there (frame PCB, battery, deck, headers, motors, mounts, props) with real
masses, but the API has no way to make it fly: no thrust or rotor actuators, no IMU
sensors, and the only floating-base check is a stand test.

Layout measured from the Menagerie `bitcraze_crazyflie_2` model (MIT): body origin at the
centre, z up, motors at (+-32.7, +-32.7) mm, 92 mm motor to motor.
"""
import numpy as np

from mechasmith import BOUGHT, Assembly, Component, Material, box, color, cyl, design, extrude, param, polygon

FR4 = Material("FR4", 1850)                 # circuit board
ABS = Material("ABS (moulded)", 1050)       # injection-moulded mounts and props
BLACK, GREEN, GOLD = color("#1a1a1a"), color("#1c2f00"), color("#f7e099")
SILVER, WHITE, PROP, BATT = color("#e5e5e5"), color("#ffffff"), color("#caD1ee"), color("#a1a8ad")
UP = (0, 0, 1.0)


@design(
    params=dict(
        motor_xy=param(32.7, (25, 45), "motor axis offset from the centre in x and y (92 mm diagonal)"),
        battery_g=param(7.1, (5, 12), "250 mAh LiPo"),
        motor_g=param(2.0, (1.5, 3.0), "7 x 16 mm coreless motor"),
        electronics_g=param(2.0, (0, 5), "components on the frame PCB"),
    ),
    base_height="auto",
)
def build(p):
    a = p.motor_xy
    corners = [(sx * a, sy * a) for sx in (1, -1) for sy in (1, -1)]
    frame = (box((-15, -15, 2.6), (15, 15, 4.2))
             + extrude(polygon([(-a + 4, -a + 4), (a - 4, a - 4)], round=4), "XY", at=2.6, depth=1.6)
             + extrude(polygon([(-a + 4, a - 4), (a - 4, -a + 4)], round=4), "XY", at=2.6, depth=1.6))
    root = Component.world("frame", frame, FR4, BLACK, collision=True)
    asm = Assembly("crazyflie2", root=root, floating=True, root_link="cf2")
    asm.attach(root, Component.world("electronics", box((-8, -8, 4.2), (8, 8, 4.5)), BOUGHT, BLACK,
                                     mass_override=p.electronics_g / 1000))
    batt = asm.attach(root, Component.world("battery", box((-10.05, -14.9, 4.2), (10.05, 14.9, 11.1)), BOUGHT, BATT,
                                            mass_override=p.battery_g / 1000))
    asm.attach(batt, Component.world("deck", box((-13.6, -10.4, 11.1), (13.6, 10.4, 12.0)), FR4, GREEN))
    for s in (1, -1):
        asm.attach(root, Component.world(f"header_{'pos' if s > 0 else 'neg'}",
                                         box((s * 10.0, -8.4, 0.5), (s * 10.8, 8.4, 18.5)), BOUGHT, GOLD,
                                         mass_override=0.0005))
    for i, (x, y) in enumerate(corners):
        u = np.array([x, y]) / np.hypot(x, y)
        arm = (x - u[0] * 12, y - u[1] * 12)
        mount = extrude(polygon([arm, (x, y)], round=5), "XY", at=-1.2, depth=5.4) \
            - cyl((x, y, 2.0), UP, 3.1, 10)
        leg = box((x + u[0] * 4 - 0.5, y + u[1] * 4 - 2, -12.5), (x + u[0] * 4 + 0.5, y + u[1] * 4 + 2, -1.2))
        mount = asm.attach(root, Component.world(f"mount_{i}", mount + leg, ABS, WHITE, collision=True))
        motor = asm.attach(mount, Component.world(f"motor_{i}", cyl((x, y, 2.9), UP, 3.1, 16.6), BOUGHT, SILVER,
                                                  mass_override=p.motor_g / 1000))
        blade = [(x + r * c, y + r * s_) for r, c, s_ in ((-22.5, 0.7071, 0.7071), (22.5, 0.7071, 0.7071))]
        asm.attach(motor, Component.world(f"prop_{i}", extrude(polygon(blade, round=2.5), "XY", at=11.2, depth=1.0),
                                          ABS, PROP))
    root.sites["imu"] = (0.0, 0.0, 0.0)
    return asm.robot()
