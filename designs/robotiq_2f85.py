"""Robotiq 2F-85 adaptive gripper as mechasmith code (round 2: hinges and loops).

Each finger is the real five-bar linkage, all pivots parallel to y:
    base --driver--> driver --coupler--> coupler
    base --spring_link--> spring_link --follower--> follower (+ pad)
    coupler and follower are pinned together, closing the loop.
The driver is motorised; the spring link carries a return spring, so the pads stay
parallel until they touch something (the adaptive grasp). One motor drives both fingers:
the left driver is coupled to the right.

Pivots, joint ranges, springs and masses measured from the Menagerie `robotiq_2f85` MJCF
(BSD-2); the parts are our own. Joint and body names match the reference.
Frame: mm, fingers close along x, pivot axes along y, the pinch point up the z axis.
"""
from mechasmith import (ALUMINIUM, BOUGHT, TPU, ActuatorModel, Assembly, Component, box, color, cyl, design,
                        extrude, param, polygon)

BLACK, GRAY, PAD = color("#262626"), color("#767676"), color("#303030")
DRIVE = ActuatorModel(kp=100.0, kv=10.0, forcerange=5.0, damping=0.1, frictionloss=0.0, armature=0.005,
                      max_velocity=2.0, source="Menagerie robotiq_2f85 (position drive on the driver joints)")


@design(
    params=dict(
        driver_pivot=param((30.6, 65.7), doc="driver pivot on the base (x, z)"),
        coupler_pivot=param((62.1, 61.6), doc="coupler pivot on the driver (x, z), open pose"),
        loop_pin=param((68.2, 109.2), doc="coupler-follower pin (x, z), open pose"),
        spring_pivot=param((13.2, 71.7), doc="spring-link pivot on the base (x, z)"),
        follower_pivot=param((50.2, 115.7), doc="follower pivot on the spring link (x, z), open pose"),
        pad_face_x=param(42.7, (35, 55), "inner face of the pad at the open pose (85.4 mm opening)"),
        pad_z=param((122.7, 160.2), doc="pad bottom and top"),
        drive_g=param(494.0, (300, 900), "motor, gearbox and electronics inside the base"),
    ),
    base_height=0.0,
    poses={"half": {"right_driver_joint": 0.4}, "closed": {"right_driver_joint": 0.8}},
)
def build(p):
    A, B, C, D, E = p.driver_pivot, p.coupler_pivot, p.loop_pin, p.spring_pivot, p.follower_pivot
    Y = (0, -1.0, 0)                     # right-finger pivot axis: positive closes the finger
    xz = lambda q: (q[0], 0.0, q[1])
    plate = lambda pts, r, y0, y1: extrude(polygon(pts, round=r), "XZ", at=y0, depth=y1 - y0)
    pair = lambda pts, r, y0, y1: plate(pts, r, y0, y1) + plate(pts, r, -y1, -y0)   # a clevis: two cheeks

    # Base: mounting disc, hollow housing, and two towers carrying both fingers' pivots.
    shell = cyl((0, 0, 27.9), (0, 0, 1), 37, 34.2) - cyl((0, 0, 28.9), (0, 0, 1), 34, 30.2)
    tower = [(-40, 44), (40, 44), (40, 80), (20, 100), (-20, 100), (-40, 80)]
    base = Component.world("base", cyl((0, 0, 5.4), (0, 0, 1), 37, 10.8) + shell + pair(tower, 2, 4, 12.5),
                           ALUMINIUM, BLACK, collision=True)
    asm = Assembly("robotiq_2f85", root=base, floating=False, root_link="base")
    asm.attach(base, Component.world("drive", box((-15, -8, 13.8), (15, 8, 37.8)), BOUGHT, BLACK,
                                     mass_override=p.drive_g / 1000))
    base.sites["pinch"] = (0.0, 0.0, 155.8)

    # Right finger (the left one is its mirror image).
    driver = Component.world("right_driver", plate([A, B], 5.5, -3.5, 3.5), ALUMINIUM, GRAY)
    asm.hinge("right_driver_joint", base, driver, xz(A), Y, (0.0, 0.8), actuator=DRIVE, child_link="right_driver")
    coupler = Component.world("right_coupler", pair([B, C], 5.0, 4.0, 9.0), ALUMINIUM, BLACK)
    asm.hinge("right_coupler_joint", driver, coupler, xz(B), Y, (-1.57, 0.0), armature=0.001,
              child_link="right_coupler")
    spring = Component.world("right_spring_link", pair([D, E], 6.0, 13.0, 18.0), ALUMINIUM, BLACK)
    asm.hinge("right_spring_link_joint", base, spring, xz(D), Y, (-0.297, 0.8), stiffness=0.05, springref=2.62,
              damping=0.00125, armature=0.001, child_link="right_spring_link")
    top = (p.pad_face_x + 7.5, p.pad_z[1] - 10)
    follower = Component.world("right_follower", plate([C, E, top], 5.0, -3.5, 3.5), ALUMINIUM, BLACK)
    asm.hinge("right_follower_joint", spring, follower, xz(E), Y, (-0.873, 0.873), armature=0.001,
              child_link="right_follower")
    pad = box((p.pad_face_x, -11, p.pad_z[0]), (p.pad_face_x + 4, 11, p.pad_z[1]))
    asm.attach(follower, Component.world("right_pad", pad, TPU, PAD, collision=True))
    asm.hinge("right_loop_pin", coupler, follower, xz(C), Y)          # closes the four-bar

    robot = asm.robot().mirror("right_", "left_", plane="YZ", axes="pseudo")
    return robot.couple("left_driver_joint", "right_driver_joint")    # one motor, both fingers
