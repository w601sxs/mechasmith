"""Robotiq 2F-85 gripper, ROUND 1: written with the library as it was (servo joints only,
no closed loops). Kept to show what the API could not express; see round 2.

Workarounds forced by the API:
  * every pivot must be a servo, so the driver pivot is a fake 1 mm "PIN" servo;
  * no closed loops, so each finger is an open chain: driver -> (coupler + follower bolted
    rigid); the spring link is dropped, and the pad no longer stays parallel;
  * no coupling, so each finger has its own actuator instead of one motor for both.

Pivots measured from the Menagerie `robotiq_2f85` MJCF (BSD-2): fingers move in the XZ plane.
"""
import numpy as np

from mechasmith import (ALUMINIUM, BOUGHT, DOWN, PLA, ActuatorModel, Assembly, Component, ServoSpec, box, color,
                        cyl, design, extrude, polygon, servo)

BLACK, GRAY = color("#262626"), color("#767676")
# A fake servo so a pivot can exist at all: 1 mm case, 3 mm horn.
PIN = ServoSpec(name="PIN", mass=0.0005, case_half_axial=0.5, case_width=1.0, case_above=0.5, case_below=0.5,
                horn_radius=3.0, horn_thickness=0.5, boss_thickness=0.0,
                actuator=ActuatorModel(kp=10.0, kv=0.0, forcerange=5.0, damping=0.1, frictionloss=0.0,
                                       armature=0.005, max_velocity=2.0, source="Menagerie 2f85 driver class"))

# Right-finger pivots (x, z), mm, at the open pose; the left finger is the mirror image in x.
A, B, C, E = (30.6, 65.7), (62.1, 61.6), (68.2, 109.2), (50.2, 115.7)


@design(params={}, base_height=0.0, poses={"closed": {"right_driver_joint": 0.8, "left_driver_joint": 0.8}})
def build(p):
    housing = cyl((0, 0, 27.9), (0, 0, 1), 37, 34.2) - cyl((0, 0, 28.9), (0, 0, 1), 34, 30.2) \
        + box((-20, -10, 44), (20, 10, 100))
    base = Component.world("base", housing, ALUMINIUM, BLACK, collision=True)
    asm = Assembly("robotiq_2f85", root=base, floating=False, root_link="base")
    asm.attach(base, Component.world("motor", box((-15, -8, 13.8), (15, 8, 37.8)), BOUGHT, BLACK,
                                     mass_override=0.35))
    for side, s in (("right", 1), ("left", -1)):
        ax, az = A
        base.connector(f"{side}_pivot", at=(s * ax, 7.0 * s, az), normal=(0, -s, 0), check=False)  # pin outside the driver
        pin = servo(f"{side}_driver_pin", PIN)
        asm.fasten(base.c(f"{side}_pivot"), pin.case.c("back"))
        asm.joint(f"{side}_driver_joint", pin, (0.0, 0.8), child_link=f"{side}_driver")
        X = lambda pts: [(s * x, z) for x, z in pts]
        driver = extrude(polygon(X([A, B]), round=6.5), "XZ", at=-5.5, depth=11)
        driver = asm.attach(pin.horn, Component.world(f"{side}_driver", driver, ALUMINIUM, GRAY))
        # coupler and follower bolted rigid to the driver: the loop cannot be closed
        coupler = extrude(polygon(X([B, C]), round=5), "XZ", at=-11.5, depth=23)
        coupler = asm.attach(driver, Component.world(f"{side}_coupler", coupler, ALUMINIUM, BLACK))
        follower = extrude(polygon(X([C, E, (50.2, 150)]), round=6), "XZ", at=-7, depth=14)
        follower = asm.attach(coupler, Component.world(f"{side}_follower", follower, ALUMINIUM, BLACK))
        pad = box((s * 45, -11, 122.7), (s * 49.5, 11, 160.2))
        asm.attach(follower, Component.world(f"{side}_pad", pad, PLA, BLACK, collision=True))
    base.sites["pinch"] = (0.0, 0.0, 155.8)
    return asm.robot()
