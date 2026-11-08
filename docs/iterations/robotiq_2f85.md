# Robotiq 2F-85: implement → critique → improve → reimplement

Reference: MuJoCo Menagerie `robotiq_2f85` (BSD-2). Each finger is a closed five-bar linkage
(driver, coupler, spring link, follower) driven by one motor for both fingers; a spring on the
spring link keeps the pads parallel until they touch something.

## Round 1: the library as it was ([robotiq_2f85_round1.py](robotiq_2f85_round1.py))

Workarounds the API forced:
- every joint had to be a servo, so the driver pivot is a fake 1 mm `PIN` servo, placed with
  connector gymnastics;
- closed loops were rejected, so each finger is an open chain (coupler and follower bolted to
  the driver) and the spring link is dropped;
- one motor could not drive two joints, so each finger got its own actuator;
- `attach` reported a motor inside its housing, and a pin horn inside a link, as "not touching".

Result (right pad, driver swept 0 → 0.8 rad, reference loops solved by simulation):

| driver | reference in / up / tilt | round 1 in / up / tilt |
|---|---|---|
| 0.4 | 20.1 / 10.8 / −0.9° | 30.8 / 0.5 / −22.9° |
| 0.8 | 41.7 / 13.2 / −1.4° | 59.4 / −11.0 / −45.8° |

The pads swing like clock hands and would cross; mass 754 g vs 1053 g.

## Library changes

- `asm.hinge(name, a, b, at, axis, range, actuator=None, stiffness=, springref=, damping=,
  armature=, pin=3.0)`: a pin with no servo (passive unless given an actuator). The pin's axis must
  pass through both parts (within `pin` mm, so clevis forks and bored holes count). If `b` is
  already connected the pin closes a loop: kept as an MJCF `equality connect`, with the pinned
  pair excluded from contact.
- `Robot.couple(follower, leader, ratio)`: one joint follows another (MJCF `equality joint`, URDF
  `<mimic>`); the follower's actuator is dropped. `Robot.mirror` copies loops and couplings.
- Joint springs (`stiffness`, `springref`) and per-joint damping and armature in the MJCF.
- `attach`: a part that sits inside another counts as touching.
- `sim.solve_pose`: in a model with loops or couplings, a pose given by the driven joints is
  solved (the actuators hold them, the rest settles under the constraints and springs, gravity
  and contacts off). The joint sweep, the renders and the viewer use it; the sweep covers only
  the driven joints, and the viewer's driver slider moves the joints that follow it.
- URDF: couplings as `<mimic>`; loop pins listed in comments (URDF cannot express loops).

## Round 2 ([designs/robotiq_2f85.py](../../designs/robotiq_2f85.py))

The real linkage: two open chains per finger, a hinge that closes the loop, the right finger
mirrored to the left, the left driver coupled to the right.

| driver | reference in / up / tilt | round 2 in / up / tilt |
|---|---|---|
| 0.4 | 20.1 / 10.8 / −0.9° | 20.8 / 10.3 / −1.2° |
| 0.8 | 41.7 / 13.2 / −1.4° | 42.5 / 12.1 / −1.8° |

- All 8 joint axes on the reference's (0.0 mm), same ranges and signs; pinch point exact.
- Mass 1056 g vs 1053 g; mass beyond every joint within 2 g.
- Loop pins hold to < 0.1 mm through the stroke. The jaws meet at 0.73 rad (the reference's
  stop at 0.8 rad leaves 1.8 mm).
- URDF loads in MuJoCo and yourdfpy and matches the MJCF (tree, mimic, loops noted).

## Still open
- URDF cannot carry the loops; Isaac Lab users need the MJCF, or a USD export with loop joints.
- `couple` across sides has to come after `mirror` (the left joints do not exist before it).
- Pads are plain blocks; the reference's pads are silicone on a metal backing.
