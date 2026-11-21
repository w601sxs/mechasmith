# API reference

Everything below is imported from `mechasmith`; designs never import build123d. Start from
[the README](../README.md) for the idea, and from a bundled example (`mechasmith copy so101`) for a
working file.

## Connecting parts

How parts get attached:
- `fasten(a, b, spin=..., at=...)` puts b's connector face flat on a's (they touch by construction).
  `at=<world point>` seats b elsewhere on a's face; the point is checked to be on a's surface.
  Servos, U-brackets, horn plates, spacers and L-brackets come with connectors.
- `attach(base, part)` is for parts drawn in world coordinates (shells, covers). Contact is checked
  on that line; a gap raises `AttachmentError: head does not touch neck_pitch_case: the gap is 7.61 mm`.
- Parts are placed as you connect them, root outward, so later parts can use earlier positions
  (`servo.anchor`, `servo.face("face")`). Connecting a placed part again must agree with where it is.
- `servo(...).case.face(BACK)` picks whichever connector currently faces a world direction.
- `loft_cavity`, `rounded_box_cavity`: intersect a plate with a shell's cavity so it reaches the
  walls exactly (how the Microduck head motor support and hip plate are made).
- `asm.link(name, around=servo, path=[...], width=, holds=Hold(next_servo, at, axis, body))` draws a
  whole printed link: a body across the full width of the servo it turns on, cut clear of that servo
  over the declared joint range (that leaves the fork), plus a pocket seating the next servo
  (`Hold(..., band=28)` for an open ring holder, `wrap=` to cover only part of the case). Both contacts
  hold by construction; a link that comes out in pieces is an error. `printed(...)` is the same
  without a fork (bases, gripper bodies); `on=part` attaches a link without a fork.
- `swept_link(anchor, axis, path, width, range, span)` is the room a link needs as it turns; give
  it to the parent's `clear=` and the parent can be as big as you like without blocking the joint.
  `diamonds(...)` makes lightening-hole grids for `cut=`.
- `asm.hinge(name, a, b, at, axis, range)` is a pin with no servo (passive, or give it an `actuator`;
  `stiffness`/`springref` for a return spring). Its axis must pass through both parts. A hinge
  between two parts that are already connected closes a loop (four-bars, parallel grippers),
  exported as an MJCF `equality connect`. `robot.couple(follower, leader)` drives two joints from
  one motor. Poses of such mechanisms are solved, so the sweep, renders and viewer move the linkage.
- `asm.rotor(name, part, at, spin=, thrust_max=, torque_ratio=)` adds a motor and propeller (thrust
  plus reaction torque, an MJCF site actuator); `asm.imu(site)` adds gyro, accelerometer and
  orientation sensors. Rotorcraft get a hover keyframe and a hover test instead of the stand test.
- `motor(name, MotorSpec)` is a cylindrical joint motor built from its datasheet (gear ratio and rotor
  inertia give armature, torque and speed limits), torque-controlled by default; `motor.drive(1.9)`
  drives a joint remotely through a belt or rod. `collision="box"|"sphere"|"capsule"|...` fits primitive
  collision shapes, `contact=` sets friction and softness, `asm.touch` adds foot sensors, and every
  build exports `home` and pose keyframes. `mirror(axes="same")` keeps both sides' axes (Unitree).
- `asm.chain(parent, [ChainJoint(name, at, axis, range, actuator, hub=, mass=), ...])` builds a limb from
  a joint table (how a URDF describes arms and humanoids): motors on the parent side, flanges and
  struts on the child side, no overlaps by construction, `hub=None` for cross pieces. `mass=` (or
  `Component.mass_budget`) is the whole link's mass; its structure takes what the motors leave.
- `asm.robot()` groups rigidly connected parts into links; `joint` splits them.
  `robot.mirror("left_", "right_")` then makes the other side.

## Parameters

Parameters are declared with ranges, so `mechasmith describe` (and any agent) can list the knobs,
and every tool checks overrides against the ranges:

```bash
uv run mechasmith describe designs/microduck.py
uv run mechasmith build designs/microduck.py --set battery_g=80 --set shin_len=45
```

## Geometry, shells and covers

Geometry helpers, all from `mechasmith`: `UP DOWN FWD BACK LEFT RIGHT`, `box`, `cyl`, `rect`, `slot`,
`circle`, `polygon`, `extrude(outline, plane, at, center, depth)`, `move`, `color("#ff661f")`, and shells
(`loft_shell`, `arch`, `rounded_box_shell`); `polygon(points, round=)` outlines any plate or link.
Covers: `enclose(shapes, axis, up, wall=, ...)` lofts a smooth hollow cover round a group of parts,
fitted to them or to given `sections` (measured from a drawing or a reference mesh with
`enclose_sections`; `contain=True` grows them where the parts would poke through), with an inner
`web` that rests on the parts; `sweep(shapes, anchor, axis, range)` is the region a link sweeps
about a joint and `turn` poses shapes, for `enclose(clear=[...])`. Covers are exact mesh booleans
turned into faceted solids. Attach a cover with `Component.world(..., clear_motion=True)` and it is
cut clear of everything that moves past it at export, after mirroring (each joint swept through
its range with the others at home, as the joint sweep check does): covers never take range from a
joint. A parent-side cover wins over child-side ones; see `src/mechasmith/covers.py`. Designs need no build123d import;
everything returned is a build123d shape, so it is still there for anything unusual.

## Training models

`--sim mjx` (or `mjcf.export(..., sim="mjx")`) also writes `robot_mjx.xml`: one self-contained file
for GPU training with MJX. It keeps the bodies, joints, inertias, actuators, sensors, sites and
keyframes, drops the visual meshes, turns every collision shape into a primitive (mesh collisions
become fitted boxes; declare `collision="box"` etc. on parts to choose), lets the robot touch only
the floor, adds the floor, and uses a 4 ms step with a lighter solver. The build checks that it
loads (and in MJX, when JAX is installed).

## Conventions
- One world frame at the home pose: millimetres, x forward, y left, z up. Drawn parts
  (`Component.world`) use it; generated parts (servos, brackets, spacers) have their own frame and are
  placed by their connectors.
- Servo frame: +X is the output axis (horn side), +Z points to the short end of the case.
- A servo is a joint: `asm.joint(name, servo, range)` turns its case and horn about its axis.
  Joint anchors, axes and every MJCF body frame are derived; none are written by hand.
- Servos carry geometry, mass and a fitted actuator model (`XL330`, with Pollen's fitted parameters).
- Parts can't be positioned by hand. The only way into a robot is a connection, which is what
  guarantees that everything is attached.

## What `build` checks
- MJCF compiles, and the scene reloads from disk.
- `robot.urdf` (Isaac Sim / Isaac Lab, ROS, PyBullet, Drake) is written from the same tree, meshes and
  inertials: link frames at the joint anchors with identity orientation, revolute joints with the
  servo's limits, effort, velocity, damping and friction; fixed-base robots bolted to a `world` link;
  sites as `<site>_frame` links. Servo position gains have no URDF field and stay in the MJCF.
  `scripts/check_urdf.py` loads it with MuJoCo and with yourdfpy and compares it with the MJCF.
- Mass and inertia come from the CAD solids, with a 3D-printing infill model for printed parts.
- Exact mesh-boolean interference at the home pose, then a sweep of every joint reporting its usable range and the first parts that collide.
- Contact-sheet renders of the home pose and any named `POSES`.
- Fixed-base arms with a `tool_site` get an arm check: the torque each servo needs to hold `HOME`
  plus `payload_kg` at the tool (as a fraction of its limit), the sag that implies, and the reach.
- Floating-base robots get a stand test: hold `HOME` with the servos alone for 3 s, then report whether it
  stays up, where it settles, per-servo torque load, the support margin, and whether servo stiffness alone
  could hold the pose upright (`pd_stiffness_ratio`; below 1 means a balance controller is required).
- `viewer.html`: a self-contained browser viewer with a slider per joint, the design's poses, and the
  measured collision-free ranges.
