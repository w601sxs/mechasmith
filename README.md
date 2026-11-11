# mechasmith

Agent-first robot design: CAD as Python code → computed mass properties → MJCF and URDF → checks → renders, in one command.

```bash
uv run mechasmith build designs/so101.py        # SO-101 arm (6x STS3215): MJCF, sweeps, arm check, viewer
uv run python scripts/compare_joints.py <reference.xml> build/so101 --site gripper=gripper
uv run mechasmith build designs/microduck.py    # Pollen's Microduck, reverse-engineered as code-CAD
uv run python scripts/compare_render.py <reference scene.xml> build/microduck designs/microduck.py compare.png
uv run python benchmarks/compare_microduck.py build/microduck   # axis-line placement and masses vs Microduck
uv run python scripts/sweep_balance.py designs/microduck.py battery_g 40,70,100   # CoM vs ankles per value
uv run python scripts/balance.py build/microduck       # where the mass sits relative to the feet
uv run mechasmith build designs/duck_leg.py     # writes build/duck_leg/{scene.xml, robot.xml, robot.urdf, meshes/, views.png, report.json}
uv run --with yourdfpy python scripts/check_urdf.py build/so101   # URDF matches the MJCF (masses, inertias, joints, frames)
uv run python benchmarks/compare_leg.py build/duck_leg   # compare against Microduck's reference numbers
uv run python scripts/sweep_param.py designs/duck_leg.py foot_reach left_ankle 12,14,16
uv run python scripts/compare_looks.py <reference.xml> build/go2/robot.xml --png looks.png --bodies   # outline overlap, 3 views
```

## Writing a design
A design is one Python file with one `@design` function. You make parts and **connect** them;
mechasmith places them. Nothing is positioned by hand, so nothing can float.

```python
from mechasmith import (LEFT, UP, XL330, Assembly, Component, box, design, param, servo, u_bracket)

@design(params=dict(neck_z=param(23.4, (15, 40), "neck axis height, mm")))
def build(p):
    trunk = Component.world("trunk", box((-50, -30, -10), (30, 30, 30)))
    trunk.connector("neck_seat", at=(20, 0, 30), normal=UP)        # checked: must be on the surface

    asm = Assembly("bot", root=trunk)
    mount = u_bracket(XL330, "neck_mount", reach=p.neck_z - 30 - 2.5)
    asm.fasten(trunk.c("neck_seat"), mount.c("base"), spin=(mount.AXIS, LEFT))  # face to face
    neck = servo("neck_pitch", XL330)
    asm.fasten(mount.c("horn"), neck.horn.c("face"))
    asm.joint("neck_pitch", neck, range=(-1.57, 1.05))             # the servo is the joint
    head = Component.world("head", box(...))
    asm.attach(neck.case, head)                                    # checked now: must touch
    return asm.robot()                                             # links come from the connections
```

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

Parameters are declared with ranges, so `mechasmith describe` (and any agent) can list the knobs,
and every tool checks overrides against the ranges:

```bash
uv run mechasmith describe designs/microduck.py
uv run mechasmith build designs/microduck.py --set battery_g=80 --set shin_len=45
```

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

## Status
- Day 1: a 5-DOF duck-class leg (XL330 servos) with collision-free full joint ranges.
- Day 2: a stylised 14-DOF duck (since removed in favour of `microduck.py`) with Microduck's joint names and axis signs. Every leg axis
  lies within 0.3 mm of Microduck's; all joints but the knees and ankles keep their full declared range;
  it stands on servos alone at Microduck's HOME pose (settling 19 degrees nose-down; Pollen's own model
  falls over under the same test).
- Known gaps: neck sits 16.6 mm higher than Microduck's; mass 464 g vs 737 g (no fasteners, bearings,
  cables or speaker modelled); shells are simple ellipsoids and rounded boxes.
- Day 3: `designs/microduck.py` reverse-engineers Microduck's look (loaf head with face plate, lens ring
  and window, orange band and jaw lobes; compact trunk with upright battery; teardrop thigh covers;
  orange cup feet) from measured part boxes, section profiles and material colours. No Pollen meshes are
  used. All 14 joint axes lie within 0.8 mm of Microduck's; it stands at HOME on servos alone (3 degrees,
  20 mm margin) with a 100 g battery chosen by sweep. New shell helpers: `loft_body`/`loft_shell`
  (lofted sections, optionally arched), `arch`, and plate outlines (now `polygon` + `extrude`).
- Known gaps: 543 g vs 737 g; round hip-roll brackets and the fixed jaw servo are not modelled.
- Day 4: design API (`@design`, typed `param`s with ranges, `symmetric` poses), `Robot.mirror`,
  `Part.mirrored` and a geometry toolkit. `microduck.py` builds one leg and mirrors it; its output is
  identical to before (mass, parts, joint axes and ranges, sweeps, stand test).
- Day 5: assemblies. Parts are connected (`fasten`, `attach`, `joint`) and mechasmith places them,
  checking contact as it goes. A contact scan of the previous Microduck found the head shell and the
  head electronics floating; now the head hangs off a motor-support plate on the roll servo and the
  electronics sit on the face plate. The hip-yaw servos, neck mount and battery are now also held by
  explicit parts (hip plate, a neck mount standing on the floor, power support). Joint axes, usable
  ranges and standing are unchanged; mass +8 g for the added parts.
- SO-101 arm (`designs/so101.py`): six STS3215 servos, fixed base, gripper. Built in the reference's
  frame from measured joint positions: all six joint axes within 0.2 mm of the SO-ARM100 MJCF, same
  axis signs and ranges, fingertip site 0.5 mm off. Every printed part is an `asm.link`: mass 600 g vs
  632 g, and the mass beyond each joint (what the servos carry) within 3.5% of the reference; the
  elbow's usable range stops where the reference's does (2.36 rad, lower arm meets the shoulder).
  `scripts/compare_links.py` renders each body next to the reference's.
  Holding full extension with 200 g needs 49% of the shoulder-lift servo. It drove these API changes:
  servo specs without a rear boss (`STS3215`), `fasten(..., at=)`, `polygon` + `extrude` in any plane
  (replacing the XZ-only `profile_plate`; `extrude` no longer depends on outline winding), the arm
  check, and `scripts/compare_joints.py` for any reference MJCF.
- One way to build: all designs use assemblies; hand placement (`Link`, `Part`, placed servos,
  world-frame bracket functions) is no longer part of the API.
- Microduck ported to the same pieces, after a body-by-body comparison with Pollen's model: the foot
  is an open tub with the ankle fork rising out of it (`asm.link`, one-sided because the shin holds the
  servo's back; the ankle swings down inside the tub and the build carves the walls where it passes) in
  a TPU sole that wraps up the sides; the thigh is a cap round both servos (`tray`); bearings where
  Pollen's model has them (`bearing`, `asm.link(bearing=)`); the head gains a floored bottom shell,
  the soft mouth, a jaw servo (`asm.lock`: mounted, not a simulated joint) with the U-shaped jaw on its
  horn, and a cradle round the head-yaw servo carrying a bearing on the roll axis (`Hold` on a servo
  that is already placed checks the pocket against it). Mass 692 g vs 737 g (was 550 g): trunk -1%,
  legs -10%, head and neck -7%, part of it mass Pollen's head-yaw body carries beyond its listed parts. Fitting
  Pollen's body masses gives printed parts at 1.07 g/cm^3, so the design's `infill` defaults to 0.85
  (`robot.print_settings(infill=)` reprints any robot). Joint axes unchanged. Pollen's own model falls
  in the servo-only stand test (80 deg nose down), and so does ours now (79 deg): the old design stood
  only because its head was too light. Remaining: ankle down to -1.40 rad (the foot meets the thigh
  cap); legs 10% light (hip-yaw bracket and foot fork smaller than Pollen's).
- Robotiq 2F-85 (`designs/robotiq_2f85.py`), done as implement → critique → improve → reimplement
  ([docs/iterations/robotiq_2f85.md](docs/iterations/robotiq_2f85.md)). Round 1 on the old API
  (servo joints only, no loops) gave fingers whose pads tilted 46 deg and crossed. That drove
  `asm.hinge`, loop closures, couplings, joint springs, solved poses and the `attach` containment
  fix. Round 2 is the real five-bar per finger: all 8 joint axes on the Menagerie model's, pads
  within ~1 mm of its path and parallel within 2 deg, mass 1056 g vs 1053 g.
- Crazyflie 2.1 (`designs/crazyflie2.py`), implement → critique → improve → reimplement
  ([docs/iterations/crazyflie2.md](docs/iterations/crazyflie2.md)). Round 1 had the parts but could
  not fly (no thrust, no sensors) and crashed the stand test. That drove `asm.rotor`, `asm.imu`,
  the hover keyframe and `hover_test`. Round 2: 27.08 g vs 27.0 g, thrust-to-weight 2.26, levels
  from a 10 degree tilt in closed loop; inertia within 8% of measured values (Menagerie's
  roll and pitch inertia are not consistent with a flat quad).
- Unitree Go2 (`designs/go2.py`), implement → critique → improve → reimplement
  ([docs/iterations/go2.md](docs/iterations/go2.md)). Round 1 had flipped right-side hips, +33% mass,
  28 overlaps at home and only position servos. That drove motor specs and units, torque mode, remote
  drives, the `same` mirror convention, primitive collisions and contact settings, foot sensors,
  keyframes and an axis-based stand test. Round 2: all 12 axes on the Menagerie model's, 14.85 kg vs
  15.21 kg, feet within 2 mm, stands on its PD gains; torso roll inertia still 43% low.
  Covers experiment (`--set covers=True`): moulded torso, thigh and calf covers shaped to sections
  measured from the Menagerie meshes raise the outline overlap (`scripts/compare_looks.py`) from
  0.654 to 0.711, with every joint keeping its full range (`clear_motion`). The rest of the gap is
  our part layout: the covers grow to hold parts the real ones don't, and they are cut wherever
  our legs pass. Off by default (the covered build takes ~80 s against 1.5 s).
- Unitree G1 (`designs/g1.py`), implement → critique → improve → reimplement
  ([docs/iterations/g1.md](docs/iterations/g1.md)). Round 1 had wrong right-side roll ranges, a
  shoulder axis 32 degrees off and 32 overlaps at home. That drove `asm.chain`, mass budgets, the
  `closest` mirror convention, dampratio, sensor noise, the pin rule, and two fixes (floor placement of
  primitive shapes, a clean JSON stdout). Round 2: all 29 axes on the Menagerie model's, no overlaps,
  and the stand test matches the reference's (0.0 vs -0.1 degrees, 0.7 mm settle each).
- Next: Hello Robot Stretch 3 (the leader arm), one CLI, tests, packaging.
