# Development history

How mechasmith got here, one robot at a time. Each example drove library changes; the
per-robot critique rounds are in [iterations/](iterations/).

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
- Microduck jet pack (`designs/microduck_jetpack.py`, `examples/microduck_jetpack/`): the Microduck
  with a backpack and two tilting 70 mm ducted fans on XL330 servos, flown as a tilt-birotor by a
  hand-written controller (take-off, hover, forward flight, turn, landing; GIF in the example).
  1248 g, thrust-to-weight 1.96 with 12 N fans (an estimate; ~1 min on 1300 mAh). It drove thrust
  vectoring in the library: `mjcf.hover_trim` solves rotor thrusts and the tilt-servo targets
  together and gives the allocation Jacobian, used by the `hover` key, the keyframes of rotorcraft
  with joints and the build's closed-loop hover check. Assumptions and gaps in the example's README.
