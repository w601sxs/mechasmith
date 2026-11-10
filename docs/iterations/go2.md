# Unitree Go2: implement → critique → improve → reimplement

Reference: MuJoCo Menagerie `unitree_go2`: 15.2 kg, 12 torque-controlled joints (±23.7 N·m hip
and thigh, ±45.43 N·m knee), damping 2, armature 0.01, friction 0.2; primitive collision geoms
with tuned foot contact; a `home` keyframe. Unitree's axis convention: both sides share the
same world axes.

## Round 1: the library as it was ([go2_round1.py](go2_round1.py))

| problem | what happened |
|---|---|
| mirroring | the right hip abduction joints came out flipped: neither `vector` nor `pseudo` is Unitree's convention |
| motors | hand-drawn cylinders with hand-set masses; solid aluminium housings made the legs +33% heavy |
| collisions | 28 overlaps at home: no motor with a stator on one link and an output flange on the other |
| actuators | position servos only; no torque motors, nothing derived from gear ratio and rotor inertia |
| stand test | its stiffness check matched joint names (`hip_pitch`, `knee`, `ankle`) and reported 0.0 |
| missing | primitive collision shapes, foot contact settings, velocity and foot-contact sensors, a home keyframe |
| comparison | a 3 mm axis offset everywhere, only because the bases sat at different heights |

## Library changes

- `MotorSpec` (datasheet: size, mass, gear, rotor inertia, peak torque, speed) and
  `motor(name, spec, reduction=)`: a unit with a stator `case` and an output `horn` flange, used
  by `asm.joint` like a servo. Armature = rotor inertia × (gear × reduction)², torque and speed
  limits follow. `GO_M8010_6` is the first spec.
- `motor.drive(reduction)` for remotely driven joints: lock the motor on a link (`asm.lock`) and
  drive another joint through a belt or rod (the Go2's knee).
- `ActuatorModel.mode = "torque"`: an MJCF torque motor (ctrl in N·m); kp/kv become the suggested
  PD gains, written to the manifest.
- `Robot.mirror(axes="same")`: both sides keep the same world axes (Unitree, many ROS robots).
  (Generalised in the G1 round to `closest`, which `same` now means.)
  `mirror` now also copies sensors and rotors.
- Per-part `collision="box" | "sphere" | "cylinder" | "capsule"` (fitted primitives) and
  `contact=dict(friction=, condim=, solimp=, priority=)`.
- `asm.imu(site, kinds)` takes body position and velocities too; `asm.touch(site, radius)` adds a
  contact sensor.
- Every build exports keyframes for `home` and the named poses (set down on the floor).
- Stand test: legs are the chains from the root to bodies near the floor, pitch joints are found
  by axis (not name), and torque motors are run with their PD gains from the manifest.
- `compare_joints.py` aligns floating robots root to root, and `--body` reports where the mass
  sits (rms distance from the centre of mass along each axis).

## Round 2 ([designs/go2.py](../../designs/go2.py))

- All 12 joint axes on the reference's (0.0 mm), signs and ranges equal.
- Mass 14.85 kg vs 15.21 kg (−2%); mass beyond every joint within 6%. Feet within 2 mm.
- Torque motors: ±23.7 and ±45.43 N·m, armature 0.01 (0.037 at the knee, reflected through its
  1.917 reduction; Menagerie uses 0.01). Ten sensors, primitive collisions, `home` keyframe.
- No collisions at home. The hips' swing is carved out of the torso. What still limits a joint
  is legs meeting legs: the two feet under the body at −0.96 rad of inward abduction, a rear
  foot reaching a front calf.
- Stands on its PD gains (stiffness ratio 1.51, settles at −8°; it sinks ~67 mm as the low Isaac
  Lab gains let the knees give. An earlier "100 mm drop" was a floor-placement bug fixed in the G1 round).
- URDF matches the MJCF in MuJoCo and yourdfpy.

## Still open
- Torso roll inertia −43% (pitch +3%, yaw −4%). The reference's mass sits 49 mm (rms) out from
  the centre line, ours 34 mm: the Go2's covers are wider than its collision box; matching it
  would mean guessing the internal layout. Widening our torso cost abduction range.
- The knee's rod is represented by `drive()`, not drawn or simulated as a linkage.
- Looks: no white covers; motor cases are exposed.
- Motor data marked (verify): GO-M8010-6 size and rotor inertia.
