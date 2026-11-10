# Crazyflie 2.1: implement → critique → improve → reimplement

Reference: MuJoCo Menagerie `bitcraze_crazyflie_2` (MIT): one rigid body, 27 g, a published
inertia, a body-thrust actuator and three body-moment actuators at one site, an IMU (gyro,
accelerometer, orientation), and a hover keyframe.

## Round 1: the library as it was ([crazyflie2_round1.py](crazyflie2_round1.py))

The parts went in fine (frame PCB, battery, deck, headers, motors in moulded mounts with legs,
props), but:
- no way to make thrust: no rotor or site actuators, so it cannot fly;
- no sensors exported, although the IMU site existed;
- the only floating-base check was the stand test, and it crashed on a robot with no joint
  actuators (`argmax` of an empty sequence);
- no hover keyframe, no check that the rotor layout can control all three axes;
- `compare_joints` compared joints and masses beyond joints, so a jointless robot got only a
  total mass.

Mass 29.1 g vs 27.0 g.

## Library changes

- `asm.rotor(name, part, at, axis, spin=±1, thrust_max, torque_ratio)`: a motor and propeller.
  Exported as an MJCF site actuator per rotor: thrust along the axis and the reaction torque
  `-spin * torque_ratio * thrust` about it, control range 0..thrust_max N.
- `asm.imu(site)`: gyro, accelerometer and orientation sensors at a site.
- A `hover` keyframe: the rotor thrusts that hold the robot still, from the rotor allocation.
- `sim.hover_test` replaces the stand test for rotorcraft: thrust-to-weight, the hover
  throttle of each rotor, whether thrust and all three moments are independently controllable
  (allocation rank), and a closed-loop flight from a 10 degree tilt through the allocation.
- The stand test skips robots with no joint actuators instead of crashing.
- `compare_joints.py --body ours=theirs` compares mass and inertia, and lists sensors.

## Round 2 ([designs/crazyflie2.py](../../designs/crazyflie2.py))

Four rotors (Bitcraze numbering and spin directions) at the prop hubs, IMU at the centre.

- Mass 27.08 g vs 27.00 g.
- Hover test: thrust-to-weight 2.26, 44% throttle per motor at hover, controllable; from a
  10 degree tilt it levels to 0.01 degrees, holds height, no yaw drift.
- Sensors: the same three as the reference.
- URDF matches the MJCF; rotor and IMU sites become frames.
- Inertia: Izz -8%, Ixx and Iyy -35% against Menagerie's (2.395e-5, 2.395e-5, 3.235e-5), but
  -6%, -8% and +2% against the measured values in Landry 2015 (1.657e-5, 1.666e-5, 2.926e-5).
  For a flat X-quad, Ixx is about Izz/2 + sum(m z^2): Menagerie's numbers would need the mass
  17 mm (rms) off the rotor plane on a craft about 30 mm tall; the measured ones imply 8.5 mm,
  ours 4.8 mm. The measured values are the consistent reference.

## Still open
- The actuators are per rotor; Menagerie's are body thrust plus three moments. An RL setup
  written for one needs a mixer for the other.
- Thrust is linear in the control; no motor dynamics, no propeller drag or gyroscopic effect.
- Thrust per motor and the torque ratio come from published figures marked (verify).
- URDF has no thrust: rotors are frames only.
- Pin headers are solid blocks, not rows of pins.
