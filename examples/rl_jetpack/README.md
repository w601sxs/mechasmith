# One policy to walk, hop, fly and land: the jet-pack Microduck

A single RL policy (Brax PPO on MJX) drives all 18 actuators of `designs/microduck_jetpack.py`:
the 14 duck servos, the 2 nacelle-tilt servos and the 2 ducted fans. Unlike the hand-written
flight controller in `examples/microduck_jetpack/fly.py`, nothing tells it which skill to use:
it is given a speed, a turn rate and a height for the trunk above the ground under it, and the
fans cost energy.

- Height = standing height: walk. If the ground ahead rises (a box), standing height means
  getting on top of it, so it has to step, hop (fan-assisted) or fly to follow the command.
- Height above standing: fly, hover; back to standing height: land.

1. Export the training model and copy it here:
   ```bash
   uv run mechasmith build designs/microduck_jetpack.py --sim mjx
   cp build/microduck_jetpack/robot_mjx.xml examples/rl_jetpack/jet_mjx.xml
   ```
2. Smoke test on CPU (a couple of minutes):
   ```bash
   cd examples/rl_jetpack
   PYTHONPATH=. uv run --no-project --python 3.12 --with jax==0.7.2 --with brax==0.14.2 --with mujoco==3.17.0 \
       --with mujoco-mjx==3.17.0 python train.py --smoke
   ```
3. Train on one GPU: `bash run_job.sh 600000000 8192` (same job script as `rl_microduck`).
4. Watch it: `python play.py policy.b64 --box 0.15 --out course.gif` (add `--with pillow`): walk to
   a box and past it, take off, fly forward, hover, land, walk.

## The task (`jet_env.py`)

Episodes are one of: on the ground with command segments switching between walking and flying
(so take-offs and landings happen at the switches); a 0.4 m box across the path with a
walk-forward command, whose height rises per simulated duck each time it gets past (3 cm to
30 cm) and drops when it falls; or an airborne start (fans at hover) told to fly or to land.

Observations: IMU, joint angles and velocities, last action, fan thrusts, command, gait clock,
plus body velocity, height above the ground and the box ahead (near and far edge, top). Those
three need sensors a real duck would add: an optical-flow / state estimate, and downward and
forward range sensors.

Actions: joint targets as offsets from home (rate-limited to the XL330's 10.8 rad/s) and each
fan's thrust as a fraction of 12 N, through a 60 ms spool-up lag (EDFs take ~0.05-0.1 s).
50 Hz was too coarse for flight, so the policy acts every 12 ms.

Rewards: track the speed, turn rate and height; when walking, the gait clock, air time, equal
steps and level hips from `rl_microduck`; when flying, legs and head held still; upright and
smooth everywhere; soft touchdowns; and fan energy (costly when walking, cheap when flying), so
it walks wherever walking works.
