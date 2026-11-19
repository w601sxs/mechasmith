# Teaching the mechasmith Microduck to walk

The simplest RL loop on a mechasmith robot: MJX (MuJoCo on the GPU, through JAX) and Brax PPO.

1. Export the training model and copy it here:
   ```bash
   uv run mechasmith build designs/microduck.py --sim mjx
   cp build/microduck/robot_mjx.xml examples/rl_microduck/duck_mjx.xml
   ```
2. Check that everything wires up, on CPU (about a minute):
   ```bash
   cd examples/rl_microduck
   uv run --no-project --python 3.12 --with jax==0.7.2 --with brax==0.14.2 --with mujoco==3.17.0 \
       --with mujoco-mjx==3.17.0 python train.py --smoke
   ```
3. Train on one GPU: `bash run_job.sh 150000000 8192` installs JAX/Brax/MJX into a fresh
   virtualenv and runs `train.py` (300 M steps took 25 minutes on one H100). On Scale Train it
   reports each step, the evaluation curve and the policy (in chunks) through `train_report`.
4. Watch it: join the policy chunks into one base64 file and run
   ```bash
   uv run --no-project --python 3.12 --with jax==0.7.2 --with brax==0.14.2 --with mujoco==3.17.0 \
       --with mujoco-mjx==3.17.0 --with pillow python play.py policy.b64 --out walk.gif
   ```
   It drives the policy through a script of commands, prints commanded against measured speeds,
   and replays the motion on the full mechasmith model.

The task (`duck_env.py`): the policy sees the IMU, joint angles and velocities, its last
action and the command (forward and sideways speed, turn rate), and sets joint targets around the
home pose 50 times a second; rewards follow the command, keep it upright and level, reward steps
with the feet in the air and keep the head still; random pushes make it robust.

Results (300 M steps, 27 minutes on one H100), measured by `play.py`:

| command | asked | measured |
|---|---|---|
| forward | 0.15 m/s | 0.147 m/s |
| forward, fast | 0.30 m/s | 0.291 m/s |
| forward and turn | 0.10 m/s, 0.6 rad/s | 0.117 m/s, 0.49 rad/s |
| backward | -0.08 m/s | -0.058 m/s (and turns 0.2 rad/s it wasn't asked to) |
| sideways | 0.08 m/s | 0.068 m/s |

It never fell. Each foot carries the duck alone for the same time (160 and 162 ms at 0.3 m/s, two
steps of a 2.7 Hz gait clock), with short double support and no flight: a fast walk, not a run.

What it took (four runs): run 1 stepped in place (the step reward paid more than moving); run 2
walked but limped, with a wrong foot-contact test; run 3 (gait clock, real contacts, speed ramp)
still limped and topped out at 0.16 m/s because its +-0.3 rad action range capped the stride and
its ramp rose on average tracking rather than the top speed. Run 4 doubled the action range,
raised the top speed only after the duck reached 85% of it, and penalised unequal single-foot times.
The job also reports the policy at every evaluation (`snapshot_chunk`), so `play.py` can show it mid-run.
