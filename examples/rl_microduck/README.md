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
   virtualenv and runs `train.py` (150 M steps took 15 minutes on one H100). On Scale Train it
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
