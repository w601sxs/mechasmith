# Paper experiments

Run everything from the repository root. Builds first: `uv run mechasmith build <design>` for
every design (headless: `MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x1024x24" ...`).

## checker/: the independent checker (Q2) and what each check catches

`checker.py` scores any MJCF with MuJoCo plus exact mesh geometry (trimesh, manifold3d), and
nothing from mechasmith: loads, mass, floating parts, detached bodies (touching, or pinned by a
hinge whose axis passes through both), overlaps, joint errors against a reference
(Articulate-Anything's definitions plus axis sign) and an optional stand test.

```bash
uv run --with rtree python paper/experiments/checker/checker.py build/so101/robot.xml \
    --reference path/to/reference.xml --stand
uv run --with rtree python paper/experiments/checker/detect.py      # writes paper/generated/detect.json
```

`detect.py` injects faults (part moved, joint moved, axis tilted or flipped) into every exported
design and records which checks catch them: MuJoCo compiling, the standard joint metric, axis
sign, and the checker's geometry. CPU only; slow on the large designs (minutes per check).

## q3/: do authoring faults matter for learned policies?

`faults.py` edits the Microduck MJX training model (axis tilt or flip, joint shift, head CoM shift,
trunk mass scale, foot contact offset; 15 models); `run_q3.py` trains PPO on each (3 seeds, with
and without domain randomisation: 96 runs) and evaluates every policy on the model it was trained
on and on the correct one; `aggregate.py` writes `paper/generated/table_q3.tex`.

```bash
# check the whole path on CPU (about 5 minutes)
uv run --no-project --python 3.12 --with jax==0.7.2 --with brax==0.14.2 --with mujoco==3.17.0 \
    --with mujoco-mjx==3.17.0 python paper/experiments/q3/run_q3.py --smoke --index 3 --workdir /tmp/q3
# on a GPU machine: runs 0..95, one per call or in ranges; resumable
bash paper/experiments/q3/run_q3_job.sh 0 95
python paper/experiments/q3/aggregate.py
```

Cost: about 27 min per run at 3e8 steps on one H100 for the correct duck, so about 43 GPU-hours
for the grid (faulty models may train slower or faster; unmeasured).

Note: the MJX model collides robot geoms only with the floor, so links overlapping each other are
invisible to training. Overlaps are therefore a checker fault (detect.py), not a Q3 fault.
