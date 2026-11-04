# mechasmith

Agent-first robot design: CAD as Python code → computed mass properties → MJCF → checks → renders, in one command.

```bash
uv run mechasmith build designs/duck_leg.py     # writes build/duck_leg/{scene.xml, robot.xml, meshes/, views.png, report.json}
uv run python benchmarks/compare_leg.py build/duck_leg   # compare against Microduck's reference numbers
uv run python scripts/sweep_param.py designs/duck_leg.py foot_reach left_ankle 12,14,16
```

## Conventions
- Author everything in one world frame at the home pose: millimetres, x forward, y left, z up.
- A joint is an anchor point and an axis in that frame; body frames are derived, never hand-written.
- Servos are components carrying geometry, mass and a fitted actuator model (`components/servo.py`).
- Brackets are generated from a servo's own frame (`brackets.py`), so they follow it when it moves.

## What `build` checks
- MJCF compiles, and the scene reloads from disk.
- Mass and inertia come from the CAD solids, with a 3D-printing infill model for printed parts.
- Exact mesh-boolean interference at the home pose, then a sweep of every joint reporting its usable range and the first parts that collide.
- Contact-sheet renders of the home pose and any named `POSES`.

## Status
Day 1: a 5-DOF duck-class leg (XL330 servos) with collision-free full joint ranges, compared against Pollen's Microduck.
Next: mirrored legs, trunk, neck and head (14 DOF), standing in simulation, browser viewer.
