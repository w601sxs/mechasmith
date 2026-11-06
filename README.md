# mechasmith

Agent-first robot design: CAD as Python code → computed mass properties → MJCF → checks → renders, in one command.

```bash
uv run mechasmith build designs/microduck.py    # Pollen's Microduck, reverse-engineered as code-CAD
uv run python scripts/compare_render.py <reference scene.xml> build/microduck designs/microduck.py compare.png
uv run mechasmith build designs/duck.py         # full 14-DOF duck: meshes, MJCF, sweeps, stand test, renders, viewer.html
uv run python benchmarks/compare_duck.py build/duck     # axis-line placement and masses vs Microduck
uv run python scripts/sweep_balance.py designs/duck.py battery_x 20,32,44   # CoM vs ankles per parameter value
uv run python scripts/balance.py build/duck            # where the mass sits relative to the feet
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
- Floating-base robots get a stand test: hold `HOME` with the servos alone for 3 s, then report whether it
  stays up, where it settles, per-servo torque load, the support margin, and whether servo stiffness alone
  could hold the pose upright (`pd_stiffness_ratio`; below 1 means a balance controller is required).
- `viewer.html`: a self-contained browser viewer with a slider per joint, the design's poses, and the
  measured collision-free ranges.

## Connectors
`brackets.py` generates parts from servo frames: `u_bracket`, `horn_plate`, `l_bracket` (two perpendicular
faces), `strap` (coplanar faces), `spacer` (facing faces) and `cover_plate` (ties several cases together).
Servos can be mounted case-on-parent (`Servo.mount_on`) or horn-on-parent (`Servo.horn_on`, the
"inverted" mounting Microduck uses to cluster servos on one link).

## Status
- Day 1: a 5-DOF duck-class leg (XL330 servos) with collision-free full joint ranges.
- Day 2: `designs/duck.py`, a full 14-DOF duck with Microduck's joint names and axis signs. Every leg axis
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
  (lofted sections, optionally arched), `arch`, `profile_plate`.
- Known gaps: 543 g vs 737 g; round hip-roll brackets and the fixed jaw servo are not modelled.
- Next: RL training in microduck_rl with this MJCF.
