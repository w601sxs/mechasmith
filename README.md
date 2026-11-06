# mechasmith

Agent-first robot design: CAD as Python code → computed mass properties → MJCF → checks → renders, in one command.

```bash
uv run mechasmith build designs/microduck.py    # Pollen's Microduck, reverse-engineered as code-CAD
uv run python scripts/compare_render.py <reference scene.xml> build/microduck designs/microduck.py compare.png
uv run python benchmarks/compare_microduck.py build/microduck   # axis-line placement and masses vs Microduck
uv run python scripts/sweep_balance.py designs/microduck.py battery_g 40,70,100   # CoM vs ankles per value
uv run python scripts/balance.py build/microduck       # where the mass sits relative to the feet
uv run mechasmith build designs/duck_leg.py     # writes build/duck_leg/{scene.xml, robot.xml, meshes/, views.png, report.json}
uv run python benchmarks/compare_leg.py build/duck_leg   # compare against Microduck's reference numbers
uv run python scripts/sweep_param.py designs/duck_leg.py foot_reach left_ankle 12,14,16
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
- `fasten(a, b, spin=...)` puts b's connector face flat on a's (they touch by construction).
  Servos, U-brackets, horn plates, spacers and L-brackets come with connectors.
- `attach(base, part)` is for parts drawn in world coordinates (shells, covers). Contact is checked
  on that line; a gap raises `AttachmentError: head does not touch neck_pitch_case: the gap is 7.61 mm`.
- Parts are placed as you connect them, root outward, so later parts can use earlier positions
  (`servo.anchor`, `servo.face("face")`). Connecting a placed part again must agree with where it is.
- `servo(...).case.face(BACK)` picks whichever connector currently faces a world direction.
- `loft_cavity`, `rounded_box_cavity`: intersect a plate with a shell's cavity so it reaches the
  walls exactly (how the Microduck head motor support and hip plate are made).
- `asm.robot()` groups rigidly connected parts into links; `joint` splits them.
  `robot.mirror("left_", "right_")` then makes the other side.

Parameters are declared with ranges, so `mechasmith describe` (and any agent) can list the knobs,
and every tool checks overrides against the ranges:

```bash
uv run mechasmith describe designs/microduck.py
uv run mechasmith build designs/microduck.py --set battery_g=80 --set shin_len=45
```

Geometry helpers, all from `mechasmith`: `UP DOWN FWD BACK LEFT RIGHT`, `box`, `cyl`, `rect`, `slot`,
`circle`, `extrude(outline, plane, at, center, depth)`, `move`, `color("#ff661f")`, and shells
(`loft_shell`, `arch`, `profile_plate`, `rounded_box_shell`). Designs need no build123d import;
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
- Mass and inertia come from the CAD solids, with a 3D-printing infill model for printed parts.
- Exact mesh-boolean interference at the home pose, then a sweep of every joint reporting its usable range and the first parts that collide.
- Contact-sheet renders of the home pose and any named `POSES`.
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
  (lofted sections, optionally arched), `arch`, `profile_plate`.
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
- One way to build: both designs use assemblies; hand placement (`Link`, `Part`, placed servos,
  world-frame bracket functions) is no longer part of the API.
- Next: one `mechasmith` CLI for scripts and benchmarks, tests, packaging.
