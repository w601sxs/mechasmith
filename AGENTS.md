# Working with mechasmith (for coding agents)

mechasmith turns a robot written as one Python file into a MuJoCo model, a URDF and a JSON report
of checks. This file is how to use it well. The full API is in [docs/api.md](docs/api.md); working
designs are in [designs/](designs/) (also bundled: `mechasmith examples`).

## The loop

1. Start from the closest example, not a blank file: `mechasmith copy <example> mybot.py`.
2. `mechasmith describe mybot.py` lists the parameters, their ranges and the poses (JSON).
3. Edit the design, or override parameters: `mechasmith build mybot.py --set name=value`.
4. Read `build/<name>/report.json` (also printed on stdout; everything else goes to stderr).
5. Look at `build/<name>/views.png` and `pose_*.png` when the shape matters. Numbers alone
   don't show a part pointing the wrong way.
6. Repeat. A build takes seconds; `--no-sweep` skips the joint sweep when you only need the model.

What to read in the report:

| Key | Meaning | What you want |
|---|---|---|
| `mass.total_g`, per body | from the CAD solids, printed parts at the design's infill | near the real robot's |
| `interference.home_collisions` | parts overlapping at the home pose | empty |
| `interference.joints.<j>.usable` / `blocked` | the range each joint really has before parts collide, and what stops it | usable equals the declared range |
| `stand` (floating base) | holds `home` on servo PD alone for 3 s: `stands`, `settled_pitch_deg`, `worst_servo`, `pd_stiffness_ratio` | stands; ratio below 1 means it needs a balance controller |
| `hover` (has rotors) | trim thrusts and tilts, thrust-to-weight, closed-loop recovery | thrust-to-weight comfortably above 1 |
| `arm` (fixed base with `tool_site`) | servo load holding `home` plus `payload_kg`, sag, reach | worst load well below 1 |

## Rules

- **Never import build123d in a design.** Everything a design needs comes from `mechasmith`.
  (What comes back is build123d shapes, if something truly unusual needs them.)
- **Never position a part by hand.** A part enters the robot only through a connection:
  `asm.fasten`, `asm.attach`, `asm.joint`, `asm.link`, `asm.hinge`, `asm.chain`. That is what
  guarantees nothing floats. There is one way to build; don't look for a second.
- **An `AttachmentError` means the geometry is wrong.** Fix the numbers that put the parts apart.
  Don't pad shapes until they touch, and don't route around the check.
- **Declare every parameter with `param(default, range, doc)`.** The range is what lets you (and
  `--set`) explore safely; overrides outside it are rejected.
- **Build one side, then mirror it** (`robot.mirror("left_", "right_")`). Don't write both sides.
- Keep design files self-contained. A design may import another design (the jet pack imports
  `microduck`), but nothing else outside `mechasmith`.

## Conventions

- Millimetres and radians. One world frame at the home pose: x forward, y left, z up.
- `Component.world(...)` parts are drawn in that frame; generated parts (servos, brackets,
  spacers) have their own frames and are placed by their connectors.
- A connector's frame: z is the outward normal, x the reference direction. `fasten` puts two
  connectors face to face; `spin=(local_vec, world_dir)` fixes the rotation about the normal.
- Servo frame: +X is the output axis (horn side), +Z points to the short end of the case.
- Joint anchors, axes and every MJCF body frame are derived. Never write them.

## Which example to start from

| You are building | Start from | Why |
|---|---|---|
| a servo arm with printed links | `so101` | `asm.link` + `Hold`, fixed base, `tool_site`, arm test |
| a small servo biped or legged robot | `microduck` | shells, covers, bearings, mirroring, stand test |
| a quadruped or humanoid on motors | `go2`, `g1` | `motor(...)` from a datasheet, `asm.chain` from a joint table, torque control, foot sensors |
| a gripper or other linkage | `robotiq_2f85` | `asm.hinge`, closed loops, `robot.couple` |
| anything that flies | `crazyflie2`; `microduck_jetpack` for tilting rotors | `asm.rotor`, `asm.imu`, hover test |
| one limb to iterate on fast | `duck_leg` | small, builds in about a second |

## Rebuilding a real robot

The examples were made by matching a public reference model. The same loop works for a new robot:

1. Get the reference (a MuJoCo Menagerie MJCF, a vendor URDF converted to MJCF, a paper's numbers).
2. Make the joint axes match first. `scripts/compare_joints.py ref.xml build/<name>` reports
   each axis as a line and how far ours is from it.
3. Then masses per body (`report.json` against the reference's inertials), then the look
   (`scripts/compare_looks.py`, `scripts/compare_render.py`).
4. Run `implement → critique → improve → reimplement`: build a first version, write down what is
   wrong with it, change the library if the design needs something it can't express, and rebuild.
   [docs/iterations/](docs/iterations/) has these write-ups for the Robotiq, Crazyflie, Go2 and G1.

## Pitfalls already found

- `fasten(..., at=point)` takes a **world** point on the first connector's face, not an offset.
  It is checked to lie on that part's surface.
- `extrude(outline, plane, ...)` grows along the plane's normal whatever the outline's winding.
- A sweep reporting `blocked` usually means a parent part is too big near the joint. Pass the
  child's `swept_link(...)` to the parent's `clear=` instead of shrinking it by guesswork.
- Microduck falls in the stand test. That is expected: Pollen's own model does too at the real
  mass. Don't "fix" it by making the head lighter.
- `--sim mjx` writes `robot_mjx.xml` for training: primitive collisions, no meshes, floor only.
  Set `collision="box"` (or `capsule`, ...) on parts where the fitted box is a poor fit.

## Working on the library itself

- `uv sync`, then `uv run mechasmith build <every design>` is the regression check: masses,
  collisions and the stand/hover/arm results should not move unless you meant them to.
- The designs are the test suite. When an example needs something the API can't express, add it
  to the library and keep the design simple, rather than working around it in the design.
