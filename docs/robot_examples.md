# Next robots for mechasmith

Research notes for choosing example robots after Microduck. The goal: each new example should
(a) be a robot people actually build or simulate, (b) have a public reference model so we can
benchmark the rebuild the way we did Microduck (joint axis lines, masses, side-by-side renders),
and (c) force one or two new library capabilities, so the library improves with every example.

Research date: 2026-11-05. Sources at the end. Claims marked *(verify)* were not confirmed
from a primary source.

---

## 1. What Isaac Sim ships

Isaac Sim 6.2's robot catalog (Content Browser, `Isaac Sim/Robots`) groups robots by form factor.
Summary of what it lists:

| Category | Count (approx.) | Representative models |
|---|---|---|
| Arms | ~80 listed, plus ~84 more FANUC | Franka Panda / FR3, UR3e–UR30, Kinova Gen3 and Jaco2, xArm6/7, Lite6, UF850, Unitree Z1, Sawyer, KUKA KR210, Yaskawa Motoman, Fanuc CRX/LR Mate, Kawasaki RS, Denso Cobotta Pro, Techman TM12, Flexiv Rizon4, Mecademic Meca500, **RobotStudio SO-100 and SO-101**, OpenArm (uni/bimanual), I2RT YAM, Yahboom Dofbot |
| Grippers | ~15 | Robotiq 2F-85, 2F-140, Hand-E; Schunk EGK/EGU/EZU; xArm and Lite6 grippers |
| Hands | ~10 | Allegro (16 DOF), Shadow Hand (24), Schunk SVH (24), Psyonic Ability (11), Unitree Dex3/Dex5 |
| Humanoids | 17 | Unitree H1, G1; Fourier GR-1; Agility Digit and Cassie; Booster T1; 1X Neo; Agibot A2D; RobotEra STAR1; Sanctuary Phoenix; Valkyrie; Tien Kung; XiaoPeng PX5 |
| Quadrupeds | 18 | Unitree A1, Go1, Go2, B2, Aliengo, Laikago; ANYmal B/C/D; Spot; DeepRobotics X30, Lite3, M20; Addverb Trakr; the "Ant" reference robot |
| Wheeled | ~20 | TurtleBot3 Burger, Jetbot, iRobot Create3, Carter, Nova Carter, Leatherback, Clearpath Jackal and Dingo, AgileX Limo, forklifts |
| Holonomic | 2 | NVIDIA Kaya, Fraunhofer O3dyn |
| Aerial | 4 | Crazyflie, a generic quadcopter, NASA Ingenuity, NTNU ARL-Robot-1 |

The pattern: Isaac Sim is dominated by commercial arms, plus RL locomotion favourites
(Unitree, ANYmal, Spot) and a few open designs (SO-100/101, OpenArm). Isaac Sim imports URDF and
MJCF *(verify the MJCF importer covers equality constraints and tendons)*, so a mechasmith URDF or
USD export would let our designs drop straight into it.

## 2. Where reference MJCFs exist (MuJoCo Menagerie)

Menagerie has 64 curated MJCF models. These are the cheapest benchmark targets because the
model is already MuJoCo-native, like Microduck's. Relevant ones:

- **Low-cost / open arms:** `robotstudio_so101`, `trs_so_arm100`, `low_cost_robot_arm`, `trossen_wx250s`,
  `trossen_vx300s`, `trossen_wxai`, `agilex_piper`, `arx_l5`, `i2rt_yam`, `seeed_rebot_devarm`,
  `dynamixel_2r`, `robotis_omy_3m`, `aloha` (dual arm)
- **Industrial / research arms:** `franka_emika_panda`, `franka_fr3`, `universal_robots_ur5e`, `ur10e`,
  `kinova_gen3`, `kuka_iiwa_14`, `ufactory_xarm7`, `ufactory_lite6`, `unitree_z1`, `rethink_robotics_sawyer`
- **Grippers and hands:** `robotiq_2f85` (and v4), `leap_hand`, `wonik_allegro`, `shadow_hand`,
  `umi_gripper`, `tetheria_aero_hand_open`
- **Humanoids:** `toddlerbot_2xc`, `toddlerbot_2xm`, `robotis_op3`, `berkeley_humanoid`, `booster_t1`,
  `unitree_g1`, `unitree_h1`
- **Quadrupeds:** `unitree_go2`, `unitree_go1`, `unitree_a1`, `anybotics_anymal_c`, `google_barkour_vb`,
  `boston_dynamics_spot`
- **Other:** `bitcraze_crazyflie_2`, `skydio_x2`, `robot_soccer_kit` (omni-wheel base),
  `hello_robot_stretch_3`, `stanford_tidybot`

Overlap with Isaac Sim, so one rebuild would serve both simulators: SO-101, Franka, UR5e, Kinova
Gen3, xArm7, Lite6, Unitree Z1, Allegro, Shadow, Robotiq 2F-85, Go1/Go2/A1, ANYmal C, Spot, G1,
H1, Booster T1, Cassie, Crazyflie.

## 3. Recommended examples, in order

Each entry: why it's worth doing, the reference to benchmark against, and the library work it
forces. Ordered so each step adds a small number of new capabilities.

### 1. SO-101 arm (The Robot Studio / Hugging Face LeRobot) — start here
- **Why:** the most-built open arm right now; the default LeRobot hardware; in Isaac Sim *and*
  Menagerie. Shows mechasmith is not only for legged robots.
- **Hardware:** 6 Feetech STS3215 bus servos (5 arm joints + gripper jaw); 3D-printed parts.
  Follower uses 1/345 gearing on all joints; the leader arm mixes 1/345, 1/191 and 1/147.
- **Reference:** `TheRobotStudio/SO-ARM100` (Apache-2.0) publishes STEP, STL, URDF and an SO-101
  MJCF in `Simulation/SO101/`; also Menagerie `robotstudio_so101`.
- **Library work:**
  - a second servo spec, `STS3215`, plus gear-ratio variants (a servo *family*, not one constant);
  - fixed-base robot with an end-effector site;
  - a single actuated gripper jaw;
  - new checks: reach/workspace envelope and payload torque at full extension (the arm equivalent
    of the stand test).
- **Benchmark:** axis-line offsets and link masses vs the SO-101 MJCF; leader/follower variants as
  two parameter sets of one design.

### 2. Koch / "Low-Cost Robot Arm" (Dynamixel)
- **Why:** the other common LeRobot arm; uses Dynamixel XL330/XL430, so it reuses our XL330.
- **Reference:** Menagerie `low_cost_robot_arm`.
- **Library work:** add `XL430`. Cheap second arm that tests whether the SO-101 API generalises.

### 3. Robotiq 2F-85 gripper
- **Why:** the most common research gripper, and a **closed-chain linkage** (four-bar fingers
  driven by one motor).
- **Reference:** Menagerie `robotiq_2f85`; Isaac Sim 2F-85.
- **Library work:**
  - loops in the connector graph: a second connection closes a loop, so export it as an MJCF
    `equality` constraint (connect/weld) instead of rejecting it;
  - passive joints (no actuator), coupled joints/tendons;
  - the loop-consistency check from the connector design becomes a real feature.

### 4. WidowX 250 / ViperX 300 (Trossen Interbotix)
- **Why:** widely used research arms (ALOHA uses ViperX/WidowX); larger Dynamixels; **prismatic**
  gripper fingers.
- **Reference:** Menagerie `trossen_wx250s`, `trossen_vx300s`, `aloha`.
- **Library work:** prismatic joints and linear actuators; XM430/XM540 servo specs; tube or
  profile links; ALOHA as a two-arm scene (multiple robots in one export).

### 5. LEAP Hand (CMU)
- **Why:** open, cheap (Dynamixel) 16-DOF hand; dexterous manipulation is a big RL use case.
- **Reference:** Menagerie `leap_hand`; the project open-sources the URDF, assembly instructions
  and API.
- **Library work:** **sub-assemblies with instancing**: build one finger module, place it four times
  plus a thumb variant. This is the connector API's real test: a sub-assembly exposes connectors
  of its own.

### 6. Open Duck Mini v2 (apirrone)
- **Why:** Microduck's sibling (the open BDX-style duck). Same topology, different proportions.
- **Reference:** `apirrone/Open_Duck_Mini` (CAD, BOM, actuator identification with BAM) and
  `apirrone/Open_Duck_Playground` for the MuJoCo/RL side *(verify the MJCF path and servo model)*.
- **Library work:** **design families**: one biped template, two parameter sets. Shows that the
  Microduck design generalises; good regression test for the connector refactor.

### 7. Quadruped: Unitree Go2 as the benchmark, Stanford Pupper as the open build
- **Why:** quadrupeds are the most-trained RL platform in Isaac Lab. Go2 is in Isaac Sim and
  Menagerie; Pupper is the open, buildable one (CAD and BOM published; MJCF not confirmed).
- **Reference:** Menagerie `unitree_go2`; Pupper's published CAD *(verify current repo: the Doggo
  repo says support moved to Pupper v3)*.
- **Library work:**
  - **two-plane symmetry** (mirror across XZ *and* YZ: one leg built, four placed);
  - a leg as a reusable sub-assembly;
  - custom actuator modules (quasi-direct-drive motors, not hobby servos) with their own actuator
    model.

### 8. ToddlerBot 2XC (Stanford)
- **Why:** fully open, 3D-printed, Dynamixel humanoid with arms; designed in Onshape and exported
  to URDF/MJCF.
- **Reference:** Menagerie `toddlerbot_2xc` / `2xm`; project docs.
- **Library work:** a full humanoid (arms + legs + head), many servos, sub-assemblies reused
  across limbs, possibly parallel linkages *(verify)*.

### 9. Berkeley Humanoid Lite
- **Why:** open humanoid built around **3D-printed cycloidal gearboxes**; repo ships URDF, MJCF and
  USD plus an Onshape export script.
- **Reference:** `HybridRobotics/Berkeley-Humanoid-Lite`.
- **Library work:** a gearbox/actuator as a designed component (motor + printed reduction ->
  `ActuatorModel` derived from gear ratio and motor constants), bigger scale, USD export check.

### 10. Wheeled base: TurtleBot3 Burger, Jetbot, or the omni "robot soccer kit"
- **Why:** smallest new-feature example; wheeled bases are a whole Isaac Sim category.
- **Reference:** Isaac Sim TurtleBot3 / Jetbot; Menagerie `robot_soccer_kit` (omni wheels).
- **Library work:** continuous joints, **velocity** actuators, caster/omni wheels (passive rollers),
  a drive check (straight-line and turn-in-place in sim).

### 11. Crazyflie 2 (Bitcraze)
- **Why:** open micro-drone, in both Isaac Sim and Menagerie; for drones, mass and inertia accuracy
  is everything, which is exactly what our CAD-derived inertia gives.
- **Library work:** **thrust actuators** at sites (no joints), propeller guards, a hover test.

### 12. Mobile manipulator: SO-101 on a wheeled base (LeKiwi / XLeRobot)
- **Why:** combines 1 and 10; LeRobot's mobile variants *(verify current names and repos)*.
- **Library work:** composing designs: an arm design mounted on a base design via connectors.

### Also worth considering
- **Industrial/research arms (UR5e, Franka FR3, Kinova Gen3):** not buildable, but excellent
  kinematic references (DH parameters known exactly), 7-DOF redundancy, cylindrical link shells.
  Good for proving "matches to 0.1 mm" claims.
- **OpenArm (unimanual/bimanual):** open arm already in Isaac Sim; check its license and model files.
- **Robotis OP3:** classic Dynamixel humanoid in Menagerie; an easier humanoid than ToddlerBot.

---

## 4. Library capabilities the examples force

| Capability | Needed by |
|---|---|
| Connector-based assembly (in progress) | all |
| Servo/actuator catalog: STS3215 (+ gear variants), XL430, XM430, XM540, custom gearbox actuators | 1, 2, 4, 7, 9 |
| Fixed-base robots, end-effector sites, reach and payload checks | 1, 2, 4 |
| Prismatic joints and linear actuators | 4, gripper variants |
| Closed chains -> MJCF equality constraints; passive and coupled joints | 3, 8 (if parallel ankles) |
| Sub-assemblies with their own connectors; instancing | 5, 7, 8, 12 |
| Symmetry across more than one plane | 7 |
| Continuous joints and velocity actuators; passive rollers | 10, 12 |
| Thrust actuators at sites | 11 |
| Design families (one template, several parameter sets) | 1 (leader/follower), 6 |
| Generic reference benchmark: any MJCF + joint-name map -> axis-line, mass and render comparison | all (generalise `benchmarks/compare_microduck.py`) |
| URDF and USD export (for Isaac Sim / Isaac Lab), checked by re-import | all; 9 has a USD reference |
| Multiple robots per scene | 4 (ALOHA), 12 |
| Sensor sites (camera, IMU) exported as MJCF sensors | 7, 8, 10, 11 |

## 5. Suggested iteration plan

1. **SO-101** on the connector API. Done when it builds from parameters, matches the reference
   MJCF's axis lines within 1 mm and its mass within 10%, passes the reach/payload check, and
   renders side by side with the reference.
2. **Generic benchmark tool** (from step 1): `mechasmith compare --ref <scene.xml> --map joints.json`.
3. **Robotiq 2F-85**: closed chains and equality constraints.
4. **LEAP Hand**: sub-assemblies and instancing.
5. **Go2-class quadruped**: two-plane symmetry and custom actuators; first Isaac Lab handoff via
   URDF/USD export.
6. **Wheeled base, then SO-101 on a base**: continuous joints, velocity control, design composition.
7. Humanoids (ToddlerBot, Berkeley Humanoid Lite) once sub-assemblies and custom actuators are solid.

## 6. Licensing note

Rebuild from *measurements* (dimensions, masses, colours, joint placement), as with Microduck,
and don't copy reference meshes into the repo. Menagerie models carry per-model licenses, and open
hardware repos vary (SO-ARM100 is Apache-2.0). Check each reference's license before publishing
derived numbers alongside an example.

---

## Sources

- Isaac Sim robot assets: [overview (6.2)](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots.html),
  [overview (6.1)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_robots.html),
  [wheeled](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_wheeled.html),
  [manipulators](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_manipulator.html),
  [humanoids](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_humanoid.html),
  [quadrupeds](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_quadruped.html),
  [holonomic](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_holonomic.html),
  [aerial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_robots_aerial.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- SO-101: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (STEP, STL, URDF, MJCF;
  Apache-2.0), [SO-101 specs](https://www.roboticscenter.ai/en/hardware/so-101/specs)
- [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)
- [Stanford Pupper](https://stanfordstudentrobotics.org/pupper), [Pupper paper](https://arxiv.org/abs/2110.00736v2)
- Open Duck Mini: [GitHub mirror of the apirrone repo](https://github.com/jofcodeusal/Open_Duck_Mini)
- LEAP Hand: [paper](https://arxiv.org/pdf/2309.06440v1.pdf), [API](https://github.com/mohammad200h/LEAP_Hand_API)
- ToddlerBot: [paper](https://arxiv.org/html/2502.00893v3), [tools docs](https://hshi74.github.io/toddlerbot/sections/04_useful_tools.html)
