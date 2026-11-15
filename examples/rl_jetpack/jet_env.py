"""One policy for the jet-pack Microduck: walk, hop, fly, hover and land (MJX, Brax pipeline env).

The policy drives all 18 actuators: the 14 duck servos and the 2 nacelle-tilt servos (position
targets as offsets from home) and the 2 ducted fans (thrust, 0..12 N each, through a first-order
spool-up lag). One decision every 12 ms (3 physics steps of 4 ms).

Commands: forward / sideways speed, turn rate, and a height for the trunk above the ground
directly beneath it. At standing height that means walk; above it, fly; and the ground beneath
includes the top of an obstacle, so "walk on at standing height" past a box means getting onto it
(or over it). Nothing says how: the fans cost energy, so walking wins where walking works.

Episodes (each is one of):
  * ground, no obstacle: command segments switch between walking and flying (take-off and
    landing happen at the switches);
  * obstacle: a 0.4 m long box across the path, walk forward at standing height; its height
    rises per simulated duck as it gets past (curriculum, 3 cm -> 30 cm), falls when it falls;
  * airborne start: dropped in mid-air with the fans at hover, commanded to fly or to land.

Observations: IMU (body rates, gravity direction), joint angles and velocities, last action, fan
thrusts, command, gait clock, and two things a real duck would need extra sensors for: body
velocity (an optical-flow / state estimate) and the obstacle ahead and height above ground (a
downward and a forward range sensor).
"""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# `mechasmith build designs/microduck_jetpack.py --sim mjx`, then copy robot_mjx.xml here.
XML = Path(__file__).with_name("jet_mjx.xml")
FANS = ["jet_left_fan", "jet_right_fan"]
LEGS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
HIP_ROLL, HIP_YAW = ["left_hip_roll", "right_hip_roll"], ["left_hip_yaw", "right_hip_yaw"]
SOLES = ["left_sole_collision", "right_sole_collision"]
BOX_HALF = (0.2, 0.6, 0.2)          # obstacle half sizes, m (0.4 long, 1.2 wide); its height is set by sinking it

REWARDS = dict(tracking_lin_vel=4.0, tracking_ang_vel=1.5, height=3.0, gait_phase=2.0, feet_air_time=0.5,
               symmetry=-5.0, lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-5.0, hip_splay=-2.0,
               legs_home=-1.0, torques=-0.002, action_rate=-0.01, stand_still=-0.5, head_still=-5.0,
               fan_power=-1.0, landing=-2.0, termination=-1.0)
FAN_COST = (2.0, 0.3)               # fan_power multiplier when walking, when flying
CMD_ACCEL, H_RATE = 0.5, 0.4        # command easing: m/s^2 and rad/s^2; m/s for the height
SERVO_SPEED = 10.8                  # rad/s, XL330 no-load speed
SPOOL_TAU = 0.06                    # s, fan spool-up / down time constant (EDF ~0.05-0.1 s)
SEGMENT, EPISODE = 400, 1500        # control steps (4.8 s, 18 s)
OBST_START, OBST_STEP, OBST_TOP = 0.03, 0.02, 0.30
P_OBSTACLE, P_AIR = 0.3, 0.2


def obstacle_spec(xml=XML):
    """The training model plus the obstacle: a mocap box the robot collides with like the floor."""
    spec = mujoco.MjSpec.from_file(str(xml))
    b = spec.worldbody.add_body(name="obstacle", mocap=True, pos=[50.0, 0, -BOX_HALF[2]])
    b.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF), contype=0, conaffinity=1,
               rgba=[0.55, 0.6, 0.7, 1])
    return spec


class JetDuck(PipelineEnv):
    def __init__(self, xml=XML, action_scale=0.6, kick_vel=0.05, n_frames=3, **kw):
        m = obstacle_spec(xml).compile()
        sys = mjcf.load_model(m)
        super().__init__(sys, backend="mjx", n_frames=n_frames, **kw)
        self._home = jp.array(m.keyframe("home").qpos)
        self._h0 = float(m.keyframe("home").qpos[2])                         # trunk height standing
        jact = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT]
        self._fan_act = jp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in FANS])
        self._joint_act = jp.array(jact)
        self._jq = jp.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] - 7 for i in jact])   # into q[7:]
        self._default = self._home[7:]
        self._lo = jp.array([m.jnt_range[m.actuator_trnid[i, 0], 0] for i in jact])
        self._hi = jp.array([m.jnt_range[m.actuator_trnid[i, 0], 1] for i in jact])
        self._tmax = jp.array([m.actuator_ctrlrange[i, 1] for i in self._fan_act.tolist()])
        self._hover = float(m.body_subtreemass[1] * 9.81 / 2) / self._tmax  # thrust fraction per fan
        jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] - 7
        self._legs = jp.array([jid(n) for n in LEGS])
        self._head = jp.array([jid(n) for n in HEAD])
        self._roll = jp.array([jid(n) for n in HIP_ROLL])
        self._yaw = jp.array([jid(n) for n in HIP_YAW])
        self._action_scale, self._kick_vel = action_scale, kick_vel
        ps = jax.jit(self.pipeline_init)(self._home, jp.zeros(sys.nv))
        geom = jax.device_get(ps.contact.geom)
        soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in SOLES]
        self._sole_rows = jp.array([[float(g1 == s or g2 == s) for g1, g2 in geom] for s in soles])

    @property
    def action_size(self):
        return 2 + len(self._jq)

    # ---- helpers
    def foot_contact(self, ps):
        d = jp.where(self._sole_rows > 0, ps.contact.dist[None, :], 1.0)
        return d.min(axis=1) < 0.002

    def ground(self, xy, box):
        """Height of the ground under a point: the floor, or the obstacle's top. box = (x, top)."""
        inside = (jp.abs(xy[0] - box[0]) < BOX_HALF[0]) & (jp.abs(xy[1]) < BOX_HALF[1])
        return jp.where(inside, jp.maximum(box[1], 0.0), 0.0)

    def sample_command(self, rng, obstacle):
        k = jax.random.split(rng, 6)
        fly = jax.random.uniform(k[0]) < 0.5
        walk = jp.array([jax.random.uniform(k[1], minval=-0.1, maxval=0.3),
                         jax.random.uniform(k[2], minval=-0.1, maxval=0.1),
                         jax.random.uniform(k[3], minval=-0.7, maxval=0.7), 0.0])
        walk = jp.where(jax.random.uniform(k[4]) < 0.15, jp.zeros(4), walk)        # stand still
        air = jp.array([jax.random.uniform(k[1], minval=-0.4, maxval=1.0),
                        jax.random.uniform(k[2], minval=-0.3, maxval=0.3),
                        jax.random.uniform(k[3], minval=-1.0, maxval=1.0),
                        jax.random.uniform(k[5], minval=0.15, maxval=1.0)])
        cmd = jp.where(fly, air, walk)
        ahead = jp.array([jax.random.uniform(k[1], minval=0.15, maxval=0.3), 0.0, 0.0, 0.0])
        return jp.where(obstacle, ahead, cmd)

    def gait_hz(self, cmd):
        return 1.8 + 3.0 * jp.abs(cmd[0])

    # ---- episode
    def reset(self, rng):
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
        u = jax.random.uniform(k1)
        obstacle, airborne = u < P_OBSTACLE, u > 1 - P_AIR
        level = jp.array(OBST_START)
        info = dict(rng=rng, level=level)
        box = self._place_box(k2, obstacle, level)
        q = self._home.at[7:].add(jax.random.uniform(k3, (self._default.shape[0],), minval=-0.05, maxval=0.05))
        q = q.at[2].add(jp.where(airborne, jax.random.uniform(k4, minval=0.2, maxval=0.8), 0.0))
        ps = self.pipeline_init(q, jp.zeros(self.sys.nv))
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[0].set(jp.array([box[0], 0.0, box[1] - BOX_HALF[2]])))
        cmd = self.sample_command(k5, obstacle)
        # an airborne start is told to fly or to land (walk)
        cmd = jp.where(airborne & (jax.random.uniform(k6) < 0.5), cmd.at[3].set(0.0), cmd)
        thrust = jp.where(airborne, self._hover, 0.0) * jp.ones(2)
        info.update(box=box, obstacle=obstacle, airborne=airborne, passed=jp.zeros((), bool), command=cmd,
                    cmd_now=jp.where(airborne, cmd, jp.zeros(4)),
                    last_act=jp.zeros(self.action_size), last_target=self._default, thrust=thrust,
                    phase=jp.zeros(()), step=0, air_time=jp.zeros(2), single=jp.zeros(2),
                    last_single=jp.zeros(2), rewards={k: 0.0 for k in REWARDS})
        obs = self._obs(ps, info)
        metrics = {f"reward/{k}": 0.0 for k in REWARDS}
        metrics.update(level=level, flying=jp.zeros(()), passed=jp.zeros(()))
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _place_box(self, rng, obstacle, level):
        k1, k2 = jax.random.split(rng)
        x = jax.random.uniform(k1, minval=0.6, maxval=1.2)
        h = jax.random.uniform(k2, minval=0.5, maxval=1.0) * level
        return jp.where(obstacle, jp.array([x, h]), jp.array([50.0, 0.0]))

    def step(self, state, action):
        info0 = state.info
        rng, kick_rng, cmd_rng, box_rng = jax.random.split(info0["rng"], 4)
        push = jax.random.uniform(kick_rng, (2,), minval=-1.0, maxval=1.0) * self._kick_vel
        push = jp.where(info0["step"] % 150 == 149, push, jp.zeros(2))
        ps0 = state.pipeline_state
        box = info0["box"]
        # the obstacle lives in info (Brax's auto-reset restores the first pipeline state), so
        # put the box where info says every step
        ps0 = ps0.replace(qvel=ps0.qvel.at[:2].add(push),
                          mocap_pos=ps0.mocap_pos.at[0].set(jp.array([box[0], 0.0, box[1] - BOX_HALF[2]])))

        # fans: commanded fraction through the spool-up lag; servos: rate-limited targets
        want_t = jp.clip(action[:2], 0.0, 1.0)
        thrust = info0["thrust"] + (want_t - info0["thrust"]) * (self.dt / (SPOOL_TAU + self.dt))
        want = jp.clip(self._default[self._jq] + action[2:] * self._action_scale, self._lo, self._hi)
        dmax = SERVO_SPEED * self.dt
        target = jp.clip(want, info0["last_target"][self._jq] - dmax, info0["last_target"][self._jq] + dmax)
        ctrl = jp.zeros(self.sys.nu).at[self._fan_act].set(thrust * self._tmax).at[self._joint_act].set(target)
        ps = self.pipeline_step(ps0, ctrl)
        x, xd = ps.x, ps.xd

        # command easing; height target above the local ground
        dc = CMD_ACCEL * self.dt
        cmd = info0["cmd_now"]
        cmd = cmd.at[:3].set(cmd[:3] + jp.clip(info0["command"][:3] - cmd[:3], -dc, dc))
        cmd = cmd.at[3].set(cmd[3] + jp.clip(info0["command"][3] - cmd[3], -H_RATE * self.dt, H_RATE * self.dt))
        flying = cmd[3] > 0.05
        walking = ~flying
        moving = jp.linalg.norm(cmd[:3]) > 0.03
        phase = (info0["phase"] + self.gait_hz(cmd) * self.dt) % 1.0

        contact = self.foot_contact(ps)
        first_contact = (info0["air_time"] > 0.1) & contact
        alone = contact & ~contact[::-1]
        single = jp.where(alone, info0["single"] + self.dt, 0.0)
        ended = ~alone & (info0["single"] > 0)
        last_single = jp.where(ended, info0["single"], info0["last_single"])
        air_time = info0["air_time"] + self.dt

        g = self.ground(x.pos[0, :2], box)
        h = x.pos[0, 2] - g
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), x.rot[0])
        fallen = (h < 0.06) | (up[2] < 0.5) | (x.pos[0, 2] > 3.0)
        done = fallen.astype(jp.float32)

        s = jp.sin(2 * jp.pi * phase)
        stance = jp.stack([s >= -0.25, s <= 0.25])
        stance = jp.where(moving, stance, jp.ones(2, dtype=bool))
        local_v = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        local_w = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        q = ps.q[7:]
        h_err = h - (self._h0 + cmd[3])
        r = dict(
            tracking_lin_vel=jp.exp(-jp.sum(jp.square(cmd[:2] - local_v[:2])) / (0.01 + 0.04 * flying)),
            tracking_ang_vel=jp.exp(-jp.square(cmd[2] - local_w[2]) / 0.1),
            height=jp.exp(-jp.square(h_err) / (0.002 + 0.01 * flying)),
            gait_phase=jp.mean((contact == stance).astype(jp.float32)) * walking,
            feet_air_time=jp.sum(jp.clip(air_time - 0.1, max=0.2) * first_contact) * moving * walking,
            symmetry=jp.abs(last_single[0] - last_single[1]) * moving * walking,
            lin_vel_z=jp.square(xd.vel[0, 2]) * walking,
            ang_vel_xy=jp.sum(jp.square(xd.ang[0, :2])),
            orientation=jp.sum(jp.square(up[:2])),
            hip_splay=(jp.sum(jp.square(q[self._roll] - self._default[self._roll]))
                       + jp.sum(jp.square(q[self._yaw] - self._default[self._yaw])) * (jp.abs(cmd[2]) < 0.1)) * walking,
            legs_home=jp.sum(jp.square(q[self._legs] - self._default[self._legs])) * flying,
            torques=jp.sum(jp.square(ps.qfrc_actuator[6:])),
            action_rate=jp.sum(jp.square(action - info0["last_act"])),
            stand_still=jp.sum(jp.abs(q[self._legs] - self._default[self._legs])) * (1 - moving) * walking,
            head_still=jp.sum(jp.square(q[self._head] - self._default[self._head])),
            fan_power=jp.mean(thrust) * jp.where(flying, FAN_COST[1], FAN_COST[0]),
            landing=jp.square(xd.vel[0, 2]) * jp.any(first_contact),
            termination=done * (info0["step"] < EPISODE - 10),
        )
        r = {k: v * REWARDS[k] for k, v in r.items()}
        reward = jp.clip(sum(r.values()) * self.dt, 0.0, 10000.0)

        # obstacle curriculum: past the box (and still up) -> taller boxes for this duck
        passed = info0["passed"] | (info0["obstacle"] & (x.pos[0, 0] > box[0] + BOX_HALF[0] + 0.1))
        newly = passed & ~info0["passed"]
        level = jp.where(newly, info0["level"] + OBST_STEP, info0["level"])
        level = jp.where(fallen & info0["obstacle"], info0["level"] - OBST_STEP, level)
        level = jp.clip(level, OBST_START, OBST_TOP)

        step = info0["step"] + 1
        seg_end = (step % SEGMENT == 0)
        info = dict(info0)
        info.update(air_time=air_time * ~contact, last_act=action, rng=rng, thrust=thrust,
                    last_target=jp.where(fallen, self._default, self._default.at[self._jq].set(target)),
                    phase=jp.where(fallen, 0.0, phase), single=jp.where(fallen, 0.0, single),
                    last_single=jp.where(fallen, 0.0, last_single), level=level, passed=passed, rewards=r)
        new_cmd = self.sample_command(cmd_rng, info0["obstacle"])
        info["command"] = jp.where(seg_end, new_cmd, info0["command"])
        info["cmd_now"] = cmd
        # A fall or the time limit ends the episode; Brax restores the first pipeline state (so
        # an airborne-start duck starts in the air again) and the rest is re-drawn here: a new
        # box at this duck's level, a new command, fans off (at hover in the air).
        restart = fallen | (step >= EPISODE)
        air = info0["airborne"]
        info["step"] = jp.where(restart, 0, step)
        info["box"] = jp.where(restart, self._place_box(box_rng, info0["obstacle"], level), box)
        info["passed"] = jp.where(restart, False, passed)
        restart_cmd = jp.where(air & (jax.random.uniform(box_rng) < 0.5), new_cmd.at[3].set(0.0), new_cmd)
        info["command"] = jp.where(restart, restart_cmd, info["command"])
        info["cmd_now"] = jp.where(restart, jp.where(air, restart_cmd, jp.zeros(4)), cmd)
        info["thrust"] = jp.where(restart, jp.where(air, self._hover, 0.0) * jp.ones(2), info["thrust"])
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in r.items()})
        metrics.update(level=level, flying=flying.astype(jp.float32), passed=passed.astype(jp.float32))
        return state.replace(pipeline_state=ps, obs=self._obs(ps, info), reward=reward, done=done, metrics=metrics,
                             info=info)

    def _obs(self, ps, info):
        inv = math.quat_inv(ps.x.rot[0])
        a = 2 * jp.pi * info["phase"]
        box = info["box"]
        p = ps.x.pos[0]
        g = self.ground(p[:2], box)
        # the obstacle ahead in the trunk's heading frame: near edge, far edge (m), top above the trunk
        fwd = math.rotate(jp.array([1.0, 0.0, 0.0]), ps.x.rot[0])
        yaw = jp.arctan2(fwd[1], fwd[0])
        dx = box[0] - p[0]
        rel = jp.array([dx * jp.cos(yaw) + p[1] * jp.sin(yaw) - BOX_HALF[0],
                        dx * jp.cos(yaw) + p[1] * jp.sin(yaw) + BOX_HALF[0], box[1] - p[2]])
        rel = jp.clip(rel, -2.0, 2.0)
        return jp.concatenate([
            math.rotate(ps.xd.ang[0], inv) * 0.25,                 # body rates (IMU gyro)
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv),          # gravity direction (IMU)
            ps.q[7:] - self._default,                              # joint angles from home
            ps.qd[6:] * 0.05,                                      # joint velocities
            info["last_act"],
            info["thrust"],                                        # fan thrust (fraction of max)
            info["cmd_now"] * jp.array([2.0, 2.0, 0.5, 2.0]),
            jp.array([jp.cos(a), jp.sin(a)]),                      # gait clock
            math.rotate(ps.xd.vel[0], inv) * 0.5,                  # body velocity (estimator)
            jp.array([(p[2] - g - self._h0) * 2.0]),               # height above ground (rangefinder)
            rel,                                                   # obstacle ahead (range sensor)
        ])
