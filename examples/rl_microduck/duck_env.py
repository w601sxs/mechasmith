"""Joystick walking for the mechasmith duck, in MJX (Brax pipeline env).

The policy sees the IMU (gravity direction and body rates), joint angles relative to home,
joint velocities, its last action, the velocity command and a gait clock, and outputs joint
position targets as offsets from home (the model's position servos hold them). One decision every
0.02 s (5 physics steps of 4 ms).

Rewards: follow the commanded forward/sideways speed and turn rate; put the feet down in time
with the gait clock (left foot down in the first half of each cycle, right foot in the second,
both near the handovers), which makes the gait alternate evenly; stay upright and level; keep the
hip roll near home (and the hip yaw too, unless turning) and the head still; take equally long
steps on both feet (no limp); move smoothly. Foot contact is the
sole-floor contact distance from MJX, not foot height.

Speed ramp: each simulated duck has its own top forward speed, starting at 0.15 m/s. A quarter of
commands are straight ahead at that top speed; after one, the top speed rises 0.025 m/s if the duck
averaged at least 85% of it, and it drops when the duck falls; the cap is 0.3 m/s. Within an episode the command eases toward its target at 0.5 m/s^2. The gait clock runs
faster at higher speed. Servo targets move at most 10.8 rad/s (the XL330's no-load speed, from
mechasmith's servo spec), so the policy cannot ask for faster joints than the servo has.

An episode ends when the duck falls (trunk below 7 cm or tipped past 60 degrees) or after 20 s.
Pushes make it robust.
"""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# mechasmith's training model of the duck: `mechasmith build designs/microduck.py --sim mjx`, then
# copy build/microduck/robot_mjx.xml here as duck_mjx.xml.
XML = Path(__file__).with_name("duck_mjx.xml")
LEGS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
HIP_ROLL = ["left_hip_roll", "right_hip_roll"]
HIP_YAW = ["left_hip_yaw", "right_hip_yaw"]
SOLES = ["left_sole_collision", "right_sole_collision"]
# Run 1 (tracking 1.5, air time 2.0) learned to step in place. Run 2 (tracking 4.0, air time 1.0) walked
# forward but limped, splayed its hips and bobbed its head; its foot-height contact test was wrong.
# Run 3 (gait clock, real contacts, speed ramp) still limped, turned worse (the hip-yaw penalty fought
# turning) and reached 0.16 of 0.3 m/s: its +-0.3 rad action range capped the stride, and its ramp rose
# on average tracking of all commands, not the top speed.
REWARDS = dict(tracking_lin_vel=4.0, tracking_ang_vel=1.5, gait_phase=3.0, feet_air_time=0.5, symmetry=-10.0,
               lin_vel_z=-2.0, ang_vel_xy=-0.05, orientation=-5.0, hip_splay=-2.0, torques=-0.002,
               action_rate=-0.01, stand_still=-0.5, head_still=-5.0, termination=-1.0)
V_START, V_TOP, V_STEP = 0.15, 0.30, 0.025   # speed ramp (m/s)
CMD_ACCEL = 0.5                              # m/s^2 (and rad/s^2) toward the target command
SERVO_SPEED = 10.8                           # rad/s, XL330 no-load speed (mechasmith ServoSpec.max_velocity)
SEGMENT = 400                                # control steps per command segment (8 s)
SETTLE = 100                                 # steps at the start of a segment left out of the speed check
PROMOTE = 0.85                               # fraction of the top speed to reach before it rises


class DuckJoystick(PipelineEnv):
    def __init__(self, xml=XML, action_scale=0.6, kick_vel=0.05, **kw):
        sys = mjcf.load(str(xml))
        super().__init__(sys, backend="mjx", n_frames=5, **kw)
        m = sys.mj_model
        self._home = jp.array(m.keyframe("home").qpos)
        self._default = self._home[7:]
        self._lo = jp.array(m.jnt_range[1:, 0])
        self._hi = jp.array(m.jnt_range[1:, 1])
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) - 1      # skip the freejoint
        self._head = jp.array([jid(n) for n in HEAD])
        self._roll = jp.array([jid(n) for n in HIP_ROLL])
        self._yaw = jp.array([jid(n) for n in HIP_YAW])
        self._action_scale = action_scale
        self._kick_vel = kick_vel
        # which rows of MJX's (fixed) contact list are each sole against the floor
        ps = jax.jit(self.pipeline_init)(self._home, jp.zeros(sys.nv))
        geom = jax.device_get(ps.contact.geom)
        soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in SOLES]
        self._sole_rows = jp.array([[float(g1 == s or g2 == s) for g1, g2 in geom] for s in soles])

    def foot_contact(self, ps):
        d = jp.where(self._sole_rows > 0, ps.contact.dist[None, :], 1.0)
        return d.min(axis=1) < 0.002

    def sample_command(self, rng, v_top):
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        cmd = jp.array([jax.random.uniform(k1, minval=-0.1, maxval=v_top),
                        jax.random.uniform(k2, minval=-0.1, maxval=0.1),
                        jax.random.uniform(k3, minval=-0.7, maxval=0.7)])
        u = jax.random.uniform(k4)
        cmd = jp.where(u < 0.1, jp.zeros(3), cmd)                                     # stand still
        cmd = jp.where((u >= 0.1) & (u < 0.2), cmd * jp.array([0.0, 0.0, 1.0]), cmd)  # turn in place
        at_top = (u >= 0.2) & (u < 0.45)
        cmd = jp.where(at_top, jp.array([v_top, 0.0, 0.0]), cmd)                      # straight at top speed
        return jp.where(jp.linalg.norm(cmd[:2]) < 0.03, cmd * jp.array([0.0, 0.0, 1.0]), cmd), at_top

    def gait_hz(self, cmd):
        return 1.8 + 3.0 * jp.abs(cmd[0])     # 2.25 Hz at 0.15 m/s, 2.7 Hz at 0.3 m/s

    def reset(self, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        q = self._home.at[7:].add(jax.random.uniform(k1, (self._default.shape[0],), minval=-0.05, maxval=0.05))
        ps = self.pipeline_init(q, jp.zeros(self.sys.nv))
        command, at_top = self.sample_command(k2, V_START)
        info = dict(rng=rng, last_act=jp.zeros(self.sys.nu), last_target=self._default,
                    command=command, at_top=at_top, cmd_now=jp.zeros(3), phase=jp.zeros(()),
                    v_top=jp.array(V_START), track=jp.zeros(()), step=0, air_time=jp.zeros(2),
                    single=jp.zeros(2), last_single=jp.zeros(2),
                    last_contact=jp.ones(2, dtype=bool), rewards={k: 0.0 for k in REWARDS})
        obs = self._obs(ps, info)
        metrics = {f"reward/{k}": 0.0 for k in REWARDS}
        metrics["v_top"] = jp.array(V_START)
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        info0 = state.info
        rng, kick_rng, cmd_rng = jax.random.split(info0["rng"], 3)
        # an occasional push: a random change of the trunk's horizontal velocity
        push = jax.random.uniform(kick_rng, (2,), minval=-1.0, maxval=1.0) * self._kick_vel
        push = jp.where(info0["step"] % 150 == 149, push, jp.zeros(2))
        ps0 = state.pipeline_state
        ps0 = ps0.replace(qvel=ps0.qvel.at[:2].add(push))
        # servo targets: offsets from home, within joint limits, moving no faster than the servo can
        want = jp.clip(self._default + action * self._action_scale, self._lo, self._hi)
        dmax = SERVO_SPEED * self.dt
        target = jp.clip(want, info0["last_target"] - dmax, info0["last_target"] + dmax)
        ps = self.pipeline_step(ps0, target)
        x, xd = ps.x, ps.xd

        # the command eases toward its target; the gait clock advances
        dc = CMD_ACCEL * self.dt
        cmd = info0["cmd_now"] + jp.clip(info0["command"] - info0["cmd_now"], -dc, dc)
        moving = jp.linalg.norm(cmd) > 0.03
        phase = (info0["phase"] + self.gait_hz(cmd) * self.dt) % 1.0

        contact = self.foot_contact(ps)
        first_contact = (info0["air_time"] > 0) & contact
        # how long each foot carried the duck alone, last time it did (for the no-limp reward)
        alone = contact & ~contact[::-1]
        single = jp.where(alone, info0["single"] + self.dt, 0.0)
        ended = ~alone & (info0["single"] > 0)
        last_single = jp.where(ended, info0["single"], info0["last_single"])
        air_time = info0["air_time"] + self.dt

        up = math.rotate(jp.array([0.0, 0.0, 1.0]), x.rot[0])
        fallen = (x.pos[0, 2] < 0.07) | (up[2] < 0.5)
        done = fallen.astype(jp.float32)

        # left foot down for the first half of the cycle, right for the second; both near the handovers
        s = jp.sin(2 * jp.pi * phase)
        stance = jp.stack([s >= -0.25, s <= 0.25])
        stance = jp.where(moving, stance, jp.ones(2, dtype=bool))

        local_v = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        local_w = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        q = ps.q[7:]
        track_lin = jp.exp(-jp.sum(jp.square(cmd[:2] - local_v[:2])) / 0.01)
        r = dict(
            tracking_lin_vel=track_lin,
            tracking_ang_vel=jp.exp(-jp.square(cmd[2] - local_w[2]) / 0.1),
            gait_phase=jp.mean((contact == stance).astype(jp.float32)),
            feet_air_time=jp.sum(jp.clip(air_time - 0.1, max=0.2) * first_contact) * moving,
            lin_vel_z=jp.square(xd.vel[0, 2]),
            ang_vel_xy=jp.sum(jp.square(xd.ang[0, :2])),
            orientation=jp.sum(jp.square(up[:2])),
            symmetry=jp.abs(last_single[0] - last_single[1]) * moving,
            hip_splay=jp.sum(jp.square(q[self._roll] - self._default[self._roll]))
            + jp.sum(jp.square(q[self._yaw] - self._default[self._yaw])) * (jp.abs(cmd[2]) < 0.1),
            torques=jp.sum(jp.square(ps.qfrc_actuator[6:])),
            action_rate=jp.sum(jp.square(action - info0["last_act"])),
            stand_still=jp.sum(jp.abs(q - self._default)) * (1 - moving),
            head_still=jp.sum(jp.square(q[self._head] - self._default[self._head])),
            termination=done * (info0["step"] < 990),
        )
        r = {k: v * REWARDS[k] for k, v in r.items()}
        reward = jp.clip(sum(r.values()) * self.dt, 0.0, 10000.0)

        # speed ramp: at the end of each command segment, raise this duck's top speed if it tracked
        # well; lower it when it falls
        step = info0["step"] + 1
        seg_step = step % SEGMENT
        track = info0["track"] + jp.where(seg_step > SETTLE, local_v[0], 0.0)    # forward speed, after settling
        seg_end = seg_step == 0
        good = info0["at_top"] & (track / (SEGMENT - SETTLE) >= PROMOTE * info0["v_top"])
        v_top = jp.where(seg_end & good, info0["v_top"] + V_STEP, info0["v_top"])
        v_top = jp.clip(jp.where(fallen, v_top - V_STEP, v_top), V_START, V_TOP)

        info = dict(info0)
        info["air_time"] = air_time * ~contact
        info["last_contact"] = contact
        info["last_act"] = action
        info["last_target"] = jp.where(fallen, self._default, target)
        info["rng"] = rng
        info["phase"] = jp.where(fallen, 0.0, phase)
        info["v_top"] = v_top
        info["track"] = jp.where(seg_end | fallen, 0.0, track)
        new_cmd, new_top = self.sample_command(cmd_rng, v_top)
        resample = seg_end | fallen
        info["command"] = jp.where(resample, new_cmd, info0["command"])
        info["at_top"] = jp.where(resample, new_top, info0["at_top"])
        info["single"] = jp.where(fallen, 0.0, single)
        info["last_single"] = jp.where(fallen, 0.0, last_single)
        info["cmd_now"] = jp.where(fallen, jp.zeros(3), cmd)
        info["step"] = jp.where(fallen | (step > 1000), 0, step)
        info["rewards"] = r
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in r.items()})
        metrics["v_top"] = v_top
        return state.replace(pipeline_state=ps, obs=self._obs(ps, info), reward=reward, done=done, metrics=metrics,
                             info=info)

    def _obs(self, ps, info):
        inv = math.quat_inv(ps.x.rot[0])
        a = 2 * jp.pi * info["phase"]
        return jp.concatenate([
            math.rotate(ps.xd.ang[0], inv) * 0.25,                 # body rates (IMU gyro)
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv),          # gravity direction (IMU)
            ps.q[7:] - self._default,                              # joint angles from home
            ps.qd[6:] * 0.05,                                      # joint velocities
            info["last_act"],
            info["cmd_now"] * jp.array([2.0, 2.0, 0.5]),
            jp.array([jp.cos(a), jp.sin(a)]),                      # gait clock
        ])


def domain_randomize(sys, rng):
    """Brax randomization_fn: per environment, scale every body's mass by U(0.8, 1.2), move its
    centre of mass by up to 5 mm per axis and set the floor-contact friction in U(0.6, 1.4).
    These are the usual smooth randomisations; the paper asks whether they hide discrete
    authoring faults."""
    @jax.vmap
    def draw(key):
        k1, k2, k3 = jax.random.split(key, 3)
        mass = sys.body_mass * jax.random.uniform(k1, sys.body_mass.shape, minval=0.8, maxval=1.2)
        ipos = sys.body_ipos + jax.random.uniform(k2, sys.body_ipos.shape, minval=-0.005, maxval=0.005)
        mu = jax.random.uniform(k3, (), minval=0.6, maxval=1.4)
        return mass, ipos, sys.geom_friction.at[:, 0].set(mu)
    mass, ipos, friction = draw(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"body_mass": 0, "body_ipos": 0, "geom_friction": 0})
    sys = sys.tree_replace({"body_mass": mass, "body_ipos": ipos, "geom_friction": friction})
    return sys, in_axes
