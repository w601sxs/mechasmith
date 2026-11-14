"""Joystick walking for the mechasmith duck, in MJX (Brax pipeline env).

The policy sees the IMU (gravity direction and body rates), joint angles relative to home,
joint velocities, its last action and the velocity command, and outputs joint position
targets as offsets from home (the model's position servos hold them). One decision every
0.02 s (5 physics steps of 4 ms). Rewards: follow the commanded forward/sideways speed and
turn rate, stay upright and level, step with the feet off the ground for a while, keep the
head still, move smoothly. An episode ends when the duck falls (trunk below 7 cm or tipped
past 60 degrees) or after 20 s. Pushes and random friction make it robust.
"""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# mechasmith's training model of the duck: `mechasmith build designs/microduck.py --sim mjx`, then
# copy build/microduck/robot_mjx.xml here as duck_mjx.xml.
XML = Path(__file__).with_name("duck_mjx.xml")
LEGS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
REWARDS = dict(tracking_lin_vel=1.5, tracking_ang_vel=0.8, lin_vel_z=-2.0, ang_vel_xy=-0.05, orientation=-5.0,
               torques=-0.002, action_rate=-0.01, feet_air_time=2.0, stand_still=-0.5, head_still=-1.0,
               termination=-1.0)


class DuckJoystick(PipelineEnv):
    def __init__(self, xml=XML, action_scale=0.3, kick_vel=0.05, **kw):
        sys = mjcf.load(str(xml))
        super().__init__(sys, backend="mjx", n_frames=5, **kw)
        m = sys.mj_model
        self._home = jp.array(m.keyframe("home").qpos)
        self._default = self._home[7:]
        self._lo = jp.array(m.jnt_range[1:, 0])
        self._hi = jp.array(m.jnt_range[1:, 1])
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) - 1      # skip the freejoint
        self._head = jp.array([jid(n) for n in HEAD])
        self._feet = jp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")])
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        mujoco.mj_forward(m, d)
        self._foot_z0 = float(d.site_xpos[self._feet[0]][2]) + 0.004       # contact when a foot site is below this
        self._action_scale = action_scale
        self._kick_vel = kick_vel

    def sample_command(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        cmd = jp.array([jax.random.uniform(k1, minval=-0.1, maxval=0.2),
                        jax.random.uniform(k2, minval=-0.1, maxval=0.1),
                        jax.random.uniform(k3, minval=-0.7, maxval=0.7)])
        return jp.where(jp.linalg.norm(cmd[:2]) < 0.03, cmd * jp.array([0.0, 0.0, 1.0]), cmd)

    def reset(self, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        q = self._home.at[7:].add(jax.random.uniform(k1, (self._default.shape[0],), minval=-0.05, maxval=0.05))
        ps = self.pipeline_init(q, jp.zeros(self.sys.nv))
        info = dict(rng=rng, last_act=jp.zeros(self.sys.nu), command=self.sample_command(k2), step=0,
                    air_time=jp.zeros(2), last_contact=jp.zeros(2, dtype=bool),
                    rewards={k: 0.0 for k in REWARDS})
        obs = self._obs(ps, info)
        metrics = {f"reward/{k}": 0.0 for k in REWARDS}
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        rng, kick_rng, cmd_rng = jax.random.split(state.info["rng"], 3)
        # an occasional push: a random change of the trunk's horizontal velocity
        push = jax.random.uniform(kick_rng, (2,), minval=-1.0, maxval=1.0) * self._kick_vel
        push = jp.where(state.info["step"] % 150 == 149, push, jp.zeros(2))
        qd = state.pipeline_state.qd.at[:2].add(push)
        ps0 = state.pipeline_state.replace(qvel=qd) if hasattr(state.pipeline_state, "qvel") else state.pipeline_state
        target = jp.clip(self._default + action * self._action_scale, self._lo, self._hi)
        ps = self.pipeline_step(ps0, target)
        x, xd = ps.x, ps.xd

        foot_z = ps.site_xpos[self._feet][:, 2]
        contact = foot_z < self._foot_z0
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["air_time"] > 0) & contact_filt
        air_time = state.info["air_time"] + self.dt

        up = math.rotate(jp.array([0.0, 0.0, 1.0]), x.rot[0])
        fallen = (x.pos[0, 2] < 0.07) | (up[2] < 0.5)
        done = fallen.astype(jp.float32)

        cmd = state.info["command"]
        local_v = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        local_w = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        q = ps.q[7:]
        moving = jp.linalg.norm(cmd) > 0.03
        r = dict(
            tracking_lin_vel=jp.exp(-jp.sum(jp.square(cmd[:2] - local_v[:2])) / 0.01),
            tracking_ang_vel=jp.exp(-jp.square(cmd[2] - local_w[2]) / 0.1),
            lin_vel_z=jp.square(xd.vel[0, 2]),
            ang_vel_xy=jp.sum(jp.square(xd.ang[0, :2])),
            orientation=jp.sum(jp.square(up[:2])),
            torques=jp.sum(jp.square(ps.qfrc_actuator[6:])),
            action_rate=jp.sum(jp.square(action - state.info["last_act"])),
            feet_air_time=jp.sum((air_time - 0.15) * first_contact) * moving,
            stand_still=jp.sum(jp.abs(q - self._default)) * (1 - moving),
            head_still=jp.sum(jp.square(q[self._head] - self._default[self._head])),
            termination=done * (state.info["step"] < 990),
        )
        r = {k: v * REWARDS[k] for k, v in r.items()}
        reward = jp.clip(sum(r.values()) * self.dt, 0.0, 10000.0)

        info = dict(state.info)
        info["air_time"] = air_time * ~contact_filt
        info["last_contact"] = contact
        info["last_act"] = action
        info["step"] = state.info["step"] + 1
        info["rng"] = rng
        info["command"] = jp.where(info["step"] % 500 == 0, self.sample_command(cmd_rng), cmd)
        info["step"] = jp.where(fallen | (info["step"] > 1000), 0, info["step"])
        info["rewards"] = r
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in r.items()})
        return state.replace(pipeline_state=ps, obs=self._obs(ps, info), reward=reward, done=done, metrics=metrics,
                             info=info)

    def _obs(self, ps, info):
        inv = math.quat_inv(ps.x.rot[0])
        return jp.concatenate([
            math.rotate(ps.xd.ang[0], inv) * 0.25,                 # body rates (IMU gyro)
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv),          # gravity direction (IMU)
            ps.q[7:] - self._default,                              # joint angles from home
            ps.qd[6:] * 0.05,                                      # joint velocities
            info["last_act"],
            info["command"] * jp.array([2.0, 2.0, 0.5]),
        ])
