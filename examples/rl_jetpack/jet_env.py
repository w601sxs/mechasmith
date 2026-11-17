"""One policy for the jet-pack Microduck: walk, hop, fly, hover and land (MJX, Brax pipeline env).

The policy drives all 18 actuators: the 14 duck servos and the 2 nacelle-tilt servos (position
targets as offsets from home) and the 2 ducted fans (thrust, 0..12 N each, through a first-order
spool-up lag). One decision every 12 ms (3 physics steps of 4 ms).

Commands: forward / sideways speed, turn rate, a height for the trunk above the ground
directly beneath it, a lightness: the share of the weight the fans should carry (0 normally;
0.5-0.85 for a moon run, long floating bounds; the Moon's gravity is 1/6 of Earth's), and hop
(1: bound with both feet together; 0: alternate feet). At standing height that means walk; above it, fly; and the ground beneath
includes the top of an obstacle, so "walk on at standing height" past a box means getting onto it
(or over it). Nothing says how: the fans cost energy, so walking wins where walking works.

Episodes (each is one of):
  * ground, no obstacle: command segments switch between walking and flying (take-off and
    landing happen at the switches);
  * obstacle: a 0.4 m long box across the path, walk forward at standing height; its height
    rises per simulated duck as it gets past (curriculum, 3 cm -> 30 cm), falls when it falls;
  * puddle (long jump): a 1.2 m wide puddle across the path, run forward at standing height;
    a foot in the water ends the episode. Not flying: the trunk must stay within 10 cm of
    standing height and the fans may boost for at most 0.4 s per leap (both penalised beyond),
    so it runs up, pushes off with a short fan boost and lands running. Its length rises per
    simulated duck as it clears them (15 cm -> 1 m);
  * airborne start: dropped in mid-air with the fans at hover, commanded to fly or to land.
On the ground, segments are walk, fly or moon run (0.3-1 m/s, lightness 0.5-0.85).

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
BOX_HALF = (0.2, 0.6, 0.4)          # obstacle half sizes, m (0.4 long, 1.2 wide); its height is set by sinking it
PUDDLE_HALF_W = 0.6                 # puddle half width, m

REWARDS = dict(tracking_lin_vel=4.0, tracking_ang_vel=1.5, height=3.0, gait_phase=2.0, feet_air_time=0.5,
               symmetry=-5.0, lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-5.0, hip_splay=-2.0,
               legs_home=-1.0, torques=-0.002, action_rate=-0.01, stand_still=-0.5, head_still=-5.0,
               fan_power=-1.0, landing=-2.0, lightness=2.0, moon_air_time=1.5, moon_rise=20.0,
               obstacle_progress=3.0, obstacle_pass=300.0, feet_flat=-15.0, boost_over=-30.0, jump_high=-100.0,
               single_support=2.0, run_phase=2.0, feet_together=-150.0, hop_phase=2.0, hop_together=150.0,
               hop_air_time=1.0, leap=500.0, topped=150.0, termination=-1.0)
FAN_COST = (6.0, 0.3, 15.0, 2.0)    # fan_power multiplier walking, flying, hopping, at a box
# Run 3: runs 1-2 stalled at 3-6 cm boxes (waiting at a box cost little, a fall lowered it), idled
# the fans at 10-30 % when walking (cost 2.0) and skimmed rather than bounded on the moon run.
# Now: a bonus for getting past a box and for forward progress until then, a fall lowers the box
# by half a step, walking fans cost 3x more, and moon strides are paid for the trunk's rise.
# Run 4 adds: flat feet on the ground (runs 1-2 ran on tip-toe), and the long jump over a puddle.
# Run 5: run 4 hopped, both feet together, at every speed: "equal steps" scored a hop perfectly (no
# single-foot steps at all), air time paid both feet at a two-footed landing, and the run-up and
# moon run had no gait terms. Now: reward time on one foot, penalise both feet landing within
# 50 ms, a running gait clock (left, flight, right, flight) for the run-up and the moon run, air
# time only for single-foot landings, and the boost cap is binding: past 0.4 s the fans are cut
# to 30 % until a foot is down again (in the long jump and on the ground, not in flight or moon).
# Run 6: hopping is a skill on command. A sixth command, hop (0/1): with it, both feet land and
# take off together in time with a hop clock (and the alternating-feet terms are off); without
# it, the run-5 terms make it walk and run on alternating feet. A quarter of ground commands hop.
# Run 7: run 6 stopped at every box taller than ~6 cm because the fall test measured the trunk's
# height above the ground under the trunk: once the trunk crossed a box's near edge with the feet
# still on the floor it counted as fallen. Falls are now measured from the lowest ground under the
# trunk or either sole. Run 6 also stepped into every puddle (longest flight 12 cm): now each
# landing in a puddle episode is paid for the distance flown beyond 15 cm, the puddle curriculum
# starts at 5 cm, and hopping pays 2.5x the walking fan cost (it hopped on 47 % fans).
# Run 8 (warm-started from run 7): run 7 climbed 10 cm boxes but stopped at 15-20 cm (the 0.4 s
# boost cap cut the fans mid-climb), fell taking off next to a box (never practised), and still
# stepped into 10 cm puddles (the leap paid nothing below 15 cm, longer than its stride). Now: boxes
# up to 60 cm; at a box (0.45 m before it to past it) the boost cap is lifted and the fans cost a
# third as much, and the first time both feet are on top pays a bonus; flight holds its height
# above the highest ground 0.15-0.35 m ahead as well as below, so it climbs before a tall box,
# not into it. Box episodes start in one of three ways: walking in from 0.6-1.2 m, standing
# 10-25 cm from the near edge (told to walk on, or to take off), or standing on a 5-35 cm box near
# its far edge (told to walk off). The leap pays the distance flown beyond 6 cm for take-offs
# within 0.4 m of the puddle. Once past an obstacle, a new one of the same kind (box at this
# duck's level, or puddle) appears 0.8-1.5 m ahead, so an episode has a course of them.
# Run 8b (from run 8 at 313 M steps): run 8 stopped at even a 10 cm box after 313 M steps, toes
# against its face. Any pass raised the box height (flights on command, walking off a box it was
# started on, even puddles), so walk-in boxes were far taller than it could climb. Now only a pass
# after climbing onto the box itself raises it, and boxes are 30-100 % (not 50-100 %) of the level.
# Puddles: from 0.4 m before a puddle to past it the fans cost a third (the boost cap still holds).
CMD_ACCEL, H_RATE, L_RATE = 0.5, 0.4, 1.0   # command easing: m/s^2, rad/s^2; m/s (height); 1/s (lightness)
SERVO_SPEED = 10.8                  # rad/s, XL330 no-load speed
SPOOL_TAU = 0.06                    # s, fan spool-up / down time constant (EDF ~0.05-0.1 s)
SEGMENT, EPISODE = 400, 1500        # control steps (4.8 s, 18 s)
OBST_START, OBST_STEP, OBST_TOP = 0.05, 0.02, 0.60
PUD_START, PUD_STEP, PUD_TOP = 0.05, 0.03, 0.6     # puddle length curriculum, m
BOOST_FRAC, BOOST_MAX = 0.3, 0.4    # a "boost" is mean fan thrust above 30 %; allowed 0.4 s per leap
P_BOX, P_PUDDLE, P_AIR = 0.3, 0.2, 0.15
P_EDGE, P_TOP = 0.35, 0.2          # of box episodes: start at the near edge / on top


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
        self._soles = jp.array(soles)
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

    def sample_command(self, rng, obstacle, puddle, edge=False, top=False):
        k = jax.random.split(rng, 9)
        hop = (jax.random.uniform(k[6]) < 0.25).astype(jp.float32)
        u = jax.random.uniform(k[0])
        walk = jp.array([jax.random.uniform(k[1], minval=-0.1, maxval=0.3),
                         jax.random.uniform(k[2], minval=-0.1, maxval=0.1),
                         jax.random.uniform(k[3], minval=-0.7, maxval=0.7), 0.0, 0.0, hop])
        walk = jp.where(jax.random.uniform(k[4]) < 0.15, jp.zeros(6), walk)        # stand still
        air = jp.array([jax.random.uniform(k[1], minval=-0.4, maxval=1.0),
                        jax.random.uniform(k[2], minval=-0.3, maxval=0.3),
                        jax.random.uniform(k[3], minval=-1.0, maxval=1.0),
                        jax.random.uniform(k[5], minval=0.15, maxval=1.0), 0.0, 0.0])
        moon = jp.array([jax.random.uniform(k[1], minval=0.3, maxval=1.0),
                         jax.random.uniform(k[2], minval=-0.1, maxval=0.1),
                         jax.random.uniform(k[3], minval=-0.5, maxval=0.5), 0.0,
                         jax.random.uniform(k[5], minval=0.5, maxval=0.85), hop])
        cmd = jp.where(u < 0.4, walk, jp.where(u < 0.72, air, moon))
        ahead = jp.array([jax.random.uniform(k[1], minval=0.15, maxval=0.3), 0.0, 0.0, 0.0, 0.0, hop])
        run = jp.array([jax.random.uniform(k[1], minval=0.5, maxval=0.9), 0.0, 0.0, 0.0, 0.0, hop])
        # next to a box: walk on (step / hop / fly up as it likes) or take off and fly over
        takeoff = jp.array([jax.random.uniform(k[7], minval=0.25, maxval=0.6), 0.0, 0.0,
                            jax.random.uniform(k[8], minval=0.25, maxval=0.7), 0.0, 0.0])
        ahead = jp.where(edge & (jax.random.uniform(k[6]) < 0.4), takeoff, ahead)
        ahead = jp.where(top & (jax.random.uniform(k[6]) < 0.15), jp.zeros(6), ahead)    # stand on top
        return jp.where(puddle, run, jp.where(obstacle, ahead, cmd))

    def gait_hz(self, cmd, running=False):
        return jp.where(running, 2.2, 1.8 + 3.0 * jp.abs(cmd[0]))

    # ---- episode
    def reset(self, rng):
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
        k7, k8 = jax.random.split(jax.random.fold_in(k1, 8))
        u = jax.random.uniform(k1)
        obstacle, puddle = u < P_BOX, (u >= P_BOX) & (u < P_BOX + P_PUDDLE)
        airborne = u > 1 - P_AIR
        bt = jax.random.uniform(k7)
        edge, top = obstacle & (bt < P_EDGE), obstacle & (bt >= P_EDGE) & (bt < P_EDGE + P_TOP)
        top_h = jax.random.uniform(k8, minval=0.05, maxval=0.35)
        level, plevel = jp.array(OBST_START), jp.array(PUD_START)
        info = dict(rng=rng, level=level, plevel=plevel, edge=edge, top=top, top_h=top_h)
        box = self._place_box(k2, obstacle, level, edge, top, top_h)
        pud = self._place_puddle(k2, puddle, plevel)
        q = self._home.at[7:].add(jax.random.uniform(k3, (self._default.shape[0],), minval=-0.05, maxval=0.05))
        q = q.at[2].add(jp.where(airborne, jax.random.uniform(k4, minval=0.2, maxval=0.8), 0.0) + top * top_h)
        ps = self.pipeline_init(q, jp.zeros(self.sys.nv))
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[0].set(jp.array([box[0], 0.0, box[1] - BOX_HALF[2]])))
        cmd = self.sample_command(k5, obstacle, puddle, edge, top)
        # an airborne start is told to fly or to land (walk)
        cmd = jp.where(airborne & (jax.random.uniform(k6) < 0.5), cmd.at[3].set(0.0), cmd)
        thrust = jp.where(airborne, self._hover, 0.0) * jp.ones(2)
        info.update(box=box, obstacle=obstacle, pud=pud, puddle=puddle, boost=jp.zeros(()), td_age=jp.ones(2),
                    airborne=airborne, passed=jp.zeros((), bool), command=cmd, toff_x=jp.zeros(()),
                    topped=top, climbed=jp.array(False), chain=jp.array(True),
                    cmd_now=jp.where(airborne, cmd, jp.zeros(6)),
                    last_act=jp.zeros(self.action_size), last_target=self._default, thrust=thrust,
                    phase=jp.zeros(()), step=0, air_time=jp.zeros(2), single=jp.zeros(2),
                    last_single=jp.zeros(2), rewards={k: 0.0 for k in REWARDS})
        obs = self._obs(ps, info)
        metrics = {f"reward/{k}": 0.0 for k in REWARDS}
        metrics.update(level=level, plevel=plevel, flying=jp.zeros(()), moon=jp.zeros(()), passed=jp.zeros(()),
                       cleared=jp.zeros(()), courses=jp.zeros(()))
        return State(ps, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _place_box(self, rng, obstacle, level, edge=False, top=False, top_h=0.0, x0=0.0):
        """(centre x, top height). Walking in: 0.6-1.2 m ahead of x0. Edge: the near edge 10-25 cm
        ahead of the start. Top: the start (x = 0) on the box, its far edge 8-25 cm ahead."""
        k1, k2 = jax.random.split(rng)
        u = jax.random.uniform(k1)
        x = jp.where(edge, BOX_HALF[0] + 0.10 + 0.15 * u,
                     jp.where(top, 0.08 + 0.17 * u - BOX_HALF[0], x0 + 0.6 + 0.6 * u))
        h = jp.where(top, top_h, jax.random.uniform(k2, minval=0.3, maxval=1.0) * level)
        return jp.where(obstacle, jp.array([x, h]), jp.array([50.0, 0.0]))

    def _place_puddle(self, rng, puddle, plevel, x0=0.0):
        """(centre x, half length); far away when this is not a puddle episode."""
        k1, k2 = jax.random.split(rng)
        L = jax.random.uniform(k2, minval=0.6, maxval=1.0) * plevel
        near = x0 + jax.random.uniform(k1, minval=0.9, maxval=1.5)
        return jp.where(puddle, jp.array([near + L / 2, L / 2]), jp.array([60.0, 0.1]))

    def in_puddle(self, xy, pud):
        return (jp.abs(xy[..., 0] - pud[0]) < pud[1]) & (jp.abs(xy[..., 1]) < PUDDLE_HALF_W)

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
        # binding boost cap (not when flying or moon-running): past BOOST_MAX in the air, at most 30 %
        dxb = ps0.x.pos[0, 0] - box[0]
        at_box = info0["obstacle"] & (dxb > -(BOX_HALF[0] + 0.45)) & (dxb < BOX_HALF[0] + 0.1) & (box[1] > 0.08)
        free_fans = (info0["cmd_now"][3] > 0.05) | (info0["cmd_now"][4] > 0.2) | at_box
        pud0 = info0["pud"]
        at_pud = info0["puddle"] & (ps0.x.pos[0, 0] > pud0[0] - pud0[1] - 0.4) & (ps0.x.pos[0, 0] < pud0[0] + pud0[1] + 0.1)
        want_t = jp.where((info0["boost"] > BOOST_MAX) & ~free_fans, jp.minimum(want_t, BOOST_FRAC - 0.01), want_t)
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
        cmd = cmd.at[4].set(cmd[4] + jp.clip(info0["command"][4] - cmd[4], -L_RATE * self.dt, L_RATE * self.dt))
        cmd = cmd.at[5].set(info0["command"][5])
        flying = cmd[3] > 0.05
        moon = (cmd[4] > 0.2) & ~flying
        walking = ~flying & ~moon
        hop = (cmd[5] > 0.5) & ~flying
        moving = jp.linalg.norm(cmd[:3]) > 0.03
        running = moon | (info0["puddle"] & ~flying)
        phase = (info0["phase"] + jp.where(hop, 2.0, self.gait_hz(cmd, running)) * self.dt) % 1.0

        contact = self.foot_contact(ps)
        first_contact = (info0["air_time"] > 0.1) & contact
        # both feet landing within 50 ms of each other (a two-footed hop)
        td_age = jp.where(first_contact, 0.0, info0["td_age"] + self.dt)
        together = jp.any(first_contact & (td_age[::-1] < 0.05))
        step_down = first_contact & ~together           # a real single-foot landing
        alone = contact & ~contact[::-1]
        single = jp.where(alone, info0["single"] + self.dt, 0.0)
        ended = ~alone & (info0["single"] > 0)
        last_single = jp.where(ended, info0["single"], info0["last_single"])
        air_time = info0["air_time"] + self.dt

        g = self.ground(x.pos[0, :2], box)
        h = x.pos[0, 2] - g
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), x.rot[0])
        pud, puddle = info0["pud"], info0["puddle"]
        sole_xy = ps.geom_xpos[self._soles, :2]
        splash = jp.any(contact & self.in_puddle(sole_xy, pud))
        # fall test from the lowest ground under the trunk or a sole (not the box top when the
        # trunk is over the box's edge and the feet are still on the floor)
        g_low = jp.minimum(g, jp.minimum(self.ground(sole_xy[0], box), self.ground(sole_xy[1], box)))
        fallen = ((x.pos[0, 2] - g_low) < 0.06) | (up[2] < 0.5) | (x.pos[0, 2] > 3.0) | splash
        done = fallen.astype(jp.float32)
        boosting = jp.mean(thrust) > BOOST_FRAC
        boost = jp.where(jp.any(contact), 0.0, info0["boost"] + boosting * self.dt)
        sole_up = ps.geom_xmat[self._soles][:, :, 2]          # each sole's normal (world)
        jumping = puddle & ~flying & ~moon

        s = jp.sin(2 * jp.pi * phase)
        stance = jp.stack([s >= -0.25, s <= 0.25])
        stance = jp.where(moving, stance, jp.ones(2, dtype=bool))
        # running: left foot down for the first 30 % of the cycle, right for 50-80 %, flight between
        run_stance = jp.stack([phase < 0.3, (phase >= 0.5) & (phase < 0.8)])
        local_v = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        local_w = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        q = ps.q[7:]
        # in flight, hold the height above the highest ground below or 0.15-0.35 m ahead
        vxy = xd.vel[0, :2]
        hd = vxy / jp.maximum(jp.linalg.norm(vxy), 0.1)      # direction of travel (shrinks when hovering)
        g_fly = jp.maximum(g, jp.maximum(self.ground(x.pos[0, :2] + 0.15 * hd, box), self.ground(x.pos[0, :2] + 0.35 * hd, box)))
        h_err = x.pos[0, 2] - jp.where(flying, g_fly, g) - (self._h0 + cmd[3])
        # obstacle: past the box, or landed past the puddle (and still up)?
        past_box = info0["obstacle"] & (x.pos[0, 0] > box[0] + BOX_HALF[0] + 0.1)
        past_pud = puddle & (x.pos[0, 0] > pud[0] + pud[1] + 0.1) & jp.any(contact)
        passed = info0["passed"] | past_box | past_pud
        newly = passed & ~info0["passed"] & ~fallen
        walking = walking & ~jumping          # the walking gait terms don't apply to the run-up
        # long jump: at each landing in a puddle episode, the distance flown beyond 15 cm
        landed = jp.any(contact) & (info0["air_time"].min() > 0.1)
        near_pud = info0["toff_x"] > pud[0] - pud[1] - 0.4
        leap = jp.clip(x.pos[0, 0] - info0["toff_x"] - 0.06, 0.0, 1.2) * landed * jumping * near_pud * ~info0["passed"] * ~splash
        on_box = (jp.abs(sole_xy[:, 0] - box[0]) < BOX_HALF[0]) & (jp.abs(sole_xy[:, 1]) < BOX_HALF[1])
        topped_now = info0["obstacle"] & (box[1] > 0.04) & jp.all(contact & on_box)
        first_top = topped_now & ~info0["topped"] & ~fallen
        topped = info0["topped"] | topped_now
        toff_x = jp.where(jp.any(contact), x.pos[0, 0], info0["toff_x"])
        r = dict(
            tracking_lin_vel=jp.exp(-jp.sum(jp.square(cmd[:2] - local_v[:2])) / (0.01 + 0.04 * (flying | moon | jumping))),
            tracking_ang_vel=jp.exp(-jp.square(cmd[2] - local_w[2]) / 0.1),
            height=jp.exp(-jp.square(h_err) / (0.002 + 0.01 * (flying | moon | jumping))),
            gait_phase=jp.mean((contact == stance).astype(jp.float32)) * walking * ~hop,
            feet_air_time=jp.sum(jp.clip(air_time - 0.1, max=0.2) * step_down) * moving * walking * ~hop,
            single_support=(jp.sum(contact) == 1) * moving * ~flying * ~hop,
            run_phase=jp.mean((contact == run_stance).astype(jp.float32)) * running * ~hop,
            feet_together=together * moving * ~flying * ~hop,
            symmetry=jp.abs(last_single[0] - last_single[1]) * moving * walking * ~hop,
            # hop on command: both feet down for the first 40 % of the hop cycle, both up after
            hop_phase=jp.mean((contact == (phase < 0.4)).astype(jp.float32)) * hop * moving,
            hop_together=together * hop,
            hop_air_time=jp.sum(jp.clip(air_time - 0.1, max=0.3) * first_contact) * hop * moving,
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
            fan_power=jp.mean(thrust) * jp.where(flying, FAN_COST[1], jp.where(at_box | at_pud, FAN_COST[3],
                                                 jp.where(hop, FAN_COST[2], FAN_COST[0]))) * ~moon,
            feet_flat=jp.sum(jp.sum(jp.square(sole_up[:, :2]), axis=1) * contact) * ~flying,
            boost_over=jp.clip(boost - BOOST_MAX, 0.0, 1.0) * ~flying * ~moon * ~at_box,
            jump_high=jp.clip(h - self._h0 - 0.10, 0.0, 0.5) * jumping,
            lightness=jp.exp(-jp.square(jp.mean(thrust) / self._hover[0] - cmd[4]) / 0.01) * moon,
            moon_air_time=jp.sum(jp.clip(air_time - 0.15, max=0.5) * step_down) * moon,
            moon_rise=jp.clip(h - self._h0, 0.0, 0.1) * ~jp.any(contact) * moon,
            obstacle_progress=jp.clip(xd.vel[0, 0], 0.0, 0.4) * (info0["obstacle"] | puddle) * ~passed,
            obstacle_pass=newly.astype(jp.float32),
            landing=jp.square(xd.vel[0, 2]) * jp.any(first_contact),
            leap=leap,
            topped=first_top.astype(jp.float32),
            termination=done * (info0["step"] < EPISODE - 10),
        )
        r = {k: v * REWARDS[k] for k, v in r.items()}
        reward = jp.clip(sum(r.values()) * self.dt, 0.0, 10000.0)

        # obstacle curriculum: taller boxes for this duck only after it got past one by climbing onto
        # it (not by a commanded flight, nor walking off a box it started on); a fall -> half a step lower
        climbed = info0["climbed"] | first_top
        level = jp.where(newly & info0["obstacle"] & climbed, info0["level"] + OBST_STEP, info0["level"])
        level = jp.where(fallen & info0["obstacle"], info0["level"] - OBST_STEP / 2, level)
        level = jp.clip(level, OBST_START, OBST_TOP)
        plevel = jp.where(newly & puddle, info0["plevel"] + PUD_STEP, info0["plevel"])
        plevel = jp.where(fallen & puddle, info0["plevel"] - PUD_STEP / 2, plevel)
        plevel = jp.clip(plevel, PUD_START, PUD_TOP)

        step = info0["step"] + 1
        seg_end = (step % SEGMENT == 0)
        info = dict(info0)
        info.update(air_time=air_time * ~contact, last_act=action, rng=rng, thrust=thrust,
                    last_target=jp.where(fallen, self._default, self._default.at[self._jq].set(target)),
                    phase=jp.where(fallen, 0.0, phase), single=jp.where(fallen, 0.0, single),
                    last_single=jp.where(fallen, 0.0, last_single), level=level, plevel=plevel, passed=passed,
                    boost=jp.where(fallen, 0.0, boost), td_age=td_age, toff_x=toff_x, topped=topped, rewards=r)
        new_cmd = self.sample_command(cmd_rng, info0["obstacle"], puddle, info0["edge"], info0["top"])
        # a course: well past this obstacle, the next one of the same kind appears ahead
        k_next = jax.random.fold_in(box_rng, 7)
        beyond = passed & (x.pos[0, 0] > jp.where(puddle, pud[0] + pud[1], box[0] + BOX_HALF[0]) + 0.4)
        nxt = beyond & (info0["obstacle"] | puddle) & ~fallen & info0["chain"]
        info["box"] = jp.where(nxt, self._place_box(k_next, info0["obstacle"], level, x0=x.pos[0, 0] + 0.2), box)
        info["pud"] = jp.where(nxt, self._place_puddle(k_next, puddle, plevel, x0=x.pos[0, 0] - 0.2), pud)
        info["passed"] = passed & ~nxt
        info["topped"] = topped & ~nxt
        info["climbed"] = climbed & ~nxt
        info["command"] = jp.where(seg_end, new_cmd, info0["command"])
        info["cmd_now"] = cmd
        # A fall or the time limit ends the episode; Brax restores the first pipeline state (so
        # an airborne-start duck starts in the air again) and the rest is re-drawn here: a new
        # box at this duck's level, a new command, fans off (at hover in the air).
        restart = fallen | (step >= EPISODE)
        air = info0["airborne"]
        info["step"] = jp.where(restart, 0, step)
        info["box"] = jp.where(restart, self._place_box(box_rng, info0["obstacle"], level, info0["edge"], info0["top"],
                                                        info0["top_h"]), info["box"])
        info["pud"] = jp.where(restart, self._place_puddle(box_rng, puddle, plevel), info["pud"])
        info["passed"] = jp.where(restart, False, info["passed"])
        info["topped"] = jp.where(restart, info0["top"], info["topped"])
        info["climbed"] = jp.where(restart, False, info["climbed"])
        restart_cmd = jp.where(air & (jax.random.uniform(box_rng) < 0.5), new_cmd.at[3].set(0.0), new_cmd)
        info["command"] = jp.where(restart, restart_cmd, info["command"])
        info["cmd_now"] = jp.where(restart, jp.where(air, restart_cmd, jp.zeros(6)), cmd)
        info["thrust"] = jp.where(restart, jp.where(air, self._hover, 0.0) * jp.ones(2), info["thrust"])
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v for k, v in r.items()})
        metrics.update(level=level, plevel=plevel, flying=flying.astype(jp.float32), moon=moon.astype(jp.float32),
                       passed=passed.astype(jp.float32), cleared=(passed & puddle).astype(jp.float32),
                       courses=nxt.astype(jp.float32))
        return state.replace(pipeline_state=ps, obs=self._obs(ps, info), reward=reward, done=done, metrics=metrics,
                             info=info)

    def _obs(self, ps, info):
        inv = math.quat_inv(ps.x.rot[0])
        a = 2 * jp.pi * info["phase"]
        box = info["box"]
        p = ps.x.pos[0]
        g = self.ground(p[:2], box)
        # the obstacle ahead (box or puddle) in the trunk's heading frame: near edge, far edge (m),
        # top above the trunk (a puddle's "top" is the floor), and which kind it is
        pud, puddle = info["pud"], info["puddle"]
        cx = jp.where(puddle, pud[0], box[0])
        half = jp.where(puddle, pud[1], BOX_HALF[0])
        top = jp.where(puddle, 0.0, box[1])
        fwd = math.rotate(jp.array([1.0, 0.0, 0.0]), ps.x.rot[0])
        yaw = jp.arctan2(fwd[1], fwd[0])
        ahead = (cx - p[0]) * jp.cos(yaw) - p[1] * jp.sin(yaw)      # the obstacle sits on y = 0
        rel = jp.clip(jp.array([ahead - half, ahead + half, top - p[2]]), -2.0, 2.0)
        return jp.concatenate([
            math.rotate(ps.xd.ang[0], inv) * 0.25,                 # body rates (IMU gyro)
            math.rotate(jp.array([0.0, 0.0, -1.0]), inv),          # gravity direction (IMU)
            ps.q[7:] - self._default,                              # joint angles from home
            ps.qd[6:] * 0.05,                                      # joint velocities
            info["last_act"],
            info["thrust"],                                        # fan thrust (fraction of max)
            info["cmd_now"] * jp.array([2.0, 2.0, 0.5, 2.0, 2.0, 1.0]),
            jp.array([jp.cos(a), jp.sin(a)]),                      # gait clock
            math.rotate(ps.xd.vel[0], inv) * 0.5,                  # body velocity (estimator)
            jp.array([(p[2] - g - self._h0) * 2.0]),               # height above ground (rangefinder)
            rel,                                                   # obstacle ahead (range sensor)
            jp.array([puddle.astype(jp.float32)]),                 # it is water, not a box (camera)
        ])
