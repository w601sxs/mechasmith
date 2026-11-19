"""Fixed test courses for a jet-duck policy: did it get past each obstacle without a fall?

Run inside training at every evaluation (train.py reports it), or locally:
    python scorecard.py policy.b64
"""
import jax
import jax.numpy as jp
import numpy as np
from brax.envs.base import State

from jet_env import BOX_HALF, JetDuck

WALK, TAKEOFF, RUN = (0.25, 0, 0, 0, 0, 0), (0.4, 0, 0, 0.5, 0, 0), (0.8, 0, 0, 0, 0, 0)
# (name, kind, size m, command): kind 0 walk in to a box, 1 start at its near edge, 2 start on top, 3 puddle
SCENARIOS = [("box10", 0, 0.10, WALK), ("box20", 0, 0.20, WALK), ("box30", 0, 0.30, WALK), ("box45", 0, 0.45, WALK),
             ("edge20", 1, 0.20, WALK), ("edge30_fly", 1, 0.30, TAKEOFF), ("edge50_fly", 1, 0.50, TAKEOFF),
             ("top20_off", 2, 0.20, WALK), ("top35_off", 2, 0.35, WALK),
             ("pud10", 3, 0.10, RUN), ("pud20", 3, 0.20, RUN), ("pud30", 3, 0.30, RUN), ("pud45", 3, 0.45, RUN)]


def make_scorecard(env, make_policy, steps=700, seeds=3):
    kind = jp.array([s[1] for s in SCENARIOS])
    size = jp.array([s[2] for s in SCENARIOS])
    cmds = jp.array([s[3] for s in SCENARIOS], jp.float32)

    def start(rng, k, L, cmd):
        st = env.reset(rng)
        is_box, is_pud = k < 3, k == 3
        bx = jp.where(k == 0, 0.9, jp.where(k == 1, BOX_HALF[0] + 0.15, 0.15 - BOX_HALF[0]))
        box = jp.where(is_box, jp.array([bx, L]), jp.array([50.0, 0.0]))
        pud = jp.where(is_pud, jp.array([1.2 + L / 2, L / 2]), jp.array([60.0, 0.1]))
        q = env._home.at[2].add(jp.where(k == 2, L, 0.0))
        ps = env.pipeline_init(q, jp.zeros(env.sys.nv))
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[0].set(jp.array([box[0], 0.0, box[1] - BOX_HALF[2]])))
        info = dict(st.info, obstacle=is_box, puddle=is_pud, airborne=jp.array(False), box=box, pud=pud,
                    edge=k == 1, top=k == 2, top_h=L, topped=k == 2, climbed=jp.array(False), chain=jp.array(False), passed=jp.array(False), command=cmd,
                    cmd_now=jp.zeros(6), thrust=jp.zeros(2), boost=jp.zeros(()), step=jp.array(1))
        return State(ps, env._obs(ps, info), st.reward, st.done, st.metrics, info)

    def run(params):
        act = make_policy(params, deterministic=True)

        def one(seed, k, L, cmd):
            def body(c, _):
                s, ok, fell, top = c
                s = s.replace(info=dict(s.info, command=cmd, step=jp.array(1)))
                a, _ = act(s.obs, jax.random.PRNGKey(0))
                s = env.step(s, a)
                fell = fell | (s.done > 0)
                ok = ok | (s.info["passed"] & ~fell)
                return (s, ok, fell, jp.maximum(top, s.pipeline_state.x.pos[0, 2])), None
            s0 = start(jax.random.PRNGKey(seed), k, L, cmd)
            (s, ok, fell, top), _ = jax.lax.scan(body, (s0, jp.array(False), jp.array(False), jp.zeros(())), None, steps)
            return ok, fell, top, s.pipeline_state.x.pos[0, 0]
        sd = jp.repeat(jp.arange(1, seeds + 1), len(SCENARIOS))
        t = lambda v: jp.tile(v, (seeds,) + (1,) * (v.ndim - 1))
        return jax.vmap(one)(sd, t(kind), t(size), t(cmds))

    run = jax.jit(run)

    def card(params):
        n = len(SCENARIOS)
        ok, fell, top, x = (np.asarray(v).reshape(seeds, n) for v in jax.device_get(run(params)))
        return {name: f"pass {ok[:, i].sum()}/{seeds} fall {(fell[:, i] & ~ok[:, i]).sum()} z{top[:, i].max():.2f}"
                for i, (name, *_) in enumerate(SCENARIOS)}
    return card


if __name__ == "__main__":
    import base64, pickle, sys
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    raw = open(sys.argv[1], "rb").read()
    blob = pickle.loads(base64.b64decode(raw) if sys.argv[1].endswith(".b64") else raw)
    params = (blob["normalizer"], jax.tree.map(lambda x: np.asarray(x, np.float32), blob["policy"])) \
        if isinstance(blob, dict) else (blob[0], blob[1])
    env = JetDuck(kick_vel=0.0)
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=(256, 256, 128, 128))
    card = make_scorecard(env, ppo_networks.make_inference_fn(nets), steps=int(sys.argv[2]) if len(sys.argv) > 2 else 700)
    for k, v in card(params).items():
        print(f"{k:12s} {v}")
