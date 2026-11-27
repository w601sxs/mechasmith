"""Evaluate a trained duck policy on a model of the duck (paper Q3).

    python evaluate.py <policy.pkl> --xml <model.xml> [--episodes 32] [--seconds 10] [--out eval.json]

Q3 trains on a faulty model and evaluates here on the correct one. Each episode starts from the
home pose with small joint noise and holds one command, with no pushes:
  slow     forward 0.15 m/s
  fast     forward 0.30 m/s
  turn     forward 0.10 m/s while turning 0.6 rad/s
Reported per command: fall rate, mean time survived, and the mean absolute error between the
commanded and measured forward speed and turn rate (over the steps before a fall).
Runs on CPU or GPU; all episodes of a command run in parallel.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax import math
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "rl_microduck"))
from duck_env import DuckJoystick  # noqa: E402

COMMANDS = {"slow": (0.15, 0.0, 0.0), "fast": (0.30, 0.0, 0.0), "turn": (0.10, 0.0, 0.6)}


def load_policy(path: str, env: DuckJoystick):
    blob = pickle.loads(Path(path).read_bytes())
    normalizer, policy = (blob["normalizer"], blob["policy"]) if isinstance(blob, dict) else (blob[0], blob[1])
    policy = jax.tree.map(lambda x: jp.asarray(x, jp.float32), policy)
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=(128, 128, 128, 128))
    return ppo_networks.make_inference_fn(nets)((normalizer, policy), deterministic=True)


def rollout_fn(env: DuckJoystick, act, command, steps: int):
    cmd = jp.array(command)

    def episode(key):
        state = env.reset(key)

        def body(carry, _):
            state, alive = carry
            info = dict(state.info, command=cmd, step=jp.array(1))   # hold the command, no pushes
            state = state.replace(info=info)
            action, _ = act(state.obs, key)
            nxt = env.step(state, action)
            ps = nxt.pipeline_state
            rot = ps.x.rot[0]
            v = math.rotate(ps.xd.vel[0], math.quat_inv(rot))
            w = math.rotate(ps.xd.ang[0], math.quat_inv(rot))
            fell = nxt.done > 0
            measured = jp.array([v[0], w[2]])
            return (nxt, alive & ~fell), (measured, alive)

        (_, alive_end), (measured, alive) = jax.lax.scan(body, (state, jp.array(True)), None, length=steps)
        return measured, alive, alive_end

    return jax.jit(jax.vmap(episode))


def evaluate(policy_path: str, xml: str, episodes: int = 32, seconds: float = 10.0, seed: int = 0) -> dict:
    env = DuckJoystick(xml=xml, kick_vel=0.0)
    act = load_policy(policy_path, env)
    steps = int(round(seconds / env.dt))
    out = {"policy": policy_path, "xml": xml, "episodes": episodes, "seconds": seconds, "commands": {}}
    for i, (name, command) in enumerate(COMMANDS.items()):
        keys = jax.random.split(jax.random.PRNGKey(seed * 100 + i), episodes)
        measured, alive, alive_end = jax.device_get(rollout_fn(env, act, command, steps)(keys))
        alive = np.asarray(alive, bool)                       # (episodes, steps): before the fall
        target = np.array([command[0], command[2]])
        err = np.abs(np.asarray(measured) - target)           # (episodes, steps, 2)
        w = alive[..., None]
        mean_err = (err * w).sum(axis=(0, 1)) / np.maximum(w.sum(axis=(0, 1)), 1)
        out["commands"][name] = {
            "command": list(command),
            "fall_rate": float(1 - np.mean(alive_end)),
            "survived_s": float(alive.sum(axis=1).mean() * env.dt),
            "speed_err": float(mean_err[0]),
            "turn_err": float(mean_err[1]),
        }
    c = out["commands"].values()
    out["summary"] = {k: float(np.mean([x[k] for x in c])) for k in ("fall_rate", "survived_s", "speed_err", "turn_err")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = evaluate(a.policy, a.xml, a.episodes, a.seconds, a.seed)
    text = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
