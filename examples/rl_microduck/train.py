"""Train the duck to walk on command with Brax PPO on MJX.

usage: python3 train.py [--steps 150000000] [--envs 8192] [--smoke]
--smoke runs a tiny CPU-sized job to check that everything wires up.
Writes policy.pkl (Brax params) and reports progress and the policy through train_report
when it runs on Scale Train.
"""
import argparse
import base64
import functools
import json
import pickle
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from duck_env import DuckJoystick

try:
    from train_report import report
except ImportError:
    def report(kind, payload):
        print(kind, json.dumps(payload)[:300])

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=150_000_000)
ap.add_argument("--envs", type=int, default=8192)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.steps, a.envs = 20_000, 32

print("devices", jax.devices(), flush=True)
report("start", {"devices": str(jax.devices()), "steps": a.steps, "envs": a.envs})
env = DuckJoystick()
t0 = time.time()


def progress(step, metrics):
    m = {k: float(v) for k, v in metrics.items() if k.startswith("eval/") and np.ndim(v) == 0}
    m = {k: round(v, 4) for k, v in m.items() if "std" not in k}
    m["step"], m["minutes"] = int(step), round((time.time() - t0) / 60, 2)
    print(json.dumps(m), flush=True)
    report("progress", m)


make_networks = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128))
make_policy, params, _ = ppo.train(
    environment=env, num_timesteps=a.steps, num_evals=4 if a.smoke else 15, episode_length=1000,
    normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=4 if a.smoke else 32,
    num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=a.envs,
    batch_size=16 if a.smoke else 256, max_grad_norm=1.0, network_factory=make_networks, seed=0,
    progress_fn=progress, num_eval_envs=8 if a.smoke else 128)

with open("policy.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)

# Send the policy back in chunks (half precision; the observation normaliser and policy only).
normalizer, policy = jax.device_get(params[0]), jax.device_get(params[1])
blob = pickle.dumps({"normalizer": jax.tree.map(np.asarray, normalizer),
                     "policy": jax.tree.map(lambda x: np.asarray(x, np.float16), policy)})
text = base64.b64encode(blob).decode()
chunk = 60_000
parts = [text[i:i + chunk] for i in range(0, len(text), chunk)]
for i, p in enumerate(parts):
    report("policy_chunk", {"i": i, "n": len(parts), "data": p})
report("done", {"minutes": round((time.time() - t0) / 60, 2), "policy_bytes": len(blob), "chunks": len(parts)})
print("done", len(blob), "bytes in", len(parts), "chunks", flush=True)
