"""Train the jet-pack duck (walk, hop, fly, land: one policy) with Brax PPO on MJX.

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

from jet_env import EPISODE, JetDuck
from scorecard import make_scorecard

try:
    from train_report import report
except ImportError:
    def report(kind, payload):
        print(kind, json.dumps(payload)[:300])

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=150_000_000)
ap.add_argument("--envs", type=int, default=8192)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--gamma", type=float, default=0.99, help="discount per 12 ms step (0.97 = ~0.4 s horizon, 0.99 = ~1.2 s)")
ap.add_argument("--s3", default=None, help="s3://bucket/prefix for resumable checkpoints (one per evaluation)")
ap.add_argument("--init", default=None, help="s3://.../ckpt.pkl to warm-start from when there is no checkpoint to resume")
a = ap.parse_args()
if a.smoke:
    a.steps, a.envs = 20_000, 32

print("devices", jax.devices(), flush=True)
report("start", {"devices": str(jax.devices()), "steps": a.steps, "envs": a.envs})
env = JetDuck()
t0 = time.time()

# Resume: Scale Train restarts a preempted job from the top with the same command, so each
# evaluation writes (normaliser, policy, value) and the step count to S3, and a restart picks
# up from the last one (the optimiser state and the per-duck curricula start over).
CKPT, DONE, RESTORE = None, 0, None
if a.s3:
    try:
        import boto3
        bucket, prefix = a.s3[5:].split("/", 1)
        s3 = boto3.client("s3")
        CKPT = (bucket, prefix.rstrip("/") + "/ckpt.pkl")
        try:
            s3.download_file(*CKPT, "/tmp/ckpt.pkl")
            with open("/tmp/ckpt.pkl", "rb") as f:
                ck = pickle.load(f)
            DONE, RESTORE = int(ck["step"]), ck["params"]
            report("resume", {"from_step": DONE, "uri": a.s3})
        except Exception as ex:   # no checkpoint yet: a fresh start (or warm, from --init)
            report("fresh", {"uri": a.s3, "note": str(ex)[:200]})
            if a.init:
                try:
                    ib, ik = a.init[5:].split("/", 1)
                    s3.download_file(ib, ik, "/tmp/init.pkl")
                    with open("/tmp/init.pkl", "rb") as f:
                        RESTORE = pickle.load(f)["params"]
                    report("init", {"uri": a.init})
                except Exception as ex2:
                    report("init_failed", {"error": str(ex2)[:300]})
        s3.put_object(Bucket=bucket, Key=prefix.rstrip("/") + "/_probe.txt", Body=b"ok")
    except Exception as ex:
        report("s3_failed", {"error": str(ex)[:300]})
        CKPT = None
REMAINING = max(a.steps - DONE, 0)


TRAIN = {}
CARD = {}    # the scorecard, built on the first evaluation   # curriculum numbers from the training metrics (obstacle level, share flying, boxes passed)


def progress(step, metrics):
    n = max(float(metrics.get("episode/length", EPISODE)), 1)
    for k in ("level", "plevel", "flying", "moon", "passed", "cleared", "courses"):
        if f"episode/{k}" in metrics:
            TRAIN[k] = round(float(metrics[f"episode/{k}"]) / n, 3)
    if not any(k.startswith("eval/") for k in metrics):
        return   # a training-metrics call: only remember the speed ramp
    m = {k: float(v) for k, v in metrics.items() if k.startswith("eval/") and np.ndim(v) == 0}
    m = {k: round(v, 4) for k, v in m.items() if "std" not in k}
    m.update({f"train/{k}": v for k, v in TRAIN.items()})
    m["step"], m["minutes"] = int(step) + DONE, round((time.time() - t0) / 60, 2)
    print(json.dumps(m), flush=True)
    report("progress", m)


def send_policy(kind, params, **extra):
    """Report the observation normaliser and policy (half precision) in base64 chunks."""
    normalizer, policy = jax.device_get(params[0]), jax.device_get(params[1])
    blob = pickle.dumps({"normalizer": jax.tree.map(np.asarray, normalizer),
                         "policy": jax.tree.map(lambda x: np.asarray(x, np.float16), policy)})
    text = base64.b64encode(blob).decode()
    parts = [text[i:i + 60_000] for i in range(0, len(text), 60_000)]
    for i, p in enumerate(parts):
        report(kind, {"i": i, "n": len(parts), "data": p, **extra})
    return len(blob), len(parts)


def snapshot(step, make_policy, params):   # a policy at every evaluation, for mid-run GIFs
    step = int(step) + DONE
    send_policy("snapshot_chunk", params, step=step)
    try:
        if "fn" not in CARD:
            CARD["fn"] = make_scorecard(JetDuck(kick_vel=0.0), make_policy, steps=200 if a.smoke else 700)
        sc = CARD["fn"](params)
        print(json.dumps({"scorecard": sc, "step": step}), flush=True)
        report("scorecard", {"step": step, **sc})
    except Exception as ex:
        report("scorecard_failed", {"error": str(ex)[:300]})
    print(json.dumps({"snapshot": step}), flush=True)
    if CKPT and int(step) > DONE:
        try:
            with open("/tmp/ckpt_new.pkl", "wb") as f:
                pickle.dump({"step": step, "params": jax.tree.map(np.asarray, jax.device_get(params))}, f)
            s3.upload_file("/tmp/ckpt_new.pkl", *CKPT)
            report("ckpt", {"step": step})
        except Exception as ex:
            report("ckpt_failed", {"error": str(ex)[:300]})


make_networks = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(256, 256, 128, 128))
make_policy, params, _ = ppo.train(
    environment=env, num_timesteps=REMAINING, num_evals=4 if a.smoke else max(2, round(25 * REMAINING / a.steps)), episode_length=EPISODE,
    normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=4 if a.smoke else 32,
    num_updates_per_batch=4, discounting=a.gamma, learning_rate=3e-4, entropy_cost=1e-2, num_envs=a.envs,
    batch_size=16 if a.smoke else 256, max_grad_norm=1.0, network_factory=make_networks, seed=0,
    progress_fn=progress, policy_params_fn=snapshot, restore_params=RESTORE, restore_value_fn=RESTORE is not None, log_training_metrics=True, training_metrics_steps=5_000_000, num_eval_envs=8 if a.smoke else 128)

with open("policy.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)

size, n = send_policy("policy_chunk", params)
report("done", {"minutes": round((time.time() - t0) / 60, 2), "policy_bytes": size, "chunks": n})
print("done", size, "bytes in", n, "chunks", flush=True)
