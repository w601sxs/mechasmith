"""Run the Q3 fault-injection grid: train on each faulty duck, evaluate on the faulty and the correct one.

    python run_q3.py --list                       # the runs, one per line with its index
    python run_q3.py --index 7 [--steps 3e8]      # one run (one GPU); for array jobs
    python run_q3.py --all [--steps 3e8]          # every run in turn
    python run_q3.py --smoke --index 1            # CPU-sized check of the whole path

A run is (model, seed, domain randomisation on/off), where model is the correct duck or one
fault from faults.MICRODUCK_GRID. Each run writes results/<run>.json with both evaluations and is
skipped if that file exists, so an interrupted grid resumes where it stopped.
Run from anywhere; paths are resolved from this file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLE = ROOT / "examples" / "rl_microduck"
CLEAN = EXAMPLE / "duck_mjx.xml"
sys.path.insert(0, str(HERE))
from faults import MICRODUCK_GRID, Fault, inject  # noqa: E402

SEEDS = (0, 1, 2)


def runs() -> list[dict]:
    models = [None, *MICRODUCK_GRID]
    return [{"fault": f, "seed": s, "dr": dr} for dr in (False, True) for f in models for s in SEEDS]


def name(r: dict) -> str:
    return f"{'clean' if r['fault'] is None else r['fault'].tag}-s{r['seed']}{'-dr' if r['dr'] else ''}"


def model_xml(fault: Fault | None, workdir: Path) -> Path:
    if fault is None:
        return CLEAN
    path = workdir / "models" / f"{fault.tag}.xml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inject(str(CLEAN), fault))
    return path


def execute(r: dict, steps: int, envs: int, smoke: bool, workdir: Path) -> Path:
    from evaluate import evaluate   # needs jax/brax; imported here so --list works without them

    out = workdir / "results" / f"{name(r)}.json"
    if out.exists():
        print("skip", out.name, flush=True)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = model_xml(r["fault"], workdir)
    policy = workdir / "policies" / f"{name(r)}.pkl"
    policy.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", "--xml", str(xml), "--out", str(policy), "--seed", str(r["seed"])]
    cmd += ["--smoke"] if smoke else ["--steps", str(steps), "--envs", str(envs)]
    cmd += ["--dr"] if r["dr"] else []
    t0 = time.time()
    log = workdir / "logs" / f"{name(r)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=EXAMPLE, stdout=fh, stderr=subprocess.STDOUT, check=True)
    minutes = (time.time() - t0) / 60
    episodes, seconds = (4, 2.0) if smoke else (64, 10.0)
    res = {
        "run": name(r),
        "fault": None if r["fault"] is None else vars(r["fault"]),
        "seed": r["seed"], "dr": r["dr"], "steps": None if smoke else steps, "train_minutes": round(minutes, 1),
        "on_train_model": evaluate(str(policy), str(xml), episodes, seconds, seed=r["seed"]),
        "on_correct_model": evaluate(str(policy), str(CLEAN), episodes, seconds, seed=r["seed"]),
    }
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(name(r), json.dumps({k: res[k]["summary"] for k in ("on_train_model", "on_correct_model")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--index", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=float, default=3e8)   # what the README's walking result used
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workdir", type=Path, default=HERE)
    a = ap.parse_args()
    rs = runs()
    if a.list:
        for i, r in enumerate(rs):
            print(i, name(r))
        return
    todo = rs if a.all else [rs[a.index]]
    for r in todo:
        execute(r, int(a.steps), a.envs, a.smoke, a.workdir)


if __name__ == "__main__":
    main()
