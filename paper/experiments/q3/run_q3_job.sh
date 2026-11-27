#!/usr/bin/env bash
# GPU entry point for the Q3 grid. One GPU per call.
#   bash run_q3_job.sh <first-index> [last-index] [steps]
# runs indices first..last of `python run_q3.py --list` (96 runs: 16 models x 3 seeds x DR off/on).
# Results land in results/<run>.json; existing results are skipped, so re-running resumes.
# About 27 min per run at 3e8 steps on one H100 for the clean duck (examples/rl_microduck/README.md).
set -euo pipefail
FIRST=${1:?first index}
LAST=${2:-$FIRST}
STEPS=${3:-300000000}
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv /tmp/venv
PY=/tmp/venv/bin/python
$PY -m pip install -q "jax[cuda12]==0.7.2" "brax==0.14.2" "mujoco==3.17.0" "mujoco-mjx==3.17.0"
$PY -c "import jax; print(jax.devices())"
for i in $(seq "$FIRST" "$LAST"); do
  $PY run_q3.py --index "$i" --steps "$STEPS"
done
