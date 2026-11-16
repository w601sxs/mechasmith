#!/usr/bin/env bash
# Scale Train entry point: install, check the GPU, train. Each step reports its outcome (and the
# tail of its log on failure) through train_report, so the result is readable without Datadog.
set -uo pipefail
STEPS=${1:-150000000}
ENVS=${2:-8192}
S3=${3:-}            # s3://bucket/prefix: resumable checkpoints
rep() {   # rep <stage> <logfile> [ok|error]
  python3 - "$1" "$2" "${3:-ok}" <<'EOF'
import json, sys
from train_report import report
stage, log, kind = sys.argv[1:4]
try:
    text = open(log).read()[-3000:]
except OSError:
    text = ""
report(kind, {"stage": stage, "log": text})
EOF
}
run() {   # run <stage> <command...>
  local stage=$1; shift
  echo "[run_job] $stage: $*"
  if "$@" 2>&1 | tee "/tmp/$stage.log"; then rep "$stage" "/tmp/$stage.log" ok
  else rep "$stage" "/tmp/$stage.log" error; exit 1; fi
}
run gpu nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# A fresh virtualenv: the image's system packages (debian-installed) can't be upgraded in place.
run venv python3 -m venv /tmp/venv
PY=/tmp/venv/bin/python
run install $PY -m pip install -q "jax[cuda12]==0.7.2" "brax==0.14.2" "mujoco==3.17.0" "mujoco-mjx==3.17.0" boto3
run jax $PY -c "import jax; print(jax.devices()); import jax.numpy as jp; print(float(jp.ones(3).sum()))"
run train $PY train.py --steps "$STEPS" --envs "$ENVS" ${S3:+--s3 "$S3"}
