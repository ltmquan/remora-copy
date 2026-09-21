#!/usr/bin/env bash
# Sequential jobs: never overlap performance measurements on the same GPUs.
set -euo pipefail
cd "$(dirname "$0")"
out="${1:?Provide a new result root}"
mkdir -p "$out"
common=(--sizes 608 960 1216 1536 --counts 1 2 4 8
  --reference-dir artifacts/references-fp64 --atol 2e-5 --rtol 2e-4
  --weights artifacts/yolov4.weights --setup-timeout 600 --trial-timeout 600
  --continue-on-failure --skip-failed-model --resume --execute
  --exclude-implementations remora-futhark-original remora-futhark-modified)
# First produce the implementation / resolution / GPU-count comparison.
.venv/bin/python -u matrix.py "${common[@]}" --policies uniform \
  --images 128 --trials 5 --out "$out/timing-uniform"
# Memory is measured independently, using at least three seconds per worker.
.venv/bin/python -u matrix.py "${common[@]}" --policies uniform --modes transfer \
  --images 8 --trials 1 --sample-memory --out "$out/memory"
# These policies distribute whole images among model replicas.
.venv/bin/python -u matrix.py "${common[@]}" --counts 2 4 8 \
  --policies round-robin weighted dynamic --images 128 --trials 5 \
  --out "$out/timing-partitions"
date -u > "$out/FINISHED"
