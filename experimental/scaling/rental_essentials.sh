#!/usr/bin/env bash
# Budget-reduced continuation: finish existing timing and measure memory endpoints.
set -euo pipefail
cd "$(dirname "$0")"
out=results/full-comparison-20260911
common=(--reference-dir artifacts/references-fp64 --atol 2e-5 --rtol 2e-4
  --weights artifacts/yolov4.weights --setup-timeout 300 --trial-timeout 120
  --continue-on-failure --skip-failed-model --resume --execute
  --sample-memory --policies uniform --modes transfer --counts 1 8
  --images 8 --trials 1)
# Main comparison has already validated these implementations at all four sizes.
.venv/bin/python -u matrix.py "${common[@]}" --sizes 608 960 1216 1536 \
  --implementations remora-torch direct-torch remora-tinygrad-im2col direct-tinygrad-im2col \
  --out "$out/memory-essentials"
# Larger sizes of these baselines failed validation; preserve those failures.
.venv/bin/python -u matrix.py "${common[@]}" --sizes 608 \
  --implementations remora-tinygrad direct-tinygrad darknet-dense \
  --out "$out/memory-baselines"
date -u > "$out/ESSENTIALS_FINISHED"
