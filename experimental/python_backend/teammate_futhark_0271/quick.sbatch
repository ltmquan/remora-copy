#!/bin/bash
#SBATCH --job-name=yolo-futhark-versions
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:08:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/teammate-futhark-0271-20260916
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
futhark=/scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916/futhark-0.27.1-linux-x86_64/bin/futhark
old=/scratch/luu.qu/remora-copy/experimental/teammate-futhark-20260915
base=/scratch/luu.qu/remora-python-20260911
mkdir -p results
"$futhark" --version > results/compiler.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
cmp model.remora "$old/model.remora"

timeout 300s "$futhark" dev --gpu-mem model.fut_soacs > model.fut_gpu_mem 2>results/gpu-memory.log
timeout 120s "$futhark" dev --backend=cuda --library model.fut_gpu_mem > results/cuda.log 2>&1
"$base/venv/bin/python" - <<'PY'
import json
from pathlib import Path
expected=json.loads(Path('inputs.json').read_text())
actual=json.loads(Path('model.json').read_text())['entry_points']['main']['inputs']
assert [v[0] for v in expected]==[v['name'] for v in actual]
assert [len(v[1]) for v in expected]==[v['type'].count('[]') for v in actual]
PY
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
timeout 180s gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" -I. harness-timing.c model.c -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o benchmark-timing
sha256sum model.remora model.fut_soacs model.c harness-timing.c benchmark-timing "$old/benchmark-timing" "$base/fresh_futhark/trained-inputs.bin" > results/hashes.txt
timeout --kill-after=10s 180s ./benchmark-timing "$base/fresh_futhark/trained-inputs.bin" results/new-ramp.bin > results/new-timing.txt 2>&1
"$base/venv/bin/python" analyze-quick.py
date -u > results/FINISHED
