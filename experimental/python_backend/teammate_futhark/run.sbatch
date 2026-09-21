#!/bin/bash
#SBATCH --job-name=yolo-teammate
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/teammate-futhark-20260915
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
base=/scratch/luu.qu/remora-python-20260911
inputs=/scratch/luu.qu/remora-copy/experimental/multi-input-validation-20260912/inputs
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
for name in timing check; do
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" -I. "harness-$name.c" model.c -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "benchmark-$name"
done
sha256sum model.remora model.c benchmark-* > results/build-sha256.txt
# ABI order must match the existing trained stream exactly.
"$base/venv/bin/python" - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('inputs.json').read_text())==json.loads(Path('/scratch/luu.qu/remora-python-20260911/fresh_futhark/inputs.json').read_text())
PY
for round in 1 2 3; do
 echo "Timing round $round"
 timeout --kill-after=10s 180s ./benchmark-timing "$base/fresh_futhark/trained-inputs.bin" "results/ramp-$round.bin" > "results/timing-$round.txt" 2>&1
done
for name in black white noise gradient checkerboard evalbox; do
 echo "Correctness $name"
 "$base/venv/bin/python" - "$base/fresh_futhark/trained-inputs.bin" "$inputs/$name.bin" input.bin <<'PY'
import sys,shutil
from pathlib import Path
source,image,out=map(Path,sys.argv[1:]);data=image.read_bytes();assert len(data)==3*608*608*4
with source.open('rb') as src,out.open('wb') as dst:
 dst.write(data);src.seek(len(data));shutil.copyfileobj(src,dst)
PY
 timeout --kill-after=10s 90s ./benchmark-check input.bin "results/$name.bin" > "results/$name.log" 2>&1
done
"$base/venv/bin/python" "$base/measure_memory.py" --output results/memory -- ./benchmark-timing "$base/fresh_futhark/trained-inputs.bin" results/memory-ramp.bin
date -u > results/FINISHED
