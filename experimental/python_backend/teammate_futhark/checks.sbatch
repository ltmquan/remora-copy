#!/bin/bash
#SBATCH --job-name=yolo-teammate-checks
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=checks-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/teammate-futhark-20260915
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
base=/scratch/luu.qu/remora-python-20260911
inputs=/scratch/luu.qu/remora-copy/experimental/multi-input-validation-20260912/inputs
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/checks-gpu.txt
for name in black white noise gradient checkerboard evalbox; do
 echo "Correctness $name"
 "$base/venv/bin/python" - "$base/fresh_futhark/trained-inputs.bin" "$inputs/$name.bin" input.bin <<'PY'
import sys,shutil
from pathlib import Path
source,image,out=map(Path,sys.argv[1:]);data=image.read_bytes();assert len(data)==3*608*608*4
with source.open('rb') as src,out.open('wb') as dst:
 dst.write(data);src.seek(len(data));shutil.copyfileobj(src,dst)
PY
 timeout --kill-after=10s 180s ./benchmark-check input.bin "results/$name.bin" > "results/$name.log" 2>&1
done
"$base/venv/bin/python" "$base/measure_memory.py" --output results/memory -- ./benchmark-timing "$base/fresh_futhark/trained-inputs.bin" results/memory-ramp.bin
date -u > results/FINISHED
