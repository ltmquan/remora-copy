#!/bin/bash
#SBATCH --job-name=draper-demo-repro
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916
module load cuda/12.8.0
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
export PATH="$PWD/futhark-0.27.1-linux-x86_64/bin:$PATH"
mkdir -p results
futhark --version > results/compiler.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
cd upstream
sha256sum yolov4* > ../results/inputs-sha256.txt
/usr/bin/time -v timeout --kill-after=10s 600s futhark dev --gpu-mem --backend=cuda --server yolov4.fut_soacs > ../results/compile.log 2>&1
/usr/bin/time -v timeout --kill-after=10s 480s futhark bench --skip-compilation --timeout=360 --json=../results/bench.json yolov4.fut > ../results/bench.log 2>&1
date -u > ../results/FINISHED
