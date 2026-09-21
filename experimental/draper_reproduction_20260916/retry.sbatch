#!/bin/bash
#SBATCH --job-name=draper-demo-retry
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:08:00
#SBATCH --output=retry-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916
module load cuda/12.8.0
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
export PATH="$PWD/futhark-0.27.1-linux-x86_64/bin:$PATH"
cp -n results/bench.log results/bench-unpatched.log
cp -n results/bench.json results/bench-unpatched.json
cd upstream
cp -n yolov4.c yolov4-unpatched.c
sed -i 's/int max_num_tokens = 1000;/int max_num_tokens = 4096;/' yolov4.c
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
timeout 180s gcc -O3 -std=c99 -Wno-unused-function -I"$cuda_root/include" yolov4.c -o yolov4 -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -lcuda -lcudart -lnvrtc -lm > ../results/recompile.log 2>&1
/usr/bin/time -v timeout --kill-after=10s 240s futhark bench --skip-compilation --no-convergence-phase --runs=30 --timeout=210 --json=../results/bench.json yolov4.fut > ../results/bench.log 2>&1
date -u > ../results/FINISHED
