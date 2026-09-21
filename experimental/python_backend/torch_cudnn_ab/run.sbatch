#!/bin/bash
#SBATCH --job-name=torch-cudnn-ab
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/torch-cudnn-ab-20260912
module load python/3.13.5 cuda/12.8.0 cuDNN/9.10.2
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
base=/scratch/luu.qu/remora-python-20260911
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
# Separate compile caches prevent reuse between enabled/disabled configurations.
round=0
for order in 'on-remora off-direct off-remora on-direct' 'off-remora on-direct on-remora off-direct' 'on-direct off-remora off-direct on-remora'; do
 round=$((round+1))
 for variant in $order; do
  state=${variant%%-*}; model=${variant#*-}
  export TEST_CUDNN=0
  if [[ "$state" == on ]]; then export TEST_CUDNN=1; fi
  export TORCHINDUCTOR_CACHE_DIR="$PWD/cache-$state" TRITON_CACHE_DIR="$PWD/triton-$state"
  echo "Round $round: $variant"
  timeout --signal=TERM --kill-after=10s 120s "$base/venv/bin/python" run_one.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --warmup 5 --runs 30 --weights "$base/yolov4.weights" --output "$PWD/results/$variant-$round" > "results/$variant-$round.log" 2>&1
 done
done
date -u > results/FINISHED
