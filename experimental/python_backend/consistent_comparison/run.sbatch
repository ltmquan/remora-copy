#!/bin/bash
#SBATCH --job-name=yolo-consistent-counts
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/consistent-comparison-20260912
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 NVIDIA_TF32_OVERRIDE=0 BEAM=0 PARALLEL=4 PYTHONUNBUFFERED=1
base=/scratch/luu.qu/remora-python-20260911
export XDG_CACHE_HOME="$base/cache"
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
"$base/venv/bin/python" prepare.py
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
for variant in fresh_futhark modified_futhark; do
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$CUDA_ROOT/include" -I"$base/$variant" "$variant/harness.c" "$base/$variant/model.c" -L"$CUDA_ROOT/lib64" -L"$CUDA_ROOT/lib64/stubs" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "$variant/benchmark"
 sha256sum "$base/$variant/model.c" "$variant/harness.c" "$variant/benchmark" > "results/$variant-sha256.txt"
done
for round in 1 2 3; do
 order='fresh_futhark baseline-remora im2col-direct modified_futhark baseline-direct im2col-remora'
 if [[ "$round" == 2 ]]; then order='modified_futhark im2col-remora baseline-direct fresh_futhark im2col-direct baseline-remora'; fi
 for variant in $order; do
  echo "Round $round: $variant"
  if [[ "$variant" == *futhark ]]; then
   timeout --kill-after=10s 180s "$variant/benchmark" "$base/$variant/trained-inputs.bin" "results/$variant-$round.bin" > "results/$variant-$round.txt" 2> "results/$variant-$round.stderr"
  else
   choice=${variant%%-*}; model=${variant#*-}
   TINYGRAD_VARIANT="$choice" timeout --kill-after=10s 120s "$base/venv/bin/python" "$base/investigate_tinygrad.py" --backend tinygrad --model "$model" --mode jit --warmup 5 --runs 30 --weights "$base/yolov4.weights" --output "$PWD/results/$variant-$round" > "results/$variant-$round.log" 2>&1
  fi
 done
done
date -u > results/FINISHED
