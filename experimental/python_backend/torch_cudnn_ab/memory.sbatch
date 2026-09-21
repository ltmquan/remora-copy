#!/bin/bash
#SBATCH --job-name=yolo-cudnn-off-memory
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=memory-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/torch-cudnn-ab-20260912
module load python/3.13.5 cuda/12.8.0 cuDNN/9.10.2
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 TEST_CUDNN=0
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache-off" TRITON_CACHE_DIR="$PWD/triton-off"
base=/scratch/luu.qu/remora-python-20260911
dark=/scratch/luu.qu/remora-copy/experimental/cudnn-ab-20260912
out="$PWD/memory-off"
mkdir -p "$out"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out/gpu.txt"
"$base/venv/bin/python" "$base/measure_memory.py" --output "$out/darknet" -- "$dark/benchmark-off" "$base/yolov4.cfg" "$out/darknet.bin" "$base/yolov4.weights"
for model in remora direct; do
 "$base/venv/bin/python" "$base/measure_memory.py" --output "$out/torch-$model" -- "$base/venv/bin/python" run_one.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --warmup 5 --runs 30 --weights "$base/yolov4.weights" --output "$out/off-$model"
done
date -u > "$out/FINISHED"
