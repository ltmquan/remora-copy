#!/bin/bash
#SBATCH --job-name=remora-python-tune
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=tune-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0
export XDG_CACHE_HOME="$PWD/cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
mkdir -p tuning
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > tuning/gpu.txt
./darknet_benchmark yolov4.cfg tuning/darknet-trained.bin yolov4.weights > tuning/darknet-trained.txt 2> tuning/darknet-trained.stderr
for model in remora direct; do
  timeout 180s venv/bin/python benchmark.py --backend torch --model "$model" --fold-bn --weights yolov4.weights --output "tuning/torch-$model-eager-trained" > "tuning/torch-$model-eager-trained.log" 2>&1 || echo "folded torch $model failed: $?"
done
for model in remora direct; do
  BEAM=2 timeout 480s venv/bin/python benchmark.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "tuning/tinygrad-$model-jit-trained" > "tuning/tinygrad-$model-jit-trained.log" 2>&1 || echo "tuned tinygrad $model failed: $?"
done
venv/bin/python compare.py tuning
