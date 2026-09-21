#!/bin/bash
#SBATCH --job-name=remora-python-graphs
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=graphs-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0
export XDG_CACHE_HOME="$PWD/cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
mkdir -p graphs
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > graphs/gpu.txt
./darknet_benchmark yolov4.cfg graphs/darknet-trained.bin yolov4.weights > graphs/darknet-trained.txt 2> graphs/darknet-trained.stderr
for model in remora direct; do
  timeout 240s venv/bin/python benchmark.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights yolov4.weights --output "graphs/torch-$model-compile-trained" > "graphs/torch-$model-compile-trained.log" 2>&1 || echo "torch CUDA graphs $model failed: $?"
done
venv/bin/python compare.py graphs
