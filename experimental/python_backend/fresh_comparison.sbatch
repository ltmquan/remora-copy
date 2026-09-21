#!/bin/bash
#SBATCH --job-name=remora-fresh-comparison
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=fresh-comparison-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 BEAM=0
export XDG_CACHE_HOME="$PWD/cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
mkdir -p fresh
hostname > fresh/node.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > fresh/gpu.txt
venv/bin/pip freeze > fresh/requirements.lock
cp fresh_futhark/build.json fresh/futhark-build.json
cp fresh_futhark/fixture.json fresh/fixture.json
cp fresh_futhark/binary-sha256.txt fresh/binary-sha256.txt
./darknet_benchmark yolov4.cfg fresh/darknet-trained.bin yolov4.weights > fresh/darknet-trained.txt 2> fresh/darknet-trained.stderr
 timeout 600s fresh_futhark/benchmark fresh_futhark/trained-inputs.bin fresh/futhark-original-trained.bin > fresh/futhark-original-trained.txt 2> fresh/futhark-original-trained.stderr
for model in remora direct; do
  timeout 180s venv/bin/python benchmark.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights yolov4.weights --output "fresh/torch-$model-compile-trained" > "fresh/torch-$model-compile-trained.log" 2>&1
done
for model in remora direct; do
  timeout 180s venv/bin/python benchmark.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "fresh/tinygrad-$model-jit-trained" > "fresh/tinygrad-$model-jit-trained.log" 2>&1
done
venv/bin/python compare.py fresh
