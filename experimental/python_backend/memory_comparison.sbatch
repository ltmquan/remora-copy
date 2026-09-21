#!/bin/bash
#SBATCH --job-name=remora-memory-matrix
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=memory-comparison-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 BEAM=0 PARALLEL=4 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$PWD/cache" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" TRITON_CACHE_DIR="$PWD/cache/triton"
mkdir -p memory_matrix/memory
hostname > memory_matrix/node.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > memory_matrix/gpu.txt
venv/bin/pip freeze > memory_matrix/requirements.lock
for variant in fresh_futhark modified_futhark; do
 mkdir -p "memory_matrix/$variant"
 cp "$variant/build.json" "$variant/fixture.json" "$variant/binary-sha256.txt" "memory_matrix/$variant/"
done
# Uninstrumented timings first. Memory sampling is done in separate processes below.
./darknet_benchmark yolov4.cfg memory_matrix/darknet-trained.bin yolov4.weights > memory_matrix/darknet-trained.txt 2> memory_matrix/darknet-trained.stderr
for variant in fresh_futhark modified_futhark; do
 timeout 300s "$variant/benchmark" "$variant/trained-inputs.bin" "memory_matrix/$variant-trained.bin" > "memory_matrix/$variant-trained.txt" 2> "memory_matrix/$variant-trained.stderr"
done
for model in remora direct; do
 timeout 180s venv/bin/python benchmark.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights yolov4.weights --output "memory_matrix/torch-$model-compile-trained" > "memory_matrix/torch-$model-compile-trained.log" 2>&1
 timeout 180s venv/bin/python benchmark.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "memory_matrix/tinygrad-$model-jit-trained" > "memory_matrix/tinygrad-$model-jit-trained.log" 2>&1
done
venv/bin/python compare.py memory_matrix
venv/bin/python measure_memory.py --output memory_matrix/memory/darknet -- ./darknet_benchmark yolov4.cfg memory_matrix/memory/darknet-trained.bin yolov4.weights
for variant in fresh_futhark modified_futhark; do
 timeout 300s venv/bin/python measure_memory.py --output "memory_matrix/memory/$variant" -- "$variant/benchmark" "$variant/trained-inputs.bin" "memory_matrix/memory/$variant-trained.bin"
done
for model in remora direct; do
 timeout 180s venv/bin/python measure_memory.py --output "memory_matrix/memory/torch-$model" -- venv/bin/python benchmark.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights yolov4.weights --output "memory_matrix/memory/torch-$model-compile-trained"
 timeout 180s venv/bin/python measure_memory.py --output "memory_matrix/memory/tinygrad-$model" -- venv/bin/python benchmark.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "memory_matrix/memory/tinygrad-$model-jit-trained"
done
venv/bin/python compare.py memory_matrix/memory
