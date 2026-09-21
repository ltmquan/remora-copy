#!/bin/bash
#SBATCH --job-name=remora-python-yolo
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4
export DEV=CUDA
export BEAM=0
export XDG_CACHE_HOME="$PWD/cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
venv/bin/pip freeze > results/requirements.lock
DARK=/scratch/luu.qu/darknet-comparison-20260910/darknet
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
g++ -O2 -std=c++17 -DGPU -I"$DARK/include" -I"$DARK/src" -I"$CUDA_ROOT/include" darknet_benchmark.cpp -L"$DARK" -Wl,-rpath,"$DARK" -ldarknet -L"$CUDA_ROOT/lib64" -lcudart -o darknet_benchmark
venv/bin/python -m unittest test_runtime -v > results/operator-tests.log 2>&1
for fixture in synthetic trained; do
  extra=()
  if [[ "$fixture" == trained ]]; then
    echo "e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562  yolov4.weights" | sha256sum -c -
    extra=(--weights yolov4.weights)
    ./darknet_benchmark yolov4.cfg "results/darknet-$fixture.bin" yolov4.weights > "results/darknet-$fixture.txt" 2> "results/darknet-$fixture.stderr"
  else
    ./darknet_benchmark yolov4.cfg "results/darknet-$fixture.bin" > "results/darknet-$fixture.txt" 2> "results/darknet-$fixture.stderr"
  fi
  for model in remora direct; do
    timeout 240s venv/bin/python benchmark.py --backend torch --model "$model" "${extra[@]}" --output "results/torch-$model-eager-$fixture" > "results/torch-$model-eager-$fixture.log" 2>&1 || echo "torch $model $fixture failed: $?"
  done
  for model in remora direct; do
    timeout 420s venv/bin/python benchmark.py --backend tinygrad --model "$model" --mode jit "${extra[@]}" --output "results/tinygrad-$model-jit-$fixture" > "results/tinygrad-$model-jit-$fixture.log" 2>&1 || echo "tinygrad $model $fixture failed: $?"
  done
  venv/bin/python compare.py results
 done
for model in remora direct; do
  timeout 300s venv/bin/python benchmark.py --backend torch --model "$model" --mode compile --fold-bn --weights yolov4.weights --output "results/torch-$model-compile-trained" > "results/torch-$model-compile-trained.log" 2>&1 || echo "torch compile $model failed: $?"
done
venv/bin/python compare.py results
