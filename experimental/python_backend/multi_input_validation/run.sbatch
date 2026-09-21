#!/bin/bash
#SBATCH --job-name=yolo-more-inputs
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/multi-input-validation-20260912
base=/scratch/luu.qu/remora-python-20260911
dark=/scratch/luu.qu/remora-copy/experimental/cudnn-ab-20260912/off
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 NVIDIA_TF32_OVERRIDE=0 BEAM=0 PARALLEL=4 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$base/cache" PYTHONPATH="$base" VALIDATION_ROOT="$PWD"
export TORCHINDUCTOR_CACHE_DIR=/scratch/luu.qu/remora-copy/experimental/torch-cudnn-ab-20260912/cache-off
export TRITON_CACHE_DIR=/scratch/luu.qu/remora-copy/experimental/torch-cudnn-ab-20260912/triton-off
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
"$base/venv/bin/python" prepare_remote.py
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
g++ -O3 -std=c++17 -DGPU -I"$dark/include" -I"$dark/src" -I"$cuda_root/include" darknet.cpp -L"$dark" -Wl,-rpath,"$dark" -ldarknet -L"$cuda_root/lib64" -lcudart -o darknet
for variant in fresh_futhark modified_futhark; do
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" -I"$base/$variant" "$variant/harness.c" "$base/$variant/model.c" -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "$variant/benchmark"
 for name in black white noise gradient checkerboard evalbox; do
  echo "$variant $name"
  # Image is the first ABI argument. Preserve all trained parameter bytes.
  "$base/venv/bin/python" - "$base/$variant/trained-inputs.bin" "inputs/$name.bin" "$variant/input.bin" <<'PY'
import sys,shutil
from pathlib import Path
source,image,out=map(Path,sys.argv[1:]); data=image.read_bytes();assert len(data)==3*608*608*4
with source.open('rb') as src,out.open('wb') as dst:
 dst.write(data);src.seek(len(data));shutil.copyfileobj(src,dst)
PY
  timeout --kill-after=10s 120s "$variant/benchmark" "$variant/input.bin" "results/$variant-$name.bin" > "results/$variant-$name.log" 2>&1
 done
done
for name in black white noise gradient checkerboard evalbox; do
 echo "darknet $name"
 ./darknet "$base/yolov4.cfg" "results/darknet-$name.bin" "$base/yolov4.weights" "inputs/$name.bin" > "results/darknet-$name.log" 2>&1
done
for model in remora direct; do
 echo "torch $model"
 TINYGRAD_VARIANT=baseline timeout --kill-after=10s 180s "$base/venv/bin/python" python_validate.py --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights "$base/yolov4.weights" --output "results/torch-$model" > "results/torch-$model.log" 2>&1
 for variant in baseline im2col; do
  echo "tinygrad $variant $model"
  TINYGRAD_VARIANT="$variant" timeout --kill-after=10s 180s "$base/venv/bin/python" python_validate.py --backend tinygrad --model "$model" --mode jit --weights "$base/yolov4.weights" --output "results/$variant-$model" > "results/$variant-$model.log" 2>&1
 done
done
date -u > results/FINISHED
