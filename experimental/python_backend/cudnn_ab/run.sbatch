#!/bin/bash
#SBATCH --job-name=darknet-cudnn-ab
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/cudnn-ab-20260912
module load python/3.13.5 cuda/12.8.0 cuDNN/9.10.2
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
base=/scratch/luu.qu/remora-python-20260911
prior=/scratch/luu.qu/remora-copy/experimental/cudnn-comparison-20260912
py="$base/venv/bin/python"
mkdir -p results off
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
hostname > results/node.txt
cp -a "$prior/darknet" heuristic
cp -a "$prior/darknet" search
tar -xzf "$prior/darknet-source.tar.gz" -C off
"$py" prepare.py search
# A single identical harness for all variants: five warmups, thirty timed calls.
"$py" - "$base/darknet_benchmark.cpp" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
assert s.count('for(int run=-1;run<10;run++){')==1
Path('benchmark.cpp').write_text(s.replace('for(int run=-1;run<10;run++){','for(int run=-5;run<30;run++){'))
PY
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
for variant in off heuristic search; do
  cudnn=1; defs=-DCUDNN
  if [[ "$variant" == off ]]; then cudnn=0; defs=; fi
  make -C "$variant" -j4 libdarknet.so CC=gcc CPP=g++ NVCC="nvcc -ccbin g++" GPU=1 CUDNN="$cudnn" CUDNN_HALF=0 OPENCV=0 AVX=0 OPENMP=0 LIBSO=1 'ARCH=-gencode arch=compute_90,code=sm_90' > "results/build-$variant.log" 2>&1
  g++ -O3 -std=c++17 -DGPU $defs -I"$variant/include" -I"$variant/src" -I"$cuda_root/include" benchmark.cpp -L"$PWD/$variant" -Wl,-rpath,"$PWD/$variant" -ldarknet -L"$cuda_root/lib64" -lcudart -o "benchmark-$variant"
  sha256sum "$variant/libdarknet.so" "$variant/src/convolutional_layer.c" "benchmark-$variant" > "results/sha256-$variant.txt"
  ldd "benchmark-$variant" > "results/linked-$variant.txt"
done
printf '%s\n' 'FP32 NVIDIA_TF32_OVERRIDE=0 CUDNN_HALF=0 OPENMP=0; same harness, 5 warmups, 30 calls per process; forward-only search retains candidate/workspace filters' > results/config.txt
# Latin-square order: each variant appears first, second, and third once.
round=0
for order in 'off heuristic search' 'heuristic search off' 'search off heuristic'; do
  round=$((round+1))
  for variant in $order; do
    echo "Round $round: $variant"
    set +e
    timeout --signal=TERM --kill-after=10s 120s "./benchmark-$variant" "$base/yolov4.cfg" "results/$variant-$round.bin" "$base/yolov4.weights" > "results/$variant-$round.txt" 2> "results/$variant-$round.stderr"
    code=$?
    set -e
    echo "$code" > "results/$variant-$round.exit"
    if [[ "$code" != 0 ]]; then echo "Failed $variant: $code"; exit "$code"; fi
  done
done
date -u > results/FINISHED
