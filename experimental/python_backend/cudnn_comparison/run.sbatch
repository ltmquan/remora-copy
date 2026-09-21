#!/bin/bash
#SBATCH --job-name=yolo-cudnn-match
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/cudnn-comparison-20260912
module load python/3.13.5 cuda/12.8.0 cuDNN/9.10.2
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
base=/scratch/luu.qu/remora-python-20260911
py="$base/venv/bin/python"
mkdir -p results/memory
hostname > results/node.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
module list > results/modules.txt 2>&1
"$py" -c 'import torch; print(torch.__version__,torch.backends.cudnn.version(),torch.backends.cudnn.enabled)' > results/torch-version.txt
tar -xzf darknet-source.tar.gz -C darknet
make -C darknet -j4 libdarknet.so CC=gcc CPP=g++ NVCC="nvcc -ccbin g++" GPU=1 CUDNN=1 CUDNN_HALF=0 OPENCV=0 AVX=0 OPENMP=0 LIBSO=1 'ARCH=-gencode arch=compute_90,code=sm_90' > results/build.log 2>&1
printf '%s\n' 'GPU=1 CUDNN=1 CUDNN_HALF=0 OPENCV=0 AVX=0 OPENMP=0 LIBSO=1 NVIDIA_TF32_OVERRIDE=0 sm_90' > results/build-config.txt
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
g++ -O3 -std=c++17 -DGPU -DCUDNN -Idarknet/include -Idarknet/src -I"$cuda_root/include" "$base/darknet_benchmark.cpp" -L"$PWD/darknet" -Wl,-rpath,"$PWD/darknet" -ldarknet -L"$cuda_root/lib64" -lcudart -o darknet_benchmark
sha256sum darknet-source.tar.gz darknet/libdarknet.so darknet_benchmark "$base/yolov4.weights" > results/sha256.txt
ldd darknet/libdarknet.so > results/linked-libraries.txt
./darknet_benchmark "$base/yolov4.cfg" results/darknet-trained.bin "$base/yolov4.weights" > results/darknet-trained.txt 2> results/darknet-trained.stderr
for model in remora direct; do
  "$py" "$base/benchmark.py" --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights "$base/yolov4.weights" --output "$PWD/results/torch-$model-compile-trained" > "results/torch-$model.log" 2>&1
done
"$py" "$base/compare.py" results > results/validation.log
"$py" -c 'import json; d=json.load(open("results/validation.json")); assert d and all(v["passed"] for v in d.values()),d'
"$py" "$base/measure_memory.py" --output results/memory/darknet -- ./darknet_benchmark "$base/yolov4.cfg" results/memory/darknet-trained.bin "$base/yolov4.weights"
for model in remora direct; do
  "$py" "$base/measure_memory.py" --output "results/memory/torch-$model" -- "$py" "$base/benchmark.py" --backend torch --model "$model" --mode compile --compile-mode reduce-overhead --fold-bn --weights "$base/yolov4.weights" --output "$PWD/results/memory/torch-$model-compile-trained"
done
date -u > results/FINISHED
