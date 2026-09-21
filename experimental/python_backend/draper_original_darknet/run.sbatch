#!/bin/bash
#SBATCH --job-name=draper-original-darknet
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:12:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/draper-original-darknet-20260916
module load python/3.13.5 cuda/12.8.0 OpenCV/4.10.0
# Keep the source's precision and threading defaults; Slurm provides four CPU cores.
unset NVIDIA_TF32_OVERRIDE ALLOW_TF32 OMP_NUM_THREADS || true
mkdir -p results pkgconfig
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
export PKG_CONFIG_PATH="$PWD/pkgconfig:${PKG_CONFIG_PATH:-}"
cat > pkgconfig/opencv4.pc <<'PC'
prefix=/shared/EL9/explorer/OpenCV/4.10.0
libdir=${prefix}/lib64
includedir=${prefix}/include/opencv4
Name: OpenCV
Description: Cluster OpenCV library paths
Version: 4.10.0
Libs: -L${libdir} -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_calib3d -lopencv_video -lopencv_dnn
Cflags: -I${includedir}
PC
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
gcc --version > results/compiler.txt
nvcc --version > results/cuda.txt
pkg-config --cflags --libs opencv4 > results/opencv.txt
ln -sf /scratch/luu.qu/remora-python-20260911/yolov4.weights darknet/yolov4.weights
# Preserve original GPU/CUDNN/AVX/OPENMP/OPENCV/ARCH defaults.
# Replace unavailable Clang toolchain with GCC; CUDA/OpenCV paths provided by modules.
timeout --kill-after=10s 540s make -C darknet -j4 libdarknet.so CC=gcc CPP='g++ -std=c++11' NVCC='nvcc -ccbin g++' > results/build.log 2>&1
g++ -std=c++17 -O3 -Wall -I"$cuda_root/include" -Idarknet/include -Idarknet/src $(pkg-config --cflags opencv4) yolov4.cc -L"$PWD/darknet" -Wl,-rpath,"$PWD/darknet" -ldarknet -L"$cuda_root/lib64" -lcuda -lcudart -lcublas -lcurand -lnvidia-ml -lpthread -lm $(pkg-config --libs opencv4) -o yolov4 > results/harness-build.log 2>&1
sha256sum darknet/Makefile darknet/libdarknet.so darknet/cfg/yolov4.cfg darknet/data/dog.jpg yolov4.cc yolov4 > results/hashes.txt
ldd yolov4 > results/linked.txt
# Exactly the original script's GPU command; default dog.jpg and one warmup.
/usr/bin/time -v timeout --kill-after=5s 90s ./yolov4 -i 10 > results/run.log 2>&1
date -u > results/FINISHED
