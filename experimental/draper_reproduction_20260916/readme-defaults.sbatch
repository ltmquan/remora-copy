#!/bin/bash
#SBATCH --job-name=draper-readme-defaults
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=readme-defaults-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916
module load cuda/12.8.0
export PATH="$PWD/futhark-0.27.1-linux-x86_64/bin:$PATH"
mkdir -p readme-defaults/results
cd readme-defaults
# Original benchmark specification, with original published filenames.
git -C ../upstream show HEAD:yolov4.fut > yolov4.fut
ln -sf ../upstream/yolov4 yolov4
ln -sf ../upstream/yolov4_input.bin yolov4_input.bin
ln -sf ../upstream/yolov4_expected.bin yolov4_expected.bin
sha256sum yolov4_input.bin yolov4_expected.bin ../upstream/yolov4.fut_soacs > results/hashes.txt
futhark --version > results/compiler.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
# Reuse CUDA server compiled from published IR, with only protocol-token limit raised.
# Defaults for warmups, run count, convergence and internal timeout are untouched.
# JSON is an additional output file; external timeout bounds cluster usage.
/usr/bin/time -v timeout --kill-after=10s 540s futhark bench --skip-compilation --json=results/bench.json yolov4.fut > results/bench.log 2>&1
date -u > results/FINISHED
