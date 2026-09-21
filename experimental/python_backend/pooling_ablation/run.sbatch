#!/bin/bash
#SBATCH --job-name=yolo-pooling-ablation
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/pooling-ablation-20260915
module load python/3.13.5 cuda/12.8.0
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
base=/scratch/luu.qu/remora-python-20260911
teammate=/scratch/luu.qu/remora-copy/experimental/teammate-futhark-20260915
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
for variant in separate-fold separate-zero cascade-fold cascade-zero; do
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" -I"$variant" harness.c "$variant/model.c" -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "$variant/benchmark"
 echo "$variant"
 timeout --kill-after=10s 90s "$variant/benchmark" "results/$variant.bin" > "results/$variant.txt" 2>&1
done
for variant in previous teammate; do
 source="$base/modified_futhark"
 if [[ "$variant" == teammate ]]; then source="$teammate"; fi
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" -I"$source" profile.c "$source/model.c" -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "profile-$variant"
 echo "Profile $variant"
 timeout --kill-after=10s 240s "./profile-$variant" "$base/fresh_futhark/trained-inputs.bin" "results/profile-$variant.bin" > "results/profile-$variant.txt" 2>&1
done
date -u > results/FINISHED
