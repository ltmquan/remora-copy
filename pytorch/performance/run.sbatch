#!/bin/bash
#SBATCH --job-name=remora-small-perf
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=performance-%j.log
set -euo pipefail
module load python/3.13.5 cuda/12.3.0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export XDG_CACHE_HOME="$PWD/cache"
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
for variant in conv_same spp_cascade composed_pipeline; do
 test -f "$variant/harness.c" || continue
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$CUDA_ROOT/include" -I"$variant" "$variant/harness.c" "$variant/model.c" -L"$CUDA_ROOT/lib64" -L"$CUDA_ROOT/lib64/stubs" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "$variant/benchmark"
done
/scratch/luu.qu/remora-python-20260911/venv/bin/python benchmark.py .
full_source=/scratch/luu.qu/remora-python-20260911/modified_futhark
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$CUDA_ROOT/include" -I"$full_source" "$full_source/harness.c" "$full_source/model.c" -L"$CUDA_ROOT/lib64" -L"$CUDA_ROOT/lib64/stubs" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o full-futhark-benchmark
/scratch/luu.qu/remora-python-20260911/venv/bin/python full_baselines.py
