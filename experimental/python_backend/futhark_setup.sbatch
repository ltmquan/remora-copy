#!/bin/bash
#SBATCH --job-name=remora-futhark-build
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=futhark-build-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
venv/bin/python export_futhark_inputs.py
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$CUDA_ROOT/include" -Ifresh_futhark fresh_futhark/harness.c fresh_futhark/model.c -L"$CUDA_ROOT/lib64" -L"$CUDA_ROOT/lib64/stubs" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o fresh_futhark/benchmark
sha256sum fresh_futhark/model.c fresh_futhark/benchmark > fresh_futhark/binary-sha256.txt
