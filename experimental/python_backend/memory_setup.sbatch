#!/bin/bash
#SBATCH --job-name=remora-memory-build
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=memory-build-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
for variant in fresh_futhark modified_futhark; do
 venv/bin/python export_futhark_inputs.py --out "$variant"
 gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$CUDA_ROOT/include" -I"$variant" "$variant/harness.c" "$variant/model.c" -L"$CUDA_ROOT/lib64" -L"$CUDA_ROOT/lib64/stubs" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "$variant/benchmark"
 sha256sum "$variant/model.c" "$variant/benchmark" > "$variant/binary-sha256.txt"
done
