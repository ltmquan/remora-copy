#!/usr/bin/env bash
# Run on rental after installing git, make, g++, and the CUDA development toolkit.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p artifacts/native
darknet_dir="$PWD/artifacts/native/darknet"
revision=58bb1326f8aefe19805b33901101697de5f09336
archive="$PWD/artifacts/native/darknet-source.tar.gz"
if [[ ! -d "$darknet_dir" ]]; then
  printf '%s  %s\n' "2f4ec20cc43ab042982fa65ed15aa45498e9bcbed0a4d56eff14c9d09a02bce4" "$archive" | sha256sum -c -
  mkdir -p "$darknet_dir"
  tar -xzf "$archive" -C "$darknet_dir"
  printf '%s\n' "$revision" > "$darknet_dir/.remora-source-revision"
fi
[[ "$(cat "$darknet_dir/.remora-source-revision")" == "$revision" ]] || { echo 'Unexpected Darknet source'; exit 1; }
# Detect architecture instead of assuming H100; record the actual build target.
python_bin="${PYTHON_BIN:-$PWD/.venv/bin/python}"
arch_flags=$("$python_bin" - <<'ARCHPY'
import torch
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable; inspect preflight before building')
capabilities=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
print(' '.join(f'-gencode arch=compute_{major}{minor},code=sm_{major}{minor}' for major,minor in capabilities))
ARCHPY
)
# Preserve the earlier cuDNN-disabled native comparison configuration.
make -C "$darknet_dir" -j"${BUILD_JOBS:-4}" CC=gcc CPP=g++ NVCC="nvcc -ccbin g++" GPU=1 CUDNN=0 CUDNN_HALF=0 OPENCV=0 AVX=0 OPENMP=0 LIBSO=1 \
  ARCH="$arch_flags"
printf '%s\n' "$revision GPU=1 CUDNN=0 CUDNN_HALF=0 OPENCV=0 AVX=0 OPENMP=0 LIBSO=1 $arch_flags" > artifacts/native/build-config.txt
DARKNET_DIR="$darknet_dir" SKIP_FUTHARK=1 bash build_native.sh
