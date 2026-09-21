#!/usr/bin/env bash
# Linux build; DARKNET_DIR must point to the pinned, GPU=1 CUDNN=0 build.
set -euo pipefail
cd -- "$(dirname -- "$0")"
cuda_root="${CUDA_ROOT:-$(dirname "$(dirname "$(command -v nvcc)")")}"
if [[ "${SKIP_FUTHARK:-0}" != 1 ]]; then
for source in artifacts/*/{original,modified}/futhark/model.c; do
  [[ -f "$source" ]] || continue
  gcc -shared -fPIC -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I"$cuda_root/include" \
    "$source" -L"$cuda_root/lib64" -L"$cuda_root/lib64/stubs" \
    -Wl,-rpath,"$cuda_root/lib64" -lcuda -lcudart -lnvrtc -lm -lpthread -o "${source%.c}.so"
done
fi
if [[ -n "${DARKNET_DIR:-}" ]]; then
  mkdir -p artifacts/native
  g++ -shared -fPIC -O2 -DGPU -I"$DARKNET_DIR/include" -I"$cuda_root/include" \
    darknet_adapter.cpp -L"$DARKNET_DIR" -Wl,-rpath,"$DARKNET_DIR" -ldarknet \
    -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart -o artifacts/native/darknet_adapter.so
fi
