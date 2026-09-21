#!/bin/bash
#SBATCH --job-name=darknet-draper-image
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:04:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-copy/experimental/darknet-draper-input-20260916
module load python/3.13.5 cuda/12.8.0
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
base=/scratch/luu.qu/remora-python-20260911
darknet=/scratch/luu.qu/remora-copy/experimental/cudnn-ab-20260912/off
published=/scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916/upstream
mkdir -p results
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > results/gpu.txt
"$base/venv/bin/python" - "$published/yolov4_input.bin" <<'PY'
from pathlib import Path
import struct,sys,hashlib
with open(sys.argv[1],'rb') as f:
 header=f.read(31);assert header[:7]==b'b\x02\x03 f32'
 assert struct.unpack('<QQQ',header[7:])==(3,608,608)
 image=f.read(3*608*608*4);assert len(image)==3*608*608*4
Path('draper-image.bin').write_bytes(image)
Path('results/image-sha256.txt').write_text(hashlib.sha256(image).hexdigest()+'\n')
PY
cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
g++ -O3 -std=c++17 -DGPU -I"$darknet/include" -I"$darknet/src" -I"$cuda_root/include" benchmark.cpp -L"$darknet" -Wl,-rpath,"$darknet" -ldarknet -L"$cuda_root/lib64" -lcudart -o benchmark
sha256sum benchmark.cpp "$darknet/libdarknet.so" "$base/yolov4.weights" > results/build-sha256.txt
for input in ramp draper; do
 args=();if [ "$input" = draper ];then args=(draper-image.bin);fi
 timeout --kill-after=5s 60s ./benchmark "$base/yolov4.cfg" "results/$input.bin" "$base/yolov4.weights" "${args[@]}" > "results/$input.log" 2>"results/$input.stderr"
done
"$base/venv/bin/python" analyze.py "$published/yolov4_expected.bin"
date -u > results/FINISHED
