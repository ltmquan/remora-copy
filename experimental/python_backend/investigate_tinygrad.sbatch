#!/bin/bash
#SBATCH --job-name=remora-tiny-investigate
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=tiny-investigate-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 PARALLEL=4 PYTHONUNBUFFERED=1 NO_COLOR=1
export XDG_CACHE_HOME="$PWD/cache"
mkdir -p tiny_investigation
hostname > tiny_investigation/node.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > tiny_investigation/gpu.txt
./darknet_benchmark yolov4.cfg tiny_investigation/darknet-trained.bin yolov4.weights > tiny_investigation/darknet-trained.txt 2> tiny_investigation/darknet-trained.stderr
# Per-kernel diagnostic timing is separate from the uninstrumented comparison.
DEBUG=2 BEAM=0 timeout 180s venv/bin/python benchmark.py --backend tinygrad --model remora --mode jit --runs 1 --weights yolov4.weights --output tiny_investigation/profile-trained > tiny_investigation/profile.log 2>&1 || echo "profile status $?"
for variant in baseline fold materialize im2col winograd; do
 TINYGRAD_VARIANT="$variant" BEAM=0 timeout 150s venv/bin/python investigate_tinygrad.py --backend tinygrad --model remora --mode jit --weights yolov4.weights --output "tiny_investigation/$variant-trained" > "tiny_investigation/$variant.log" 2>&1 || echo "$variant status $?"
done
# Bounded diagnostic, reusing the existing cache; log which searches finish.
TRACE_BEAM=1 DEBUG=2 BEAM=2 timeout --kill-after=10s 180s venv/bin/python investigate_tinygrad.py --backend tinygrad --model remora --mode jit --weights yolov4.weights --output tiny_investigation/beam2-trained > tiny_investigation/beam2.log 2>&1 || echo "beam2 diagnostic status $?"
venv/bin/python compare.py tiny_investigation
