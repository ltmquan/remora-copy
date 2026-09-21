#!/bin/bash
#SBATCH --job-name=remora-correctness
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=device-%j.log
set -euo pipefail
module load python/3.13.5 cuda/12.8.0
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/cache"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
/scratch/luu.qu/remora-python-20260911/venv/bin/python check_device.py fixtures --device cuda --output device-results.json
if test -f yolov4-current.py; then
  set +e
  timeout --signal=TERM --kill-after=10s 120s /scratch/luu.qu/remora-python-20260911/venv/bin/python check_full_network.py \
    --assets /scratch/luu.qu/remora-python-20260911 \
    --module yolov4-current.py --names yolov4-entry-names.json \
    --weights /scratch/luu.qu/remora-python-20260911/yolov4.weights --output full-network-results.json
  probe_status=$?
  printf '%s\n' "$probe_status" > full-network-exit-status.txt
  exit "$probe_status"
fi
