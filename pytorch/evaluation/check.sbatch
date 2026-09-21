#!/bin/bash
#SBATCH --job-name=remora-general-check
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=gpu-%j.log
set -euo pipefail
module load python/3.13.5 cuda/12.3.0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XDG_CACHE_HOME="$PWD/cache"
/scratch/luu.qu/remora-python-20260911/venv/bin/python evaluation/run.py --prepared --device cuda --out evaluation/evidence
