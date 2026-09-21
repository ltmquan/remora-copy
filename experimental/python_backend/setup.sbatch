#!/bin/bash
#SBATCH --job-name=remora-python-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=setup-%j.log
set -euo pipefail
module load python/3.13.5
python3 -m venv venv
venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
venv/bin/pip install tinygrad==0.14.0 numpy==2.5.3
venv/bin/pip freeze > requirements.lock
curl -fL --max-time 240 https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights -o yolov4.weights
printf '%s\n' 'e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562  yolov4.weights' | sha256sum -c -
