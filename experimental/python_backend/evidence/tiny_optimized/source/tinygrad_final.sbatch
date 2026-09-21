#!/bin/bash
#SBATCH --job-name=remora-tiny-optimized
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=tiny-optimized-%j.log
set -euo pipefail
cd /scratch/luu.qu/remora-python-20260911
module load python/3.13.5 cuda/12.8.0
export OMP_NUM_THREADS=4 DEV=CUDA ALLOW_TF32=0 BEAM=0 PARALLEL=4 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$PWD/cache"
mkdir -p tiny_optimized/memory
# Select on numerical eligibility first, then latency. Preserve the selection evidence.
venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('tiny_investigation');validation=json.loads((root/'validation.json').read_text());eligible=[]
for name in ['baseline','fold','materialize','im2col','winograd']:
 p=root/f'{name}-trained.json'
 if p.exists() and validation.get(f'{name}-trained vs darknet',{}).get('passed'):
  v=json.loads(p.read_text());eligible.append((v['median_seconds'],name))
if not eligible:raise RuntimeError('No numerically valid completed tinygrad candidate')
chosen=min(eligible)[1]
Path('tiny_optimized/selection.json').write_text(json.dumps({'selected':chosen,'eligible':eligible},indent=2)+'\n')
Path('tiny_optimized/selected.txt').write_text(chosen+'\n')
PY
variant=$(cat tiny_optimized/selected.txt)
hostname > tiny_optimized/node.txt
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > tiny_optimized/gpu.txt
./darknet_benchmark yolov4.cfg tiny_optimized/darknet-trained.bin yolov4.weights > tiny_optimized/darknet-trained.txt 2> tiny_optimized/darknet-trained.stderr
for model in remora direct; do
 for choice in baseline "$variant"; do
  TINYGRAD_VARIANT="$choice" timeout 90s venv/bin/python investigate_tinygrad.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "tiny_optimized/$choice-$model-trained" > "tiny_optimized/$choice-$model.log" 2>&1
 done
 TINYGRAD_VARIANT="$variant" timeout 90s venv/bin/python measure_memory.py --output "tiny_optimized/memory/$variant-$model" -- venv/bin/python investigate_tinygrad.py --backend tinygrad --model "$model" --mode jit --weights yolov4.weights --output "tiny_optimized/memory/$variant-$model-trained"
done
venv/bin/python -m unittest test_tinygrad_optimized -v > tiny_optimized/operator-tests.log 2>&1
venv/bin/python compare.py tiny_optimized
# Independently verify Remora and direct optimized outputs match.
venv/bin/python - <<'PY'
import json,numpy as np
from pathlib import Path
from compare import compare
r=Path('tiny_optimized');name=(r/'selected.txt').read_text().strip()
v=compare(np.fromfile(r/f'{name}-remora-trained.bin',dtype='<f4'),np.fromfile(r/f'{name}-direct-trained.bin',dtype='<f4'))
(r/'paired-validation.json').write_text(json.dumps(v,indent=2)+'\n')
assert v['passed'],v
PY
