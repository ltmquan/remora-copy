"""Bounded by the Slurm caller: compare current full compiler to direct YOLO."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--assets', type=Path, required=True)
p.add_argument('--module', type=Path, required=True)
p.add_argument('--names', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.assets))
from benchmark import fixture
from direct import YOLOv4
from runtime import Ops

torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
report = {'passed': False, 'stage': 'setup', 'gpu': torch.cuda.get_device_name(),
          'torch': torch.__version__, 'generated_sha256': hashlib.sha256(a.module.read_bytes()).hexdigest()}
def save():
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
save()
model = YOLOv4(a.assets / 'yolov4.cfg')
arrays = fixture(model, a.weights)
names = json.loads(a.names.read_text())
if set(names) != set(arrays) or len(names) != len(arrays):
    raise ValueError('Parsed Remora parameters and independent cfg fixture differ')
spec = importlib.util.spec_from_file_location('current_yolo', a.module)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report['lowering'] = module.LOWERING
ops = Ops('torch', 'cuda', False)
inputs = {k: ops.tensor(v) for k, v in arrays.items()}
report['stage'] = 'direct_reference'; save()
expected = ops.realize(model(ops, inputs)).detach().cpu().numpy()
if not np.isfinite(expected).all():
    raise ValueError('Nonfinite direct reference')
np.save(a.output.with_suffix('.reference.npy'), expected)
report['stage'] = 'compiled_inference'; save()
start = time.monotonic()
actual = module.ENTRIES['main'](*(inputs[k] for k in names)).detach().cpu().numpy()
report['inference_seconds'] = time.monotonic() - start
np.save(a.output.with_suffix('.actual.npy'), actual)
if actual.shape != expected.shape:
    raise ValueError(f'Output shapes differ: {actual.shape} vs {expected.shape}')
diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
limit = 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))
bad = (diff > limit) | ~np.isfinite(actual)
report.update(stage='complete', passed=not bool(bad.any()), shape=list(actual.shape),
              mismatches=int(bad.sum()), max_abs=float(diff.max()),
              max_error_over_tolerance=float((diff/limit).max()))
save()
raise SystemExit(0 if report['passed'] else 1)
