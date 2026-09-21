"""Recheck saved interpreter fixtures on a device; no frontend required."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--device', default='cuda')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.directory / 'results.json').read_text())
if not manifest.get('passed'):
    raise SystemExit('CPU correctness suite must pass before device validation')
torch.set_num_threads(2)
torch.set_grad_enabled(False)
if a.device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
report = {'device': a.device, 'torch': torch.__version__, 'source_sha256': manifest['source_sha256'],
          'gpu': torch.cuda.get_device_name() if a.device == 'cuda' else None,
          'comparisons': [], 'passed': False}
for case in manifest['cases']:
    for lowering in case['lowerings']:
        path = a.directory / (case['name'] + '_' + lowering + '.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['lowerings'][lowering]['generated_sha256']:
            raise ValueError(f'Generated module hash differs: {path}')
        spec = importlib.util.spec_from_file_location('generated', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for fixture in case['fixtures']:
            prefix = case['name'] + '_' + str(fixture['seed'])
            result = {'case': case['name'], 'lowering': lowering, 'selected': module.LOWERING, 'seed': fixture['seed']}
            try:
                with np.load(a.directory / (prefix + '_inputs.npz')) as data:
                    inputs = [torch.from_numpy(data[k]).to(a.device) for k in data.files]
                expected = np.load(a.directory / (prefix + '_reference.npy'))
                actual = module.ENTRIES['main'](*inputs).detach().cpu().numpy()
                if actual.shape != expected.shape:
                    raise ValueError(f'shape {actual.shape} != {expected.shape}')
                diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                limit = manifest['atol'] + manifest['rtol'] * np.abs(expected.astype(np.float64))
                bad = (diff > limit) | ~np.isfinite(actual) | ~np.isfinite(expected)
                result.update(passed=not bool(bad.any()), mismatches=int(bad.sum()),
                              max_abs=float(diff.max(initial=0)), max_error_over_tolerance=float((diff / limit).max(initial=0)))
            except Exception as exc:
                result.update(passed=False, error=str(exc))
            report['comparisons'].append(result)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(result), flush=True)
report['passed'] = bool(report['comparisons']) and all(r['passed'] for r in report['comparisons'])
a.output.write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(0 if report['passed'] else 1)
