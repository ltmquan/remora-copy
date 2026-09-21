"""Summarize cuDNN-off memory without reusing instrumented timings."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
p = root/'memory-off'
ref = np.fromfile(p/'darknet.bin', dtype='<f4').astype(np.float64)
assert ref.size == 1933155 and np.isfinite(ref).all()
summary = {}
for name in ('darknet','torch-remora','torch-direct'):
    d = json.loads((p/f'{name}.json').read_text())
    assert d['exit_code']==0
    samples = [s for s in d['samples'] if s['phase']!='validation']
    assert len(samples)>=2
    if name != 'darknet':
        model = name.split('-')[1]
        bench = json.loads((p/f'off-{model}.json').read_text())
        assert bench['cudnn_enabled'] is False and bench['changed_input_validation']=='passed'
        a = np.fromfile(p/f'off-{model}.bin',dtype='<f4').astype(np.float64)
        assert a.shape==ref.shape
        delta = np.abs(a-ref)
        assert np.isfinite(a).all() and np.all(delta<=1e-5+1e-4*np.abs(ref))
    summary[name] = {'gpu_process_peak_MiB':max(s['gpu_mib'] for s in samples),
        'sample_count':len(samples), 'scope':'setup/warmup/inference; extra correctness phase excluded',
        'cudnn_enabled':False, 'sampling_requested_ms':d['sampling_requested_ms']}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
