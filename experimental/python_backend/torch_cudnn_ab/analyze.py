"""Compare matched PyTorch cuDNN configurations, excluding setup."""
import json
from pathlib import Path
import statistics
import numpy as np

root = Path(__file__).resolve().parent
p = root/'results'
ref = np.fromfile(p/'on-remora-1.bin',dtype='<f4').astype(np.float64)
assert ref.size == 1933155 and np.isfinite(ref).all()
summary = {}
for model in ('remora','direct'):
    for state in ('on','off'):
        key = f'{state}-{model}'
        rounds = []
        for n in (1,2,3):
            d = json.loads((p/f'{key}-{n}.json').read_text())
            assert d['cudnn_enabled'] == (state=='on')
            assert d['runs']==30 and d['warmup']==5
            assert d['changed_input_validation']=='passed'
            a = np.fromfile(p/f'{key}-{n}.bin',dtype='<f4').astype(np.float64)
            assert a.shape == ref.shape
            delta = np.abs(a-ref)
            bad = int(np.count_nonzero(~np.isfinite(a)|(delta>1e-5+1e-4*np.abs(ref))))
            rounds.append({'round':n,'resident_ms':d['median_seconds']*1000,
                'transfer_ms':d['end_to_end_median_seconds']*1000,
                'failed_values':bad,'max_absolute_error':float(delta.max())})
        passed = all(r['failed_values']==0 for r in rounds)
        summary[key] = {'passed':passed,'rounds':rounds,
            'resident_ms':statistics.median(r['resident_ms'] for r in rounds) if passed else None,
            'transfer_ms':statistics.median(r['transfer_ms'] for r in rounds) if passed else None}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert all(d['passed'] for d in summary.values()),'Validation failed; do not accept timings'
lines=['# PyTorch cuDNN on/off check','',
    'One H200 allocation, 608×608 YOLOv4, trained weights, identical FP32/TF32-off policy. '
    'Both paths use compiled execution with batch one. The explicit cuDNN flag is recorded '
    'and checked after execution; on/off compile caches are separate. Three rounds alternate '
    'execution order, with five warmups and thirty timed calls per mode in each process. '
    'Compilation and warmup are excluded. Aggregates are medians of the three process medians.', '',
    '| Model | cuDNN | Resident ms | Transfer-inclusive ms | Resident round medians | Transfer round medians |',
    '|---|---|---:|---:|---|---|']
for key,d in summary.items():
    state,model=key.split('-')
    lines.append(f'| {model} | {state} | {d["resident_ms"]:.3f} | {d["transfer_ms"]:.3f} | '+
        ', '.join(f'{r["resident_ms"]:.3f}' for r in d['rounds'])+' | '+
        ', '.join(f'{r["transfer_ms"]:.3f}' for r in d['rounds'])+' |')
lines += ['', 'All twelve outputs passed comparison over 1,933,155 values against cuDNN-on '
    'Remora/round1 at atol=1e-5, rtol=1e-4. Each process also checked a changed input '
    'against eager execution. This is fixture validation, not a general equivalence proof.', '',
    'The separate Darknet on/off study was a different allocation; comparisons between '
    'the studies are contextual, not a matched four-way benchmark.', '']
(root/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
