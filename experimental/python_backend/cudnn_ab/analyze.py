"""Validate controlled Darknet outputs before reporting cuDNN timings."""
from pathlib import Path
import json
import re
import statistics
import numpy as np

root = Path(__file__).resolve().parent
p = root / 'results'
reference = np.fromfile(p / 'off-1.bin', dtype='<f4').astype(np.float64)
assert reference.size == 22743*85 and np.isfinite(reference).all()
summary = {'reference': 'cuDNN off, round 1', 'atol': 1e-5, 'rtol': 1e-4,
           'order': [['off','heuristic','search'],['heuristic','search','off'],['search','off','heuristic']],
           'gpu': (p/'gpu.txt').read_text().strip(), 'variants': {}}
for variant in ('off','heuristic','search'):
    rounds = []
    for n in (1,2,3):
        assert (p/f'{variant}-{n}.exit').read_text().strip() == '0'
        output = np.fromfile(p/f'{variant}-{n}.bin',dtype='<f4').astype(np.float64)
        assert output.shape == reference.shape
        delta = np.abs(output-reference)
        bad = ~np.isfinite(output) | (delta > 1e-5+1e-4*np.abs(reference))
        text = (p/f'{variant}-{n}.txt').read_text()
        series = {}
        for field in ('run_seconds','predict_seconds'):
            series[field] = {int(i):float(v)*1000 for i,v in re.findall(r'^'+field+r' (-?\d+) ([0-9.]+)',text,re.M) if int(i)>=0}
            assert len(series[field]) == 30
        rounds.append({'round':n, 'failed_values':int(bad.sum()), 'max_absolute_error':float(delta.max()),
            'total_ms':statistics.median(series['run_seconds'].values()),
            'prediction_ms':statistics.median(series['predict_seconds'].values()),
            'decode_ms':statistics.median(series['run_seconds'][i]-series['predict_seconds'][i] for i in range(30)),
            'setup_seconds':float(re.search(r'^context_seconds ([0-9.]+)',text,re.M)[1]),
            'all_total_ms':list(series['run_seconds'].values())})
    passed = all(r['failed_values']==0 for r in rounds)
    summary['variants'][variant] = {'passed':passed,'rounds':rounds,
        'median_round_total_ms':statistics.median(r['total_ms'] for r in rounds) if passed else None,
        'median_round_prediction_ms':statistics.median(r['prediction_ms'] for r in rounds) if passed else None,
        'median_round_decode_ms':statistics.median(r['decode_ms'] for r in rounds) if passed else None}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert all(v['passed'] for v in summary['variants'].values()), 'Numerical failure; no speedup claim'
lines=['# Controlled Darknet cuDNN comparison','',
       'One Slurm-allocated H200; same source, FP32 with TF32 disabled, checkpoint, 608×608 ramp image, '
       'compiler flags and harness. Each variant runs in three processes, alternating order; '
       'each process has five warmups and thirty timed single-image calls. Setup/search time is excluded.', '',
       '“Search” replaces the forward convolution algorithm heuristic with measured cuDNN algorithm search. '
       'The existing workspace and candidate filters are retained. This does not claim an exhaustive optimization of Darknet.', '',
       '| Variant | Total ms | Prediction + transfers ms | CPU decode ms | Total medians by round, ms |',
       '|---|---:|---:|---:|---|']
for name,v in summary['variants'].items():
    lines.append(f'| {name} | {v["median_round_total_ms"]:.3f} | {v["median_round_prediction_ms"]:.3f} | '
                 f'{v["median_round_decode_ms"]:.3f} | '+', '.join(f'{r["total_ms"]:.3f}' for r in v['rounds'])+' |')
lines += ['', 'Each displayed aggregate is the median of the three process medians; component medians '
          'need not sum exactly. All nine outputs passed the full 1,933,155-value comparison against off/round1 '
          'at atol=1e-5, rtol=1e-4. NMS is excluded; dense decode is included. '
          'This test does not include PyTorch and must not replace the prior matched PyTorch/Darknet allocation selectively.', '']
(root/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines))
