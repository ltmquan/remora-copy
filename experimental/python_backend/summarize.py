"""Render a compact, auditable timing table from retained JSON/text records."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('evidence');a=p.parse_args();root=Path(a.evidence)
print('| Run set / implementation | GPU-resident median ms | Transfer-inclusive median ms | First run + warm-ups s |')
print('|---|---:|---:|---:|')
for group in ['memory_matrix','tiny_investigation','tiny_optimized','fresh','results','tuning','graphs']:
    d=root/group
    if not d.exists(): continue
    for f in sorted(d.glob('*trained.json')):
        v=json.loads(f.read_text())
        if 'median_seconds' not in v: continue
        print(f'| {group} / {f.stem} | {1000*v["median_seconds"]:.3f} | {1000*v["end_to_end_median_seconds"]:.3f} | {v["first_eager_seconds"]+sum(v["warmup_seconds"]):.3f} |')
    for name in ['futhark-original','fresh_futhark','modified_futhark']:
        f=d/f'{name}-trained.txt'
        if f.exists():
            lines=f.read_text().splitlines()
            resident=[float(s.split()[-1]) for s in lines if s.startswith('resident_run_seconds')]
            transfer=[float(s.split()[-1]) for s in lines if s.startswith('transfer_run_seconds')]
            warm=[float(s.split()[-1]) for s in lines if s.startswith('resident_warmup_seconds')]
            print(f'| {group} / {name} | {1000*statistics.median(resident):.3f} | {1000*statistics.median(transfer):.3f} | {sum(warm):.3f} |')
    f=d/'darknet-trained.txt'
    if f.exists():
        lines=f.read_text().splitlines()
        times=[float(s.split()[-1]) for s in lines if s.startswith('run_seconds')]
        pred=[float(s.split()[-1]) for s in lines if s.startswith('predict_seconds') and s.split()[1]!='-1']
        print(f'| {group} / Darknet native + dense decode | — | {1000*statistics.median(times):.3f} | — |')
        print(f'<!-- Darknet predict API (includes transfers): {1000*statistics.median(pred):.3f} ms -->')
print('\nValidation:')
for f in root.glob('*/validation.json'):
    v=json.loads(f.read_text());failed=[k for k,r in v.items() if not r['passed']]
    print(f'{f.parent.name}: {len(v)} comparisons, {len(failed)} failed; maximum absolute difference {max(r["max_abs"] for r in v.values()):.9g}')
    for k in failed: print(k,v[k])

print('\nGPU process memory (sampled MiB, separate instrumented runs):')
for f in sorted(root.glob('*/memory/*.json')):
    v=json.loads(f.read_text())
    if 'gpu_process_peak_mib' in v:print(f.parent.parent.name,f.stem,v.get('gpu_setup_inference_peak_mib',max(s['gpu_mib'] for s in v['samples'] if s['phase']!='validation')),v['phase_gpu_peak_mib'])
