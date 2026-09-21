"""Compare latency only when correctness gates pass."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
print('| Size | Split | Latency ms | Speedup vs compiled 1 GPU | Speedup vs per-layer 1 GPU |')
print('|---|---|---:|---:|---:|')
for size in (608,1216):
 paths={k:a.directory/f'{size}-{k}.json' for k in ('compiled','one','equal','unequal')}
 ds={k:json.loads(f.read_text()) for k,f in paths.items() if f.exists()}
 for k,d in ds.items():
  if not d['passed']:
   print(f'| {size} | {k} | correctness failed | — | — |');continue
  speed=lambda base:f'{ds[base]["median_ms"]/d["median_ms"]:.3f}×' if base in ds and ds[base]['passed'] else '—'
  print(f'| {size} | {k} | {d["median_ms"]:.3f} | {speed("compiled")} | {speed("one")} |')
