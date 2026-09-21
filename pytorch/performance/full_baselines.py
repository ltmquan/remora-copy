"""Rerun existing full-network baselines; validate outputs before reporting timings."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np

assets=Path('/scratch/luu.qu/remora-python-20260911')
out=Path.cwd()
build=json.loads((assets/'modified_futhark/build.json').read_text())
assert build['source_sha256']=='334346d83578f49a52507fab11d0cfed0da269b10f20ab012ade4f6ca98a6e89'
with (out/'full-direct.log').open('w') as log:
 subprocess.run([sys.executable,str(assets/'benchmark.py'),'--backend','torch','--model','direct','--mode','eager','--device','cuda','--weights',str(assets/'yolov4.weights'),'--runs','10','--warmup','3','--output',str(out/'full-direct.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
with (out/'full-futhark.log').open('w') as log:
 subprocess.run([str(out/'full-futhark-benchmark'),str(assets/'modified_futhark/trained-inputs.bin'),str(out/'full-futhark.bin')],stdout=log,stderr=subprocess.STDOUT,check=True)
x=np.fromfile(out/'full-futhark.bin',dtype='<f4');y=np.fromfile(out/'full-direct.bin',dtype='<f4')
assert x.shape==y.shape==(22743*85,)
diff=np.abs(x.astype(np.float64)-y.astype(np.float64));limit=1e-5+1e-4*np.abs(y.astype(np.float64))
assert np.isfinite(x).all() and np.isfinite(y).all() and (diff<=limit).all(),f'max difference {diff.max()}'
f_samples=[float(line.split()[2])*1000 for line in (out/'full-futhark.log').read_text().splitlines() if line.startswith('resident_run_seconds ')]
assert len(f_samples)==10
pt=json.loads((out/'full-direct.json').read_text())
report={'passed':True,'source_sha256':build['source_sha256'],'shape':[22743,85],'scalars':len(x),'max_abs':float(diff.max()),'max_error_over_tolerance':float((diff/limit).max()),'direct_pytorch_eager_median_ms':pt['median_seconds']*1000,'remora_futhark_median_ms':statistics.median(f_samples),'futhark_samples_ms':f_samples,'direct_samples_ms':[v*1000 for v in pt['run_seconds']],'futhark_binary_sha256':hashlib.sha256((out/'full-futhark-benchmark').read_bytes()).hexdigest(),'futhark_build':build,'gpu':pt['gpu'],'new_compiler_full_network':'Not timed: prior general-backend attempt did not complete within its test limit. See small workload timings.'}
(out/'full-baselines.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
