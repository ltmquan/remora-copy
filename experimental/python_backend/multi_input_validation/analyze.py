from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parent
cases=json.loads((root/'inputs.json').read_text())['images']
variants=['fresh_futhark','modified_futhark','torch-remora','torch-direct','baseline-remora','baseline-direct','im2col-remora','im2col-direct']
results=[]
for case in cases:
 name=case['name'];ref=np.fromfile(root/'results'/f'darknet-{name}.bin',dtype='<f4').astype(np.float64)
 assert ref.size==1933155 and np.isfinite(ref).all()
 for variant in variants:
  a=np.fromfile(root/'results'/f'{variant}-{name}.bin',dtype='<f4').astype(np.float64)
  assert a.shape==ref.shape
  error=np.abs(a-ref);bad=~np.isfinite(a)|(error>1e-5+1e-4*np.abs(ref))
  results.append({'input':name,'pipeline':variant,'passed':not bool(bad.any()),'failed_values':int(bad.sum()),'max_absolute_error':float(error.max())})
summary={'atol':1e-5,'rtol':1e-4,'images':6,'pipelines_including_darknet':9,'comparisons':results,'passed':all(r['passed'] for r in results)}
(root/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
# Keep failure details in summary before returning a failed exit status.
if not summary['passed']:raise SystemExit(1)
