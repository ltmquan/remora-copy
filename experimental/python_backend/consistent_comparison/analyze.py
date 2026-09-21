"""Validate and aggregate the six newly standardized pipelines."""
from pathlib import Path
import json, statistics
import numpy as np
root=Path(__file__).resolve().parent
p=root/'results'
ref=np.fromfile(root.parent/'torch_cudnn_ab/memory-off/darknet.bin',dtype='<f4').astype(np.float64)
assert ref.size==1933155 and np.isfinite(ref).all()
summary={}
for key in ('fresh_futhark','modified_futhark','baseline-remora','baseline-direct','im2col-remora','im2col-direct'):
    rounds=[]
    for n in (1,2,3):
        prefix=p/f'{key}-{n}'
        a=np.fromfile(prefix.with_suffix('.bin'),dtype='<f4').astype(np.float64)
        assert a.shape==ref.shape
        error=np.abs(a-ref)
        assert np.isfinite(a).all() and np.all(error<=1e-5+1e-4*np.abs(ref)), (key,n)
        if 'futhark' in key:
            lines=[line.split() for line in prefix.with_suffix('.txt').read_text().splitlines()]
            medians={}
            for phase in ('resident','transfer'):
                samples=[float(v[2]) for v in lines if len(v)==3 and v[0]==phase+'_run_seconds']
                warmups=[v for v in lines if len(v)==3 and v[0]==phase+'_warmup_seconds']
                assert len(samples)==30 and len(warmups)==5
                medians[phase]=statistics.median(samples)*1000
            r,t=medians['resident'],medians['transfer']
        else:
            d=json.loads(prefix.with_suffix('.json').read_text())
            assert d['runs']==30 and d['warmup']==5
            r,t=d['median_seconds']*1000,d['end_to_end_median_seconds']*1000
        rounds.append({'round':n,'resident_ms':r,'transfer_ms':t,'max_absolute_error':float(error.max())})
    summary[key]={'rounds':rounds,'resident_ms':statistics.median(x['resident_ms'] for x in rounds),'transfer_ms':statistics.median(x['transfer_ms'] for x in rounds),'passed':True}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
