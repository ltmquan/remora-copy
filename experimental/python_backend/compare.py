"""Compare every dense value; never convert a numerical failure into a speedup claim."""
import argparse
import json
from pathlib import Path
import numpy as np

def compare(a,b,atol=1e-5,rtol=1e-4):
    if a.shape!=b.shape: raise ValueError(f'Shape mismatch {a.shape}, {b.shape}')
    finite=np.isfinite(a)&np.isfinite(b)
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
    ok=finite & (diff<=atol+rtol*np.abs(b))
    return {'elements':a.size,'nonfinite':int((~finite).sum()),'failed':int((~ok).sum()),
            'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
            'atol':atol,'rtol':rtol,'passed':bool(ok.all())}

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    report={}
    for fixture in ['synthetic','trained']:
        ref=root/f'darknet-{fixture}.bin'
        if not ref.exists(): continue
        b=np.fromfile(ref,dtype='<f4')
        for path in sorted(root.glob(f'*-{fixture}.bin')):
            if path==ref: continue
            result=np.fromfile(path,dtype='<f4')
            report[path.stem+' vs darknet']=compare(result,b)
        for backend,mode in [('torch','eager'),('torch','compile'),('tinygrad','jit')]:
            left=root/f'{backend}-remora-{mode}-{fixture}.bin';right=root/f'{backend}-direct-{mode}-{fixture}.bin'
            if left.exists() and right.exists():
                report[f'{backend}-{mode}-{fixture} remora vs direct']=compare(np.fromfile(left,dtype='<f4'),np.fromfile(right,dtype='<f4'))
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items(): print(k, v)

if __name__=='__main__': main()
