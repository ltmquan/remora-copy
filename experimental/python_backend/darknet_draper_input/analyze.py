from pathlib import Path
import json,statistics,sys,struct
import numpy as np
p=Path('results');r={}
for name in ['ramp','draper']:
 lines=[line.split() for line in (p/f'{name}.log').read_text().splitlines()]
 total=[float(x[2])*1000 for x in lines if len(x)==3 and x[0]=='run_seconds']
 predict=[float(x[2])*1000 for x in lines if len(x)==3 and x[0]=='predict_seconds' and int(x[1])>=0]
 assert len(total)==len(predict)==3
 r[name]={'total_samples_ms':total,'total_mean_ms':statistics.mean(total),'predict_samples_ms':predict,'predict_mean_ms':statistics.mean(predict)}
a=np.fromfile(p/'draper.bin',dtype='<f4')
with open(sys.argv[1],'rb') as f:
 header=f.read(23);assert header[:7]==b'b\x02\x02 f32';assert struct.unpack('<QQ',header[7:])==(22743,85)
 b=np.frombuffer(f.read(),dtype='<f4')
assert a.size==b.size==22743*85 and np.isfinite(a).all() and np.isfinite(b).all()
e=np.abs(a.astype(float)-b.astype(float));bad=e>1e-5+1e-4*np.abs(b.astype(float))
r['published_expected_comparison']={'failed_values':int(bad.sum()),'max_absolute_error':float(e.max()),'reference':'published Futhark demo expected output, not an independent new Darknet run'}
(p/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
