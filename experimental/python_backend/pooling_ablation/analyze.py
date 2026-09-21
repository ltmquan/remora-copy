from pathlib import Path
import json,statistics
import numpy as np
p=Path(__file__).resolve().parent
x=(((np.arange(512*19*19,dtype=np.int64)*17)%251-125)/np.float32(125)).astype(np.float32).reshape(512,19,19)
def pool(k):
 padded=np.pad(x,((0,0),(k//2,k//2),(k//2,k//2)),constant_values=-1e30)
 return np.lib.stride_tricks.sliding_window_view(padded,(k,k),axis=(1,2)).max(axis=(-1,-2))
ref=np.concatenate([pool(13),pool(9),pool(5),x],axis=0)
summary={}
for v in ['separate-fold','separate-zero','cascade-fold','cascade-zero']:
 a=np.fromfile(p/'results'/f'{v}.bin',dtype='<f4').reshape(2048,19,19)
 assert np.array_equal(a,ref),v
 lines=[s.split() for s in (p/'results'/f'{v}.txt').read_text().splitlines()]
 rounds=[]
 for n in (1,2,3):
  times=[float(s[3])*1000 for s in lines if len(s)==4 and s[0]=='run' and int(s[1])==n]
  assert len(times)==100
  rounds.append(statistics.median(times))
 summary[v]={'round_medians_ms':rounds,'median_ms':statistics.median(rounds),'bitwise_equal_numpy_reference':True}
(p/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
# Raw Futhark profiling JSON is retained separately, never used as uninstrumented latency.
for v in ['previous','teammate']:
 s=(p/'results'/f'profile-{v}.txt').read_text()
 report=s.split('FUTHARK_MEMORY_REPORT ',1)[1]
 try:d=json.loads(report)
 except json.JSONDecodeError:d={'raw':report}
 (p/'results'/f'profile-{v}.json').write_text(json.dumps(d,indent=2)+'\n')
