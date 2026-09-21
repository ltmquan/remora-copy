from pathlib import Path
import json,statistics
import numpy as np
p=Path('results'); summary={'versions':{},'checks':[]}
for version in ['old','new']:
 rounds=[]
 for n in (1,2,3):
  lines=[s.split() for s in (p/f'{version}-timing-{n}.txt').read_text().splitlines()]
  result={'round':n}
  for phase in ['resident','transfer']:
   samples=[float(x[2]) for x in lines if len(x)==3 and x[0]==phase+'_run_seconds']
   warm=[x for x in lines if len(x)==3 and x[0]==phase+'_warmup_seconds']
   assert len(samples)==30 and len(warm)==5
   result[phase+'_ms']=statistics.median(samples)*1000
  rounds.append(result)
 summary['versions'][version]={'rounds':rounds,**{phase+'_ms':statistics.median(r[phase+'_ms'] for r in rounds) for phase in ['resident','transfer']}}
for n in (1,2,3):
 a=np.fromfile(p/f'old-ramp-{n}.bin',dtype='<f4');b=np.fromfile(p/f'new-ramp-{n}.bin',dtype='<f4')
 assert a.size==b.size==22743*85 and np.isfinite(a).all() and np.isfinite(b).all()
 err=np.abs(a.astype(float)-b.astype(float));bad=err>1e-5+1e-4*np.abs(a.astype(float))
 summary['checks'].append({'round':n,'bitwise_equal':(p/f'old-ramp-{n}.bin').read_bytes()==(p/f'new-ramp-{n}.bin').read_bytes(),'failed_values':int(bad.sum()),'max_absolute_error':float(err.max())})
summary['new_vs_old_percent_change']={phase:100*(summary['versions']['new'][phase+'_ms']/summary['versions']['old'][phase+'_ms']-1) for phase in ['resident','transfer']}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
