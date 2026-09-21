from pathlib import Path
import json, statistics
p=Path('results'); lines=[s.split() for s in (p/'new-timing.txt').read_text().splitlines()]
r={'compiler':'0.27.1','processes':1,'warmups_per_mode':5,'timed_calls_per_mode':3}
for phase in ['resident','transfer']:
 samples=[float(x[2])*1000 for x in lines if len(x)==3 and x[0]==phase+'_run_seconds']
 assert len(samples)==3
 r[phase]={'samples_ms':samples,'mean_ms':statistics.mean(samples)}
(p/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
