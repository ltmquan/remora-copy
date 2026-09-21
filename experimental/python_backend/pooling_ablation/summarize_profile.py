"""Attribute executed SPP kernels using inspected generated CUDA names."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent/'results'
summary={}
# The inclusive ranges contain the three windows' index generation, extraction,
# and maxima. They exclude adjacent conv/activation and later concat consumers.
for v,start,end,names in [('previous','main.segmap_295187','main.segmap_299949',[295816,297057,297699,298940,299582,299949]),('teammate','main.segmap_304382','main.segmap_305030',[304454,304520,304709,304775,304964,305030])]:
 events=json.loads((p/f'profile-{v}.json').read_text())['events']
 lo=[i for i,e in enumerate(events) if e['name']==start][-1];hi=[i for i,e in enumerate(events) if e['name']==end][-1]
 section=events[lo:hi+1];reductions=[e for e in section if e['name'] in ['main.segmap_'+str(n) for n in names]]
 first=next(e['name'] for e in events if e['name'].startswith('main.'));starts=[i for i,e in enumerate(events) if e['name']==first];assert len(starts)==2
 network=[e for e in events[starts[-1]:] if e['name'].startswith('main.')]
 summary[v]={'scope':'second of two profiled invocations, transfer mode; not a warmed latency estimate','pooling_kernel_ms':sum(e['duration'] for e in section)/1000,'max_reduction_kernel_ms':sum(e['duration'] for e in reductions)/1000,'max_reduction_launches':len(reductions),'main_kernel_ms':sum(e['duration'] for e in network)/1000,'pooling_events':section}
(p/'profile-attribution.json').write_text(json.dumps(summary,indent=2)+'\n')
