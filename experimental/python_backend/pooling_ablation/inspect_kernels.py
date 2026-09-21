from pathlib import Path
import re,ast,json,collections
base=Path(__file__).resolve().parent.parent;out=base/'pooling_ablation/results'
for v,folder in [('previous','modified_futhark'),('teammate','teammate_futhark')]:
 s=(base/folder/'model.c').read_text();s=s.split('static const char *gpu_program[] = {',1)[1].split('NULL};',1)[0];strings=re.findall(r'"(?:[^"\\]|\\.)*"',s)
 program=''.join(ast.literal_eval(x) for x in strings)
 sections=re.split(r'FUTHARK_KERNEL_SIZED\(',program)[1:];kernels={}
 for sec in sections:
  name=sec.split('_dim',1)[0];kernels[name.replace('zi','.')]=sec
 d=json.loads((out/f'profile-{v}.json').read_text());events=collections.defaultdict(list)
 for e in d['events']:events[e['name']].append(e['duration'])
 print(v)
 for name,body in kernels.items():
  if 'fmax32(' in body and name in events:
   print(name, 'max',body.count('fmax32('),'mul',body.count('fmul32('),'exp',body.count('exp32('),'events',len(events[name]),'us',round(sum(events[name]),3))
 (out/f'{v}-kernels.json').write_text(json.dumps(kernels))
