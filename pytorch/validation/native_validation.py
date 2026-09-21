"""Regenerate auto lowering and replay saved interpreter fixtures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from remora_torch import compile_source
from yolo_differential import compare
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--remora',required=True);a=p.parse_args()
old=json.loads((a.fixtures/'results.json').read_text());assert old['passed']
a.out.mkdir(parents=True,exist_ok=True);result={k:v for k,v in old.items() if k not in ('cases','passed','failed_fixtures','failed_compilations')};result.update(cases=[],passed=False)
torch.set_num_threads(1);torch.set_grad_enabled(False)
for case in old['cases']:
 name=case['name'];source=a.fixtures/(name+'.remora')
 code=compile_source(source=source,remora=a.remora,lowering='auto',timeout=120)
 path=a.out/(name+'_auto.py');path.write_text(code)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rec={'name':name,'lowerings':{'auto':{'selected':module.LOWERING,'generated_sha256':hashlib.sha256(code.encode()).hexdigest()}},'fixtures':[]};result['cases'].append(rec)
 for fixture in case['fixtures']:
  seed=fixture['seed'];prefix=f'{name}_{seed}'
  for suffix in ['_inputs.npz','_reference.npy']:shutil.copyfile(a.fixtures/(prefix+suffix),a.out/(prefix+suffix))
  with np.load(a.out/(prefix+'_inputs.npz')) as inputs:args=[torch.from_numpy(inputs[k]) for k in inputs.files]
  expected=np.load(a.out/(prefix+'_reference.npy'));actual=module.main(*args).numpy()
  comparison=compare(actual,expected,result['atol'],result['rtol'])
  rec['fixtures'].append({'seed':seed,'comparisons':{'auto':comparison}})
  print(name,module.LOWERING,seed,comparison,flush=True)
  (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
result['passed']=all(f['comparisons']['auto']['passed'] for c in result['cases'] for f in c['fixtures'])
(a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(0 if result['passed'] else 1)
