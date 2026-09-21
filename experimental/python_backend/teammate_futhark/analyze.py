from pathlib import Path
import ast,json,statistics
import numpy as np
root=Path(__file__).resolve().parent;p=root/'results'
prior=root.parent/'multi_input_validation';refdir=prior/'results'
# Reuse the exact CPU NMS implementation without running its CLI body.
tree=ast.parse((prior/'compare_nms.py').read_text());ns={'np':np}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'compare_nms.py','exec'),ns)
rounds=[]
for n in (1,2,3):
 lines=[s.split() for s in (p/f'timing-{n}.txt').read_text().splitlines()];r={'round':n}
 for phase in ['resident','transfer']:
  samples=[float(x[2]) for x in lines if len(x)==3 and x[0]==phase+'_run_seconds']
  warm=[x for x in lines if len(x)==3 and x[0]==phase+'_warmup_seconds']
  assert len(samples)==30 and len(warm)==5
  r[phase+'_ms']=statistics.median(samples)*1000
 rounds.append(r)
checks=[]
for name in ['ramp-1','ramp-2','ramp-3','black','white','noise','gradient','checkerboard','evalbox']:
 refpath=root.parent/'torch_cudnn_ab/memory-off/darknet.bin' if name.startswith('ramp') else refdir/f'darknet-{name}.bin'
 a=np.fromfile(p/f'{name}.bin',dtype='<f4').reshape(22743,85);ref=np.fromfile(refpath,dtype='<f4').reshape(22743,85)
 assert np.isfinite(a).all() and np.isfinite(ref).all()
 delta=np.abs(a.astype(float)-ref.astype(float));bad=delta>1e-5+1e-4*np.abs(ref.astype(float))
 aa=ns['nms'](a);rr=ns['nms'](ref);ad={(d['candidate'],d['class_id']):d for d in aa};rd={(d['candidate'],d['class_id']):d for d in rr};common=ad.keys()&rd.keys()
 checks.append({'input':name,'passed_dense':not bool(bad.any()),'failed_values':int(bad.sum()),'max_absolute_error':float(delta.max()),'detections':aa,'darknet_detections':rr,'same_nms_candidates_classes':ad.keys()==rd.keys(),'max_matched_confidence_error':max((abs(ad[k]['confidence']-rd[k]['confidence']) for k in common),default=0),'max_matched_xywh_error_pixels':max((float(np.max(np.abs(np.array(ad[k]['xywh'])-rd[k]['xywh']))*608) for k in common),default=0)})
m=json.loads((p/'memory.json').read_text());assert m['exit_code']==0
samples=[s for s in m['samples'] if s['phase']!='validation'];assert len(samples)>1
summary={'rounds':rounds,'resident_ms':statistics.median(r['resident_ms'] for r in rounds),'transfer_ms':statistics.median(r['transfer_ms'] for r in rounds),'gpu_footprint_MiB':max(s['gpu_mib'] for s in samples),'checks':checks,'atol':1e-5,'rtol':1e-4,'nms_confidence':.25,'nms_iou':.45}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
