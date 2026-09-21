"""FP64 diagnostic reference; never part of the FP32 timing comparisons."""
import json
from pathlib import Path
import numpy as np
import torch
from model import YOLOv4, fixture, image, Ops, ROOT

torch.set_grad_enabled(False);torch.set_num_threads(4)
class DoubleOps(Ops):
 def tensor(self,a):return torch.tensor(np.asarray(a),device=self.device,dtype=torch.float64)

model=YOLOv4(ROOT/'artifacts/608/yolov4.cfg');values=fixture(model,ROOT/'artifacts/yolov4.weights');values.pop('image')
ops=DoubleOps('torch','cuda',True);inputs={k:ops.tensor(v) for k,v in values.items()}
folder=ROOT/'artifacts/references/608';refs=np.load(folder/'outputs.npz');records=[]
for i,key in enumerate(('ramp','zero')):
 inputs['image']=ops.tensor(image(608,i));y=ops.numpy(model(ops,inputs)).copy();np.save(folder/f'fp64-{key}.npy',y)
 for backend,a in [('torch',refs[key]),('darknet',np.load(folder/f'darknet-{key}.npy'))]:
  delta=np.abs(a-y);record={'input':key,'backend':backend,'max_absolute':float(delta.max()),'violations_1e5_1e4':int(np.count_nonzero(delta>1e-5+1e-4*np.abs(y))),'max_normalized_error':float(np.max(delta/(1e-5+1e-4*np.abs(y))))};records.append(record);print(record,flush=True)
(folder/'fp64-diagnostic.json').write_text(json.dumps(records,indent=2)+'\n')
