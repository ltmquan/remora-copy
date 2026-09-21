"""Validate and time the full native-lowered network with live input tensors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch

root=Path.cwd();assets=Path('/scratch/luu.qu/remora-python-20260911')
sys.path.insert(0,str(assets))
from benchmark import fixture
from direct import YOLOv4

torch.set_grad_enabled(False);torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
reference_report=json.loads((root/'full-baselines.json').read_text());assert reference_report['passed']
model=YOLOv4(assets/'yolov4.cfg');arrays=fixture(model,assets/'yolov4.weights')
names=json.loads((root/'yolov4-entry-names.json').read_text());assert set(names)==set(arrays)
args=[torch.from_numpy(arrays[k]).cuda() for k in names]
path=root/'yolov4-native.py';spec=importlib.util.spec_from_file_location('native_yolo',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.LOWERING=='native'
expected=np.fromfile(root/'full-direct.bin',dtype='<f4').reshape(22743,85)
torch.cuda.synchronize();start=time.perf_counter();actual=m.main(*args);torch.cuda.synchronize();first=time.perf_counter()-start
result=actual.cpu().numpy();assert result.shape==expected.shape
error=np.abs(result.astype(np.float64)-expected.astype(np.float64));limit=1e-5+1e-4*np.abs(expected.astype(np.float64))
assert np.isfinite(result).all() and (error<=limit).all(),f'max difference {error.max()}'
np.save(root/'native-full-output.npy',result)
for _ in range(3):m.main(*args);torch.cuda.synchronize()
samples=[]
for _ in range(10):
 torch.cuda.synchronize();start=time.perf_counter();y=m.main(*args);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
# Ensure the generated graph consumes changed data, not a captured reference result.
changed=args.copy();changed[0]=torch.zeros_like(args[0]);delta=(m.main(*changed)-actual).abs().max().item();assert delta>1e-5
report={'passed':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'lowering':m.LOWERING,'source_sha256':reference_report['source_sha256'],'generated_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_convolutions':m.NATIVE_OPS.count('conv'),'shape':list(result.shape),'scalars':result.size,'max_abs':float(error.max()),'max_error_over_tolerance':float((error/limit).max()),'changed_image_max_delta':delta,'first_seconds':first,'warmups':3,'runs':10,'samples_ms':samples,'median_ms':statistics.median(samples),'direct_median_ms':reference_report['direct_pytorch_eager_median_ms'],'futhark_median_ms':reference_report['remora_futhark_median_ms']}
(root/'native-full-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
