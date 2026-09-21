import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from direct import YOLOv4
from runtime import Ops

os.environ['ALLOW_TF32']='0'
ROOT=Path(__file__).resolve().parent

def fixture(model,weights=None):
    """Darknet-order f32 arrays; synthetic constants match the existing C++ fixture."""
    values={'image':(np.arange(3*608*608,dtype=np.int32)%251).astype(np.float32).reshape(3,608,608)/np.float32(250)}
    cpp=(ROOT/'darknet_benchmark.cpp').read_text()
    factors={int(i):np.float32(f) for i,f in re.findall(r'net.layers\[(\d+)\].weights\[j\]=\(\(int\)\(j%23\)-11\)\*([0-9.e+-]+)f;',cpp)}
    f=open(weights,'rb') if weights else None
    if f:
        major,minor,revision=np.fromfile(f,dtype='<i4',count=3)
        np.fromfile(f,dtype='<i8' if major*10+minor>=2 and major<1000 and minor<1000 else '<i4',count=1)
    for i,(oc,ic,k,bn) in model.parameters.items():
        for name in ['bias']+(['scale','mean','var'] if bn else [])+['w']:
            shape=(oc,ic,k,k) if name=='w' else (oc,)
            if f:
                a=np.fromfile(f,dtype='<f4',count=int(np.prod(shape)))
                if a.size!=np.prod(shape): raise ValueError('Truncated weights')
                a=a.reshape(shape)
            elif name=='w': a=((np.arange(np.prod(shape),dtype=np.int32)%23-11).astype(np.float32)*factors[i]).reshape(shape)
            else: a=np.full(shape,{'bias':0.01,'scale':1,'mean':0,'var':1}[name],np.float32)
            values[f'{name}-{i:03}']=a
    if f:
        if f.read(1): raise ValueError('Unused checkpoint data')
        f.close()
    return values

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['torch','tinygrad'],required=True)
    p.add_argument('--model',choices=['remora','direct'],required=True)
    p.add_argument('--mode',choices=['eager','compile','jit'],default='eager')
    p.add_argument('--compile-mode',choices=['default','reduce-overhead'],default='default')
    p.add_argument('--tinygrad-opt',choices=['baseline','fold','materialize','im2col','winograd'],default=os.getenv('TINYGRAD_VARIANT','baseline'))
    p.add_argument('--device'); p.add_argument('--weights'); p.add_argument('--fold-bn',action='store_true')
    p.add_argument('--runs',type=int,default=10); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--output',required=True)
    a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    device=a.device or ('cuda' if a.backend=='torch' else 'CUDA')
    if a.backend=='tinygrad': os.environ['DEV']=device
    setup=time.perf_counter()
    if a.backend=='torch':
        import torch
        torch.set_grad_enabled(False); torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS','4')))
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=True
    model=YOLOv4(ROOT/'yolov4.cfg'); arrays=fixture(model,a.weights)
    graph=json.loads((ROOT/'generated_yolov4.json').read_text())
    if {k:list(v.shape) for k,v in arrays.items()}!=graph['inputs']: raise ValueError('Independent cfg and Remora parameter shapes disagree')
    input_hash=hashlib.sha256()
    for k,v in arrays.items(): input_hash.update(k.encode()); input_hash.update(v.tobytes())
    if a.tinygrad_opt!='baseline':
        if a.backend!='tinygrad':raise ValueError('--tinygrad-opt requires tinygrad')
        from tinygrad_optimized import OptimizedOps
        ops=OptimizedOps(device,a.tinygrad_opt)
    else:ops=Ops(a.backend,device,a.fold_bn)
    inputs={k:ops.tensor(v) for k,v in arrays.items()}
    if a.model=='remora':
        from generated_yolov4 import forward
    else: forward=model
    # Populate static decode constants before compilation/capture.
    for side,aw,ah in [(76,[12,19,40],[16,36,28]),(38,[36,76,72],[75,55,146]),(19,[142,192,459],[110,243,401])]:
        ops.decode_constants(side,aw,ah)
    def run(image):
        args=dict(inputs); args['image']=image
        return ops.realize(forward(ops,args))
    ops.sync(); setup=time.perf_counter()-setup
    print(f'setup_seconds {setup:.6f}',flush=True)
    print('first inference started',flush=True)
    start=time.perf_counter(); y=run(inputs['image']); ops.sync(); first=time.perf_counter()-start
    print(f'first_eager_seconds {first:.6f}',flush=True)
    if a.mode=='compile':
        if a.backend!='torch': raise ValueError('compile requires torch')
        run=torch.compile(run,fullgraph=True,mode=a.compile_mode)
    if a.mode=='jit':
        if a.backend!='tinygrad': raise ValueError('jit requires tinygrad')
        from tinygrad import TinyJit
        run=TinyJit(run)
    warm=[]
    for _ in range(a.warmup):
        start=time.perf_counter(); y=run(inputs['image']); ops.sync(); warm.append(time.perf_counter()-start)
        print(f'warmup {_}: {warm[-1]:.6f}s',flush=True)
    print("MEMORY_PHASE resident",flush=True)
    times=[]
    for _ in range(a.runs):
        ops.sync(); start=time.perf_counter(); y=run(inputs['image']); ops.sync(); times.append(time.perf_counter()-start)
    result=ops.numpy(y).copy()
    # Transfer-inclusive measurement matches Python input upload and full dense
    # output download; Darknet transfers only head outputs then decodes on CPU.
    transfer_warmup=[]
    # A fresh image can change torch.compile alias guards. Warm that call
    # signature separately so its compilation is not charged to inference.
    for _ in range(a.warmup):
        ops.sync(); start=time.perf_counter(); im=ops.tensor(arrays['image']); yy=run(im); ops.numpy(yy); ops.sync()
        transfer_warmup.append(time.perf_counter()-start)
    print("MEMORY_PHASE transfer",flush=True)
    end_to_end=[]
    for _ in range(a.runs):
        ops.sync(); start=time.perf_counter(); im=ops.tensor(arrays['image']); yy=run(im); ops.numpy(yy); ops.sync(); end_to_end.append(time.perf_counter()-start)
    result.astype('<f4').tofile(out.with_suffix('.bin'))
    if result.shape!=(22743,85) or not np.isfinite(result).all(): raise ValueError('Invalid dense output')
    print("MEMORY_PHASE validation",flush=True)
    # Check that graph capture still consumes a different image, outside timing.
    changed=ops.tensor(np.zeros_like(arrays['image']))
    changed_actual=ops.numpy(run(changed)).copy()
    changed_args=dict(inputs); changed_args['image']=changed
    changed_expected=ops.numpy(ops.realize(forward(ops,changed_args))).copy()
    np.testing.assert_allclose(changed_actual,changed_expected,atol=1e-5,rtol=1e-4)
    changed_delta=float(np.max(np.abs(changed_actual-result)))
    if a.weights and changed_delta<=1e-5: raise ValueError('Trained model did not respond to changed image')
    report=vars(a)|{'effective_fold_bn':ops.fold_bn,'changed_input_max_delta':changed_delta,'changed_input_validation':'passed','setup_seconds':setup,'first_eager_seconds':first,'warmup_seconds':warm,
        'run_seconds':times,'median_seconds':float(np.median(times)),
        'transfer_warmup_seconds':transfer_warmup,'end_to_end_seconds':end_to_end,'end_to_end_median_seconds':float(np.median(end_to_end)),
        'tinygrad_beam':os.getenv('BEAM','0'),'input_sha256':input_hash.hexdigest(),'remora_ast_sha256':graph['ast_sha256'],
        'versions':{k:importlib.metadata.version(k) for k in ['numpy', 'torch', 'tinygrad']},
        'precision':'float32; torch TF32 disabled; tinygrad ALLOW_TF32=0', 'device':device}
    if a.backend=='torch': report.update(gpu=torch.cuda.get_device_name() if device.startswith('cuda') else device,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
