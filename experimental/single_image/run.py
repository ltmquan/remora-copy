"""Single-image latency, one process, all devices synchronized per measurement."""
from pathlib import Path
import argparse,json,os,sys,time,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'scaling'))
import model as model_module
from spatial import SpatialOps

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--size',type=int,default=1216)
    p.add_argument('--devices',default='0,1');p.add_argument('--split',default='1,1')
    p.add_argument('--min-side',type=int,default=608)
    p.add_argument('--implementation',choices=['direct-torch','remora-torch'],default='direct-torch')
    p.add_argument('--mode',choices=['spatial','single-compiled'],default='spatial')
    p.add_argument('--cudnn',choices=['on','off'],default='off')
    p.add_argument('--eager-shards',action='store_true')
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--runs',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    import torch
    devices=list(map(int,a.devices.split(',')));weights=list(map(float,a.split.split(',')))
    assert devices[0]==0 and len(set(devices))==len(devices) and len(weights)==len(devices)
    assert a.runs>0 and a.warmup>=2 and all(w>0 for w in weights)
    assert a.size%32==0 and max(devices)<torch.cuda.device_count()
    if a.mode=='single-compiled':assert devices==[0]
    torch.cuda.set_device(0);torch.backends.cudnn.enabled=a.cudnn=='on'
    torch.set_grad_enabled(False)
    # A bounded family of YOLO convolution shapes shares the kernel code object.
    torch._dynamo.config.cache_size_limit=256
    torch._dynamo.config.accumulated_cache_size_limit=4096
    # Construct a separate direct eager model for numerical checks before timing.
    reference=model_module.Model('direct-torch',a.size,a.weights,compile_model=False)
    expected=[reference.infer(i,'transfer').copy() for i in (0,1)]
    del reference
    import gc;gc.collect();torch.cuda.empty_cache()
    if a.mode=='spatial':
        model_module.Ops=lambda *args,**kwargs:SpatialOps(devices,weights,a.min_side,not a.eager_shards)
    started=time.perf_counter()
    m=model_module.Model(a.implementation,a.size,a.weights,compile_model=a.mode=='single-compiled')
    validation=[]
    for i in (0,1):
        for _ in range(a.warmup):m.infer(i,'resident')
        actual=m.infer(i,'transfer');error=np.abs(actual.astype(float)-expected[i].astype(float))
        bad=~np.isfinite(actual)|(error>1e-5+1e-4*np.abs(expected[i]))
        validation.append({'input':i,'failed_values':int(bad.sum()),'max_absolute_error':float(error.max())})
    setup=time.perf_counter()-started
    values=[]
    for _ in range(a.runs):
        m.ops.sync();start=time.perf_counter();m.infer(0,'resident');values.append(time.perf_counter()-start)
    record=vars(a)|{'weights':str(a.weights),'output':str(a.output),'devices':devices,'split':weights,'setup_seconds':setup,'seconds':values,'median_ms':statistics.median(values)*1000,'validation':validation,'passed':all(v['failed_values']==0 for v in validation),'torch':torch.__version__,'gpu_names':[torch.cuda.get_device_name(d) for d in devices],'peer_access':[[torch.cuda.can_device_access_peer(i,j) if i!=j else True for j in devices] for i in devices],'timing_scope':'One image; input/output on GPU0; includes scatter, overlapping input rows, gather, synchronization; excludes setup/warmup/host transfers','precision':'FP32 TF32 off','shard_compilation':a.mode=='spatial' and not a.eager_shards}
    if a.mode=='spatial':
        before_bytes=m.ops.communication_bytes;before_count=m.ops.split_convolutions
        m.infer(0,'resident');record['logical_peer_bytes_per_image']=m.ops.communication_bytes-before_bytes
        record['split_convolutions_per_image']=m.ops.split_convolutions-before_count
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['median_ms','passed','validation']},indent=2))
    if not record['passed']:raise SystemExit(2)
if __name__=='__main__':main()
