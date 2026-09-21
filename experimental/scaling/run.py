"""Single-host data parallel inference, with bounded setup and trial deadlines."""
import argparse, hashlib, json, multiprocessing as mp, os, queue, statistics, subprocess, time, traceback
from pathlib import Path
from partition import assignments
ROOT=Path(__file__).resolve().parent
IMPLEMENTATIONS=['remora-torch','direct-torch','remora-tinygrad','direct-tinygrad','remora-tinygrad-im2col','direct-tinygrad-im2col','remora-futhark-original','remora-futhark-modified','darknet-dense']

def worker(rank,gpu,a,commands,replies,counter):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    threads=str(max(1,min(4,(os.cpu_count() or 1)//len(a.gpus))))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(name,threads)
    try:
        import numpy as np
        from model import Model
        started=time.perf_counter()
        if 'futhark' in a.implementation:
            from native import Futhark
            m=Futhark(a.implementation,a.size,a.weights)
        elif a.implementation=='darknet-dense':
            from native import Darknet
            m=Darknet(a.implementation,a.size,a.weights)
        else: m=Model(a.implementation,a.size,a.weights)
        # Warm both signatures, then validate replay on both distinct inputs.
        for _ in range(3):
            for i in range(2): m.infer(i,a.mode)
        refs=np.load(a.reference_dir/str(a.size)/'outputs.npz'); validation=[]
        for i,key in enumerate(('ramp','zero')):
            actual=m.infer(i,a.mode)
            if a.mode=='resident': actual=m.ops.numpy(actual).copy()
            if not np.isfinite(actual).all(): raise ValueError('Nonfinite output')
            delta=np.abs(actual-refs[key])
            validation.append({'input':key,'max_absolute_error':float(delta.max()),'strict_violations':int(np.count_nonzero(delta>1e-5+1e-4*np.abs(refs[key]))),'atol':a.atol,'rtol':a.rtol})
            np.testing.assert_allclose(actual,refs[key],atol=a.atol,rtol=a.rtol)
        ts=[]
        for i in range(6):
            t=time.perf_counter(); m.infer(i,a.mode); ts.append(time.perf_counter()-t)
        # Verify actual CUDA device identity before accepting a multi-GPU run.
        import ctypes as C
        driver=C.CDLL('libcuda.so.1'); dev=C.c_int(); uuid=(C.c_ubyte*16)()
        if driver.cuDeviceGet(C.byref(dev),0): raise RuntimeError('Cannot resolve CUDA device')
        get_uuid=getattr(driver,'cuDeviceGetUuid_v2',driver.cuDeviceGetUuid)
        if get_uuid(C.byref(uuid),dev): raise RuntimeError('Cannot resolve GPU UUID')
        raw=bytes(uuid).hex(); gpu_uuid='GPU-'+ '-'.join([raw[:8],raw[8:12],raw[12:16],raw[16:20],raw[20:]])
        replies.put(('ready',rank,{'compilation_cache':getattr(m,'cache_state',None),'gpu_uuid':gpu_uuid,'validation':validation,'setup_seconds':time.perf_counter()-started,'rate':1/statistics.median(ts),'pid':os.getpid()}))
        while True:
            cmd=commands.get()
            if cmd is None: break
            ids=cmd
            if m.backend=='torch': m.ops.T.cuda.reset_peak_memory_stats()
            replies.put(('armed',rank,None))
            start_at=commands.get()
            while time.perf_counter()<start_at: time.sleep(min(.001,max(0,start_at-time.perf_counter())))
            started=time.perf_counter(); completed=[]
            if a.policy=='dynamic':
                while True:
                    with counter.get_lock():
                        first=counter.value; counter.value+=a.chunk
                    if first>=a.images: break
                    for index in range(first,min(first+a.chunk,a.images)):
                        m.infer(index,a.mode); completed.append(index)
            else:
                for index in ids: m.infer(index,a.mode); completed.append(index)
            memory_replays=0
            if a.sample_memory:
                while time.perf_counter()-started<a.memory_min_seconds:
                    m.infer(memory_replays%2,a.mode); memory_replays+=1
            finished=time.perf_counter()
            replies.put(('done',rank,{'memory_replays':memory_replays,'ids':completed,'start':started,'finish':finished,'memory':m.memory()}))
    except BaseException:
        replies.put(('error',rank,traceback.format_exc()))

def receive(replies,processes,event,count,timeout):
    deadline=time.monotonic()+timeout; values={}
    while len(values)<count:
        if time.monotonic()>deadline: raise TimeoutError(f'{event} exceeded explicit {timeout}s limit')
        try: kind,rank,data=replies.get(timeout=.2)
        except queue.Empty:
            if any(p.exitcode is not None for p in processes): raise RuntimeError('Worker exited before completing '+event)
            continue
        if kind=='error': raise RuntimeError(f'GPU worker {rank}: {data}')
        if kind!=event or rank in values: raise RuntimeError(f'Unexpected worker event {kind}')
        values[rank]=data
    return [values[i] for i in range(count)]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--implementation',choices=IMPLEMENTATIONS,required=True)
    p.add_argument('--size',type=int,default=608); p.add_argument('--gpus',nargs='+',default=['0'])
    p.add_argument('--images',type=int,default=128); p.add_argument('--trials',type=int,default=5)
    p.add_argument('--policy',choices=['uniform','round-robin','weighted','dynamic'],default='uniform')
    p.add_argument('--chunk',type=int,default=1); p.add_argument('--mode',choices=['resident','transfer'],default='resident')
    p.add_argument('--weights',type=Path,required=True); p.add_argument('--reference-dir',type=Path,default=ROOT/'artifacts/references')
    p.add_argument('--atol',type=float,default=1e-5); p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--setup-timeout',type=float,default=1800); p.add_argument('--trial-timeout',type=float,default=600)
    p.add_argument('--memory-min-seconds',type=float,default=3)
    p.add_argument('--sample-memory',action='store_true',help='Separate memory pass; do not use these timings for the main table')
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not (0<=a.atol<float('inf') and 0<=a.rtol<float('inf')): p.error('Invalid correctness tolerances')
    if a.memory_min_seconds<=0: p.error('Memory sampling duration must be positive')
    if a.setup_timeout<=0 or a.trial_timeout<=0: p.error('Timeouts must be positive')
    if a.implementation=='darknet-dense' and a.mode!='transfer': p.error('Native Darknet supports transfer-inclusive timing only')
    if min(a.images,a.trials,a.chunk)<1 or a.size<32 or a.size%32 or len(set(a.gpus))!=len(a.gpus): p.error('Invalid counts, dimensions, or duplicate GPUs')
    result={'status':'starting','config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'timing':'synchronized whole-request makespan; setup and correctness excluded','memory_scope':'see per-worker scope: PyTorch trial reset, Futhark context lifetime; includes persistent model and two resident inputs'}
    ctx=mp.get_context('spawn'); replies=ctx.Queue(); counter=ctx.Value('i',0); commands=[]; processes=[]
    try:
        metadata=json.loads((a.reference_dir/str(a.size)/'metadata.json').read_text())
        sha=hashlib.file_digest(a.weights.open('rb'),'sha256').hexdigest()
        cfg=ROOT/'artifacts'/str(a.size)/'yolov4.cfg'
        if metadata['checkpoint_sha256']!=sha or metadata['cfg_sha256']!=hashlib.sha256(cfg.read_bytes()).hexdigest(): raise ValueError('Reference checkpoint/config mismatch')
        result['reference']=metadata
        result['environment']={key:os.getenv(key) for key in ['BEAM','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']}
        result['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.glob('*.py'))}
        result['model_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT/'artifacts'/str(a.size)).rglob('*')) if path.is_file() and path.suffix in ('.json','.py','.so','.cfg','.remora')}
        result['backend_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT.parent/'python_backend').glob('*.py'))}
        for rank,gpu in enumerate(a.gpus):
            q=ctx.Queue(); commands.append(q); proc=ctx.Process(target=worker,args=(rank,gpu,a,q,replies,counter)); proc.start(); processes.append(proc)
        ready=receive(replies,processes,'ready',len(processes),a.setup_timeout); result['workers']=ready
        if len({w['gpu_uuid'] for w in ready})!=len(ready): raise RuntimeError('Multiple workers landed on the same GPU')
        allocation=assignments(a.images,len(processes),'weighted' if a.policy=='weighted' else ('round-robin' if a.policy=='round-robin' else 'uniform'),[r['rate'] for r in ready])
        result['trials']=[]
        result['measurement_kind']='memory' if a.sample_memory else 'timing'
        for trial in range(a.trials):
            counter.value=0
            for q,ids in zip(commands,allocation): q.put(ids)
            receive(replies,processes,'armed',len(processes),30)
            sampler=None
            if a.sample_memory:
                from memory import Sampler
                sampler=Sampler([w['pid'] for w in ready]); sampler.start()
            start=time.perf_counter()+.1
            for q in commands: q.put(start)
            try: done=receive(replies,processes,'done',len(processes),a.trial_timeout)
            finally: footprint=sampler.finish() if sampler else None
            if sampler and (footprint['samples']<2 or len(footprint['process_peak_bytes_by_gpu'])!=len(processes)):
                raise RuntimeError('Insufficient memory samples or missing GPU processes')
            ids=[i for w in done for i in w['ids']]
            if sorted(ids)!=list(range(a.images)): raise ValueError('Missing or duplicated requests')
            elapsed=max(w['finish'] for w in done)-start
            result['trials'].append({'seconds':elapsed,'images_per_second':None if a.sample_memory else a.images/elapsed,'workers':done,'footprint':footprint})
        result['median_images_per_second']=None if a.sample_memory else statistics.median(t['images_per_second'] for t in result['trials'])
        result['status']='passed'
    except BaseException as e:
        result['status']='timeout' if isinstance(e,TimeoutError) else 'failed'; result['error']=traceback.format_exc()
    finally:
        for q in commands: q.put(None)
        for proc in processes:
            proc.join(timeout=2)
            if proc.is_alive(): proc.terminate(); proc.join(timeout=3)
            if proc.is_alive(): proc.kill(); proc.join()
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],a.output,flush=True)
    if result['status']!='passed': raise SystemExit(1)
if __name__=='__main__': main()
