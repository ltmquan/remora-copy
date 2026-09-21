"""One GPU, one model replica. Imported only after worker GPU isolation."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'python_backend'))
from direct import YOLOv4
from benchmark import fixture
from runtime import Ops

CHECKPOINT_SHA='e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562'

def image(size, index):
    # Two distinct deterministic images; request IDs remain unique in the scheduler.
    if index % 2: return np.zeros((3,size,size),np.float32)
    return (np.arange(3*size*size,dtype=np.int32)%251).astype(np.float32).reshape(3,size,size)/np.float32(250)

class ReferenceOps(Ops):
    def tensor(self,a):
        return self.T.tensor(np.asarray(a),device=self.device,dtype=self.T.float64)

class Model:
    def __init__(self, implementation, size, weights, compile_model=True, reference_float64=False):
        self.size=size; self.backend='torch' if 'torch' in implementation else 'tinygrad'
        device='cuda' if self.backend=='torch' else 'CUDA'
        os.environ['DEV']=device; os.environ['ALLOW_TF32']='0'; os.environ.setdefault('BEAM','0')
        if self.backend=='torch':
            import torch
            torch.set_grad_enabled(False); torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS','4')))
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=True
        cfg=ROOT/'artifacts'/str(size)/'yolov4.cfg'; direct=YOLOv4(cfg)
        arrays=fixture(direct,weights); arrays.pop('image')
        if self.backend=='tinygrad' and implementation.endswith('im2col'):
            from tinygrad_optimized import OptimizedOps
            self.ops=OptimizedOps(device,'im2col')
        else: self.ops=Ops(self.backend,device,fold_bn=self.backend=='torch')
        if reference_float64:
            if self.backend!='torch' or compile_model: raise ValueError('FP64 is only for eager reference generation')
            self.ops=ReferenceOps('torch',device,fold_bn=True)
        params={k:self.ops.tensor(v) for k,v in arrays.items()}
        if implementation.startswith('remora-'):
            directory=cfg.parent/'original'
            graph=json.loads((directory/'generated_yolov4.json').read_text())
            shapes={k:list(v.shape) for k,v in arrays.items()}; shapes['image']=[3,size,size]
            if shapes!=graph['inputs']: raise ValueError('Source/cfg parameter or image shape mismatch')
            spec=importlib.util.spec_from_file_location('scaled_yolo',directory/'generated_yolov4.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); forward=module.forward
        else: forward=direct
        for div,aw,ah in [(8,[12,19,40],[16,36,28]),(16,[36,76,72],[75,55,146]),(32,[142,192,459],[110,243,401])]:
            self.ops.decode_constants(size//div,aw,ah)
        def run(im):
            values=dict(params); values['image']=im
            return self.ops.realize(forward(self.ops,values))
        self.host=[image(size,i) for i in range(2)]
        self.resident=[self.ops.tensor(im) for im in self.host]
        # Fill persistent folded-weight caches eagerly. Capturing their creation
        # would retain CUDA-graph outputs as parameters of subsequent calls.
        run(self.resident[0]); self.ops.sync()
        if compile_model:
            if self.backend=='torch': run=torch.compile(run,fullgraph=True,mode='reduce-overhead')
            else:
                from tinygrad import TinyJit
                run=TinyJit(run)
        self.run=run
        self.ops.sync()

    def infer(self,index,mode):
        x=self.resident[index%2] if mode=='resident' else self.ops.tensor(self.host[index%2])
        y=self.run(x)
        if mode=='transfer': y=self.ops.numpy(y).copy()
        self.ops.sync()
        return y

    def memory(self):
        if self.backend=='torch':
            import torch
            return {'live_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'scope':'trial since reset, includes persistent allocations'}
        return {'live_peak_bytes':None,'reserved_peak_bytes':None}

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--size',type=int,required=True); p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--reference-dir',type=Path,default=ROOT/'artifacts/references'); a=p.parse_args()
    sha=hashlib.file_digest(a.weights.open('rb'),'sha256').hexdigest()
    if sha!=CHECKPOINT_SHA: raise ValueError('Expected the official checkpoint from the original comparison')
    m=Model('direct-torch',a.size,a.weights,compile_model=False,reference_float64=a.precision=='float64')
    out=a.reference_dir/str(a.size); out.mkdir(parents=True,exist_ok=True)
    values=[m.infer(i,'transfer') for i in range(2)]
    expected=(sum(3*(a.size//d)**2 for d in (8,16,32)),85)
    if any(v.shape!=expected or not np.isfinite(v).all() for v in values): raise ValueError('Invalid reference')
    np.savez(out/'outputs.npz',ramp=values[0],zero=values[1])
    (out/'metadata.json').write_text(json.dumps({'checkpoint_sha256':sha,'size':a.size,'reference':'independent direct PyTorch eager '+a.precision,'precision':a.precision,'cfg_sha256':hashlib.sha256((ROOT/'artifacts'/str(a.size)/'yolov4.cfg').read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__': main()
