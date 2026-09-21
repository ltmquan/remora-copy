"""Partition convolution output rows; retain the full graph on the primary GPU."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python_backend'))
from runtime import Ops

def partitions(rows, weights):
    if rows < len(weights) or any(w <= 0 for w in weights):
        raise ValueError('Positive weights and at least one output row per GPU required')
    edges=[0]
    for i in range(1,len(weights)):
        edge=round(rows*sum(weights[:i])/sum(weights))
        edges.append(max(edges[-1]+1,min(edge,rows-(len(weights)-i))))
    edges.append(rows)
    return list(zip(edges,edges[1:]))

class SpatialOps(Ops):
    def __init__(self, devices, weights, min_side, compile_kernels=True):
        super().__init__('torch',f'cuda:{devices[0]}',fold_bn=True)
        self.devices=devices; self.weights=weights; self.min_side=min_side
        self.compile_kernels=compile_kernels; self.kernels={}; self.remote_weights={}
        self.communication_bytes=0; self.split_convolutions=0

    def sync(self):
        for d in self.devices: self.T.cuda.synchronize(d)

    def conv(self,x,p,stride,pad,activation,shape):
        ic,n,oc,out,k=shape
        assert tuple(x.shape)==(ic,n,n) and tuple(p['w'].shape)==(oc,ic,k,k)
        key=id(p['w'])
        if key not in self.fused:
            w,b=p['w'],p['bias']
            if 'variance' in p:
                denom=(p['variance']+1e-5).sqrt()
                w=w*(p['scale']/denom).reshape(oc,1,1,1)
                b=b-p['scale']*p['mean']/denom
            self.fused[key]=(w,b)
        w,b=self.fused[key]
        split=len(self.devices)>1 and n>=self.min_side and out>=len(self.devices)
        devices=self.devices if split else self.devices[:1]
        ranges=partitions(out,self.weights) if split else [(0,out)]
        # Pad only the global boundary. Interior cuts receive real neighboring rows.
        padded=self.F.pad(x,(pad,pad,pad,pad)).unsqueeze(0)
        ready=self.T.cuda.Event();ready.record(self.T.cuda.current_stream(self.devices[0]))
        outputs=[]
        for d,(lo,hi) in zip(devices,ranges):
            with self.T.cuda.device(d):
                stream=self.T.cuda.current_stream(d);stream.wait_event(ready)
                wk=(key,d)
                if wk not in self.remote_weights:
                    self.remote_weights[wk]=(w.to(d,non_blocking=True),b.to(d,non_blocking=True))
                ww,bb=self.remote_weights[wk]
                patch=padded[:,:,lo*stride:(hi-1)*stride+k,:].contiguous().to(d,non_blocking=True)
                kernel_key=(d,key,lo,hi)
                if kernel_key not in self.kernels:
                    def kernel(xx,ww,bb,stride=stride,activation=activation):
                        yy=self.F.conv2d(xx,ww,bb,stride=stride)
                        if activation=='mish':yy=yy*(yy.exp()+1).log().tanh()
                        elif activation=='leaky':yy=self.T.maximum(yy,yy*.1)
                        return yy
                    self.kernels[kernel_key]=self.T.compile(kernel,fullgraph=True) if self.compile_kernels else kernel
                yy=self.kernels[kernel_key](patch,ww,bb)
                done=self.T.cuda.Event();done.record(stream)
                outputs.append((yy,done))
                if d!=devices[0]:self.communication_bytes+=(patch.numel()+yy.numel())*4
        with self.T.cuda.device(devices[0]):
            for _,done in outputs:self.T.cuda.current_stream(devices[0]).wait_event(done)
            result=self.T.cat([yy.to(devices[0],non_blocking=True) for yy,_ in outputs],dim=2)[0]
        if split:self.split_convolutions+=1
        assert tuple(result.shape)==(oc,out,out)
        return result
