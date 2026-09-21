"""Opt-in FP32 tinygrad lowerings; no torch, cuDNN, or external CUDA kernels.

im2col makes convolution's reused patch matrix explicit and contiguous so
matrix multiplication can use a regular reduction rather than padded indexing.
Winograd is an experimental alternative and must pass the same output tolerance.
"""
from runtime import Ops

class OptimizedOps(Ops):
    def __init__(self,device,variant):
        super().__init__('tinygrad',device,fold_bn=True)
        self.variant=variant

    def conv(self,x,p,stride,pad,activation,shape):
        if self.variant in ('fold','materialize','winograd'):
            if self.variant=='winograd':
                from tinygrad.helpers import Context
                with Context(WINO=1):y=super().conv(x,p,stride,pad,activation,shape)
            else:y=super().conv(x,p,stride,pad,activation,shape)
            return y.contiguous() if self.variant=='materialize' else y
        ic,n,oc,out,k=shape
        if tuple(x.shape)!=(ic,n,n) or tuple(p['w'].shape)!=(oc,ic,k,k):raise ValueError('Convolution shape mismatch')
        key=id(p['w'])
        if key not in self.fused:
            w,b=p['w'],p['bias']
            if 'variance' in p:
                factor=p['scale']/(p['variance']+1e-5).sqrt()
                w=w*factor.reshape(oc,1,1,1);b=b-factor*p['mean']
            self.fused[key]=(w.reshape(oc,ic*k*k).contiguous().realize(),b.realize())
        w,b=self.fused[key]
        # Shape (C, OH, OW, KH, KW), then (C*KH*KW, OH*OW).
        if k==1 and stride==1 and pad==0:columns=x.reshape(ic,n*n)
        else:
            patches=x.pad(((0,0),(pad,pad),(pad,pad)))._pool((k,k),stride=stride)
            columns=patches.permute(0,3,4,1,2).reshape(ic*k*k,out*out).contiguous()
        y=(w@columns).reshape(oc,out,out)+b.reshape(oc,1,1)
        if activation=='mish':y=y*(y.exp()+1).log().tanh()
        elif activation=='leaky':y=y.maximum(y*0.1)
        return y.contiguous()
