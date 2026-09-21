"""FP32 tensor operations shared by the source-generated and direct models."""
import numpy as np

class Ops:
    def __init__(self, backend, device, fold_bn=False):
        self.backend, self.device, self.fold_bn = backend, device, fold_bn
        self.fused = {}
        self.constants = {}
        if backend == 'torch':
            import torch
            import torch.nn.functional as F
            self.T, self.F = torch, F
        else:
            from tinygrad import Tensor
            self.T = Tensor

    def tensor(self, a):
        a = np.asarray(a, dtype=np.float32)
        if self.backend == 'torch': return self.T.tensor(a, device=self.device)
        return self.T(a, device=self.device).realize()

    def sync(self):
        if self.backend == 'torch':
            if self.device.startswith('cuda'): self.T.cuda.synchronize()
        else:
            from tinygrad import Device
            Device[self.device].synchronize()

    def realize(self, x):
        return x if self.backend == 'torch' else x.realize()

    def numpy(self, x):
        if self.backend == 'torch': return x.detach().cpu().numpy()
        return x.numpy()

    def cat(self, xs, axis):
        if self.backend == 'torch': return self.T.cat(xs,dim=axis)
        return xs[0].cat(*xs[1:],dim=axis)

    def add(self,a,b): return a+b

    def conv(self, x, p, stride, pad, activation, shape):
        ic,n,oc,out,k = shape
        if tuple(x.shape) != (ic,n,n) or tuple(p['w'].shape) != (oc,ic,k,k):
            raise ValueError(f'Convolution shape mismatch: {x.shape}, {p["w"].shape}, {shape}')
        key = id(p['w'])
        if self.fold_bn and key in self.fused:
            w,b = self.fused[key]
        else:
            w,b = p['w'],p['bias']
            if 'variance' in p:
                denom = (p['variance']+1e-5).sqrt()
                w = w*(p['scale']/denom).reshape(oc,1,1,1)
                b = b-(p['scale']*p['mean'])/denom
            if self.fold_bn:
                w,b = self.realize(w),self.realize(b)
                self.fused[key] = (w,b)
        if self.backend == 'torch':
            y = self.F.conv2d(x.unsqueeze(0),w,b,stride=stride,padding=pad)[0]
        else:
            y = x.unsqueeze(0).conv2d(w,b,stride=stride,padding=pad)[0]
        if activation == 'mish': y = y*(y.exp()+1).log().tanh()
        elif activation == 'leaky':
            y = self.T.maximum(y,y*0.1) if self.backend == 'torch' else y.maximum(y*0.1)
        return y

    def spp(self,x):
        def pool(k):
            if self.backend == 'torch':
                return self.F.max_pool2d(self.F.pad(x,(k//2,)*4,value=-1e30),k,stride=1)
            return x.pad(((0,0),(k//2,k//2),(k//2,k//2)),value=-1e30).max_pool2d(k,stride=1)
        return self.cat([pool(13),pool(9),pool(5),x],0)

    def upsample(self,x,s):
        c,h,w = x.shape
        return x.reshape(c,h,1,w,1).expand(c,h,s,w,s).reshape(c,h*s,w*s)

    def decode_constants(self,s,aw,ah):
        key=(s,tuple(aw),tuple(ah))
        if key not in self.constants:
            self.constants[key] = (
                self.tensor(np.arange(s,dtype=np.float32).reshape(1,s,1)),
                self.tensor(np.arange(s,dtype=np.float32).reshape(1,1,s)),
                self.tensor(aw).reshape(3,1,1),self.tensor(ah).reshape(3,1,1))
        return self.constants[key]

    def decode(self,x,aw,ah,sxy,net,classes):
        s=x.shape[-1]; z=x.reshape(3,classes+5,s,s)
        def exp(v): return 2.718281828459045 ** v
        def sig(v): return 1/(1+exp(-v))
        row,col,a,b=self.decode_constants(s,aw,ah)
        shift=-0.5*(sxy-1)
        planes=[((col+(sig(z[:,0])*sxy+shift))/s).unsqueeze(1),
                ((row+(sig(z[:,1])*sxy+shift))/s).unsqueeze(1),
                (exp(z[:,2])*a/net).unsqueeze(1),
                (exp(z[:,3])*b/net).unsqueeze(1),sig(z[:,4:])]
        y=self.cat(planes,1).reshape(3,classes+5,s*s)
        y=y.permute(0,2,1)
        return y.reshape(3*s*s,classes+5)
