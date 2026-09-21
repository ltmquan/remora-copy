"""Check the new im2col lowering against cuDNN on signed data and padding edges."""
import os,unittest
os.environ['ALLOW_TF32']='0'
import numpy as np
from runtime import Ops
from tinygrad_optimized import OptimizedOps

class OptimizedConvolutionTests(unittest.TestCase):
    def test_shapes_activations_and_repeated_input(self):
        import torch
        torch.set_grad_enabled(False);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        rng=np.random.default_rng(20260911)
        for ic,n,oc,k,stride,pad,activation in [(3,8,4,3,2,1,'mish'),(4,7,6,1,1,0,'leaky'),(3,9,5,3,1,1,'linear')]:
            with self.subTest(k=k,stride=stride,activation=activation):
                base=Ops('torch','cuda',fold_bn=True);opt=OptimizedOps('CUDA','im2col')
                arrays={'w':rng.normal(size=(oc,ic,k,k)).astype('f4'),'bias':rng.normal(size=oc).astype('f4'),
                    'scale':rng.normal(size=oc).astype('f4'),'mean':rng.normal(size=oc).astype('f4'),'variance':rng.uniform(.1,2,size=oc).astype('f4')}
                params=[{key:o.tensor(v) for key,v in arrays.items()} for o in [base,opt]]
                out=(n+2*pad-k)//stride+1
                for _ in range(2):
                    x=rng.normal(size=(ic,n,n)).astype('f4');results=[]
                    for o,p in zip([base,opt],params):
                        y=o.realize(o.conv(o.tensor(x),p,stride,pad,activation,[ic,n,oc,out,k]));o.sync();results.append(o.numpy(y))
                    np.testing.assert_allclose(results[1],results[0],atol=2e-5,rtol=2e-4)
if __name__=='__main__':unittest.main()
