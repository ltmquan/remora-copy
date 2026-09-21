"""Differential operator tests on signed, nonconstant data on the allocated GPU."""
import unittest
import os
os.environ["ALLOW_TF32"]="0"
import numpy as np
from runtime import Ops

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_grad_enabled(False)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        cls.ops=[Ops('torch','cuda'),Ops('tinygrad','CUDA')]
        cls.rng=np.random.default_rng(42)

    def check(self,fn,arrays):
        values=[]
        for ops in self.ops:
            xs=[ops.tensor(x) for x in arrays]
            y=ops.realize(fn(ops,*xs));ops.sync();values.append(ops.numpy(y))
        np.testing.assert_allclose(values[1],values[0],atol=2e-5,rtol=2e-4)

    def test_strided_conv_with_bn(self):
        arrays=[self.rng.normal(size=shape).astype(np.float32) for shape in [(3,8,8),(4,3,3,3),(4,),(4,),(4,)]]
        var=self.rng.uniform(0.1,2,size=4).astype(np.float32)
        self.check(lambda o,x,w,b,m,sc,v:o.conv(x,{'w':w,'bias':b,'mean':m,'scale':sc,'variance':v},2,1,'mish',[3,8,4,4,3]),arrays+[var])

    def test_spp_negative_padding(self):
        self.check(lambda o,x:o.spp(x),[-self.rng.uniform(0.1,5,size=(2,15,15)).astype(np.float32)])

    def test_upsample_order(self):
        self.check(lambda o,x:o.upsample(x,2),[np.arange(18,dtype=np.float32).reshape(2,3,3)])

    def test_decode_anchor_order(self):
        self.check(lambda o,x:o.decode(x,[12,19,40],[16,36,28],1.2,608,2),[self.rng.normal(size=(21,3,3)).astype(np.float32)])

if __name__=='__main__': unittest.main()
