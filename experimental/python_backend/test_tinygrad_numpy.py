"""GPU-independent checks of patch order, padding and fixed-weight caching."""
import os,unittest
import numpy as np
from tinygrad_optimized import OptimizedOps

class NumpyReferenceTests(unittest.TestCase):
    def test_convolution_variants(self):
        rng=np.random.default_rng(9)
        for variant in ['im2col','winograd']:
            for ic,n,oc,k,stride,pad in [(3,8,4,3,2,1),(4,7,6,1,1,0),(3,9,5,3,1,1)]:
                with self.subTest(variant=variant,k=k,stride=stride):
                    ops=OptimizedOps(os.getenv('TEST_DEVICE','CPU'),variant)
                    arrays={'w':rng.normal(size=(oc,ic,k,k)).astype('f4')*.1,'bias':rng.normal(size=oc).astype('f4'),
                        'scale':rng.normal(size=oc).astype('f4'),'mean':rng.normal(size=oc).astype('f4'),'variance':rng.uniform(.1,2,size=oc).astype('f4')}
                    params={key:ops.tensor(v) for key,v in arrays.items()}
                    factor=arrays['scale']/np.sqrt(arrays['variance']+np.float32(1e-5))
                    w=arrays['w']*factor[:,None,None,None];b=arrays['bias']-factor*arrays['mean']
                    out=(n+2*pad-k)//stride+1
                    for _ in range(2):
                        x=rng.normal(size=(ic,n,n)).astype('f4')
                        patches=np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(pad,pad),(pad,pad))),(k,k),axis=(1,2))[:,::stride,::stride]
                        expected=np.einsum('oikl,iyzkl->oyz',w,patches)+b[:,None,None]
                        expected=expected*np.tanh(np.log(np.exp(expected)+1))
                        actual=ops.numpy(ops.conv(ops.tensor(x),params,stride,pad,'mish',[ic,n,oc,out,k]).realize())
                        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4)
if __name__=='__main__':unittest.main()
