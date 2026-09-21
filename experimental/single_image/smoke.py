"""CPU checks for stride alignment and exact spatial halo coverage (NumPy only)."""
import ast
from pathlib import Path
import numpy as np
# Load only the pure partition function; no Torch installation needed.
tree=ast.parse((Path(__file__).parent/'spatial.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='partitions')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'spatial.py','exec'),ns)
partitions=ns['partitions']
def conv(x,w,stride):
    windows=np.lib.stride_tricks.sliding_window_view(x,w.shape)[::stride,::stride]
    return np.einsum('xyij,ij->xy',windows,w)
rng=np.random.default_rng(42);checks=0
for n in (19,38,76):
 for k in (1,3):
  for stride in (1,2):
   pad=k//2;x=rng.normal(size=(n,n));w=rng.normal(size=(k,k));x=np.pad(x,pad)
   ref=conv(x,w,stride);rows=ref.shape[0]
   for weights in ([1],[1,1],[1,2],[2,1],[1,1,1,1]):
    ranges=partitions(rows,weights)
    assert ranges[0][0]==0 and ranges[-1][1]==rows
    parts=[conv(x[lo*stride:(hi-1)*stride+k,:],w,stride) for lo,hi in ranges]
    np.testing.assert_allclose(np.concatenate(parts,axis=0),ref,rtol=0,atol=0)
    checks+=1
print(f'{checks} partition/halo convolution checks passed')
