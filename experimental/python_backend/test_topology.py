"""Independent topology equivalence, without either GPU runtime.

Canonical DAG hashes include parameters, layer connectivity, stride/padding,
activation and anchors. This catches shared-runtime comparisons accidentally
running the same network twice after an incorrect source translation.
"""
import hashlib
import json
from pathlib import Path
import unittest
from direct import YOLOv4
from generated_yolov4 import forward

ROOT=Path(__file__).resolve().parent

def canonical(x):
    if isinstance(x,Node): return x.key
    if isinstance(x,(int,float)): return float(x)
    if isinstance(x,(list,tuple)): return [canonical(v) for v in x]
    if isinstance(x,dict): return {k:canonical(v) for k,v in x.items()}
    return x

class Node:
    def __init__(self,op,*args):
        self.op,self.args=op,args
        self.key=hashlib.sha256(json.dumps(canonical([op,args]),sort_keys=True).encode()).hexdigest()
    def __add__(self,other): return Node('add',self,other)

class Trace:
    backend='torch'
    def __init__(self): self.F=self
    def conv(self,*args): return Node('conv',*args)
    def add(self,*args): return Node('add',*args)
    def cat(self,xs,axis):
        flat=[]
        for x in xs:
            if x.op=='cat' and x.args[1]==axis: flat.extend(x.args[0])
            else: flat.append(x)
        if len(flat)==1: return flat[0]
        return Node('cat',flat,axis)
    def pad(self,x,padding,value): return Node('pad',x,padding,value)
    def max_pool2d(self,x,k,stride): return Node('pool',x,k,stride)
    def spp(self,x):
        return self.cat([self.max_pool2d(self.pad(x,(k//2,)*4,-1e30),k,1) for k in (13,9,5)]+[x],0)
    def upsample(self,*args): return Node('upsample',*args)
    def decode(self,*args): return Node('decode',*args)

class TopologyTests(unittest.TestCase):
    def test_source_matches_independent_cfg(self):
        graph=json.loads((ROOT/'generated_yolov4.json').read_text())
        inputs={name:Node('input',name) for name in graph['inputs']}
        remora=forward(Trace(),inputs)
        direct=YOLOv4(ROOT/'yolov4.cfg')(Trace(),inputs)
        self.assertEqual(remora.key,direct.key)

if __name__=='__main__': unittest.main()
