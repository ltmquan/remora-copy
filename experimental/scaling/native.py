"""Library adapters for the existing Futhark and Darknet implementations."""
import ctypes as C
import json, os, hashlib
import numpy as np
from model import ROOT, YOLOv4, fixture, image

class Futhark:
    backend='futhark'
    def __init__(self, implementation, size, weights):
        self.size=size
        directory=ROOT/'artifacts'/str(size)/implementation.removeprefix('remora-futhark-')/'futhark'
        self.lib=C.CDLL(str(directory/'model.so'))
        def function(name,restype,args):
            f=getattr(self.lib,name); f.restype=restype; f.argtypes=args; return f
        self.bind=function; ptr=C.c_void_p
        self.cfg=function('futhark_context_config_new',ptr,[])()
        identity=hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES','default').encode()).hexdigest()[:12]
        cache=directory/f'cuda-{identity}.cache'
        self.cache_state={'path':str(cache),'existed_before_context':cache.exists()}
        # Futhark validates source/build options before accepting cached PTX.
        function('futhark_context_config_set_cache_file',None,[ptr,C.c_char_p])(self.cfg,str(cache).encode())
        self.ctx=function('futhark_context_new',ptr,[ptr])(self.cfg)
        if not self.ctx: raise RuntimeError('Futhark context creation failed')
        self.sync=function('futhark_context_sync',C.c_int,[ptr])
        self.error=function('futhark_context_get_error',C.c_char_p,[ptr])
        self.new={}; self.free={}
        for rank in (1,2,3,4):
            self.new[rank]=function(f'futhark_new_f32_{rank}d',ptr,[ptr,ptr]+[C.c_int64]*rank)
            self.free[rank]=function(f'futhark_free_f32_{rank}d',C.c_int,[ptr,ptr])
        abi=json.loads((directory/'model.json').read_text())['entry_points']['main']
        self.entry=function(abi['cfun'],C.c_int,[ptr,C.POINTER(ptr)]+[ptr]*len(abi['inputs']))
        self.values=function('futhark_values_f32_2d',C.c_int,[ptr,ptr,ptr])
        arrays=fixture(YOLOv4(ROOT/'artifacts'/str(size)/'yolov4.cfg'),weights); arrays.pop('image')
        self.names=[item['name'] for item in abi['inputs']]
        self.params={key:self.upload(value) for key,value in arrays.items()}
        self.host=[image(size,i) for i in range(2)]; self.resident=[self.upload(x) for x in self.host]
        self.output=None; self.check(self.sync(self.ctx))
        self.ops=self
    def check(self,code):
        if code: raise RuntimeError((self.error(self.ctx) or b'Futhark runtime error').decode())
    def upload(self,a):
        value=self.new[a.ndim](self.ctx,a.ctypes.data,*a.shape)
        if not value: raise RuntimeError('Futhark allocation failed')
        return value
    def infer(self,index,mode):
        # Keep at most one result alive; input transfer is inside transfer mode.
        if self.output: self.check(self.free[2](self.ctx,self.output)); self.output=None
        im=self.resident[index%2] if mode=='resident' else self.upload(self.host[index%2])
        args=[im if name=='image' else self.params[name] for name in self.names]
        result=C.c_void_p(); self.check(self.entry(self.ctx,C.byref(result),*args)); self.output=result
        if mode=='transfer':
            output=self.numpy(result); self.check(self.free[3](self.ctx,im))
        else: output=result
        self.check(self.sync(self.ctx)); return output
    def numpy(self,y):
        out=np.empty((sum(3*(self.size//d)**2 for d in (8,16,32)),85),np.float32)
        self.check(self.values(self.ctx,y,out.ctypes.data)); self.check(self.sync(self.ctx)); return out
    def memory(self):
        report=self.bind('futhark_context_report',C.c_void_p,[C.c_void_p])(self.ctx)
        if not report: raise RuntimeError('Missing Futhark memory report')
        try: data=json.loads(C.string_at(report))
        finally:
            free=C.CDLL(None).free; free.argtypes=[C.c_void_p]; free(report)
        return {'live_peak_bytes':data.get('memory',{}).get("space 'device'"),'reserved_peak_bytes':None,'scope':'Futhark context lifetime, includes setup'}

class Darknet:
    backend='darknet'
    def __init__(self,implementation,size,weights):
        self.lib=C.CDLL(str(ROOT/'artifacts/native/darknet_adapter.so'))
        self.lib.yolo_open.argtypes=[C.c_char_p,C.c_char_p]; self.lib.yolo_open.restype=C.c_void_p
        self.lib.yolo_infer.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]; self.lib.yolo_infer.restype=C.c_int
        self.handle=self.lib.yolo_open(str(ROOT/'artifacts'/str(size)/'yolov4.cfg').encode(),str(weights).encode())
        if not self.handle: raise RuntimeError('Darknet initialization failed')
        self.host=[image(size,i) for i in range(2)]
        self.output=np.empty((sum(3*(size//d)**2 for d in (8,16,32)),85),np.float32)
    def infer(self,index,mode):
        if mode!='transfer': raise ValueError('Native Darknet network_predict includes transfers')
        if self.lib.yolo_infer(self.handle,self.host[index%2].ctypes.data,self.output.ctypes.data,self.output.size): raise RuntimeError('Darknet inference failed')
        return self.output.copy()
    def memory(self): return {'live_peak_bytes':None,'reserved_peak_bytes':None}
