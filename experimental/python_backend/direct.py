"""Independent YOLOv4 implementation constructed from Darknet cfg, no Remora IR.

Both PyTorch and tinygrad use their native conv2d through Ops. Layer connectivity,
parameter shapes and YOLO anchors are independently derived from cfg.
"""
from pathlib import Path

def read_cfg(path):
    sections=[]
    for raw in Path(path).read_text().splitlines():
        line=raw.split('#',1)[0].strip()
        if not line: continue
        if line.startswith('['): sections.append({'type':line[1:-1]})
        else:
            k,v=line.split('=',1); sections[-1][k.strip()]=v.strip()
    return sections

class YOLOv4:
    def __init__(self,cfg):
        sections=read_cfg(cfg); self.net=sections[0]; self.layers=sections[1:]
        self.shapes=[]; self.parameters={}
        c=int(self.net['channels']); n=int(self.net['width'])
        for i,l in enumerate(self.layers):
            t=l['type']
            if t=='convolutional':
                oc,k,s=int(l['filters']),int(l['size']),int(l.get('stride',1))
                pad=k//2 if int(l.get('pad',0)) else int(l.get('padding',0))
                out=(n+2*pad-k)//s+1
                self.parameters[i]=(oc,c,k,'batch_normalize' in l)
                l['_shape']=[c,n,oc,out,k]; l['_pad']=pad; c,n=oc,out
            elif t=='route':
                ids=[int(v) for v in l['layers'].split(',')]
                l['_ids']=[i+j if j<0 else j for j in ids]
                shapes=[self.shapes[j] for j in l['_ids']]
                if len({v[1] for v in shapes})!=1: raise ValueError('Route dimensions differ')
                c,n=sum(v[0] for v in shapes),shapes[0][1]
            elif t=='shortcut':
                l['_from']=i+int(l['from'])
                if self.shapes[l['_from']]!=(c,n): raise ValueError('Shortcut dimensions differ')
            elif t=='upsample': n*=int(l['stride'])
            elif t=='maxpool':
                k,s=int(l['size']),int(l['stride']); n=(n+k-1-k)//s+1
            elif t!='yolo': raise ValueError(f'Unsupported cfg layer: {t}')
            self.shapes.append((c,n))

    def __call__(self,ops,inputs):
        x=inputs['image']; outputs=[]; heads=[]
        for i,l in enumerate(self.layers):
            t=l['type']
            if t=='convolutional':
                p={'w':inputs[f'w-{i:03}'],'bias':inputs[f'bias-{i:03}']}
                if 'batch_normalize' in l:
                    p.update({k:inputs[f'{v}-{i:03}'] for k,v in [('mean','mean'),('variance','var'),('scale','scale')]})
                x=ops.conv(x,p,int(l.get('stride',1)),l['_pad'],l['activation'],l['_shape'])
            elif t=='route': x=ops.cat([outputs[j] for j in l['_ids']],0)
            elif t=='shortcut': x=x+outputs[l['_from']]
            elif t=='upsample': x=ops.upsample(x,int(l['stride']))
            elif t=='maxpool':
                k=int(l['size']); pad=k//2
                if ops.backend=='torch': x=ops.F.max_pool2d(ops.F.pad(x,(pad,)*4,value=-1e30),k,stride=int(l['stride']))
                else: x=x.pad(((0,0),(pad,pad),(pad,pad)),value=-1e30).max_pool2d(k,stride=int(l['stride']))
            elif t=='yolo':
                anchors=[float(v) for v in l['anchors'].split(',')]; mask=[int(v) for v in l['mask'].split(',')]
                heads.append(ops.decode(x,[anchors[2*j] for j in mask],[anchors[2*j+1] for j in mask],float(l.get('scale_x_y',1)),int(self.net['width']),int(l['classes'])))
            outputs.append(x)
        return ops.cat(heads,0)
