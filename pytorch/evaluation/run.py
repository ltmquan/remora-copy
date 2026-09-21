"""Small, explicit Remora kernel ports; never claims full DARPA application coverage."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'validation'))
from remora_torch import compile_source
from yolo_differential import literal, parse_result

def cases(n,k,m):
    dot=f'(def (fun (pair-product (left [Float {k}]) (right [Float {k}])) (@reduce/zero (Float) ({k} []) f.+ 0.0 (f.* left right))))'
    return [
      ('squared_l2',f'''(entry (main (query [Float {k}]) (dataset [Float {n} {k}]))
        ((fn ((row [Float {k}])) (let ((val delta (f.- query row)))
          (@reduce/zero (Float) ({k} []) f.+ 0.0 (f.* delta delta)))) dataset))''',[(k,),(n,k)],lambda q,x:((x-q)**2).sum(-1)),
      ('linear_projection',dot+f'''(entry (main (rows [Float {n} {k}]) (weights [Float {m} {k}]))
        ((fn ((row [Float {k}])) (pair-product row weights)) rows))''',[(n,k),(m,k)],lambda x,w:x@w.T),
      ('rms_norm',f'''(entry (main (rows [Float {n} {k}]) (weight [Float {k}]))
        ((fn ((row [Float {k}])) (f.* weight (f./ row (sqrt (f.+ 0.00001
          (f./ (@reduce/zero (Float) ({k} []) f.+ 0.0 (f.* row row)) {k}.0)))))) rows))''',[(n,k),(k,)],lambda x,w:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*w),
      ('reconstruction_error',f'''(entry (main (features [Float {n} {k}]) (reconstructed [Float {n} {k}]))
        ((fn ((x [Float {k}]) (y [Float {k}]))
          (let ((val delta (f.- x y))) (sqrt (@reduce/zero (Float) ({k} []) f.+ 0.0 (f.* delta delta))))) features reconstructed))''',[(n,k),(n,k)],lambda x,y:(x-y).square().sum(-1).sqrt()),
    ]

def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def measure(fn,args,device):
    for _ in range(3):fn(*args)
    if device=='cuda':torch.cuda.synchronize()
    samples=[]
    for _ in range(10):
        start=time.perf_counter();fn(*args)
        if device=='cuda':torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return sorted(samples)[len(samples)//2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--remora',default=os.environ.get('REMORA'));p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,default=ROOT/'evaluation/evidence');p.add_argument('--prepared',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    report={'torch':torch.__version__,'device':a.device,'device_name':torch.cuda.get_device_name() if a.device=='cuda' else 'CPU','scope':'kernel ports only, not full DARPA applications','cases':[]}
    for size,dims in [('small',(3,4,2)),('medium',(128,128,96))]:
      for name,source,shapes,reference in cases(*dims):
        stem=name+'-'+size;path=a.out/(stem+'.remora')
        if not a.prepared:path.write_text(source+'\n')
        mods={}
        for lowering in ('tensor','static'):
          generated=a.out/(stem+'-'+lowering+'.py')
          if not a.prepared:generated.write_text(compile_source(source=path,remora=a.remora,lowering=lowering))
          mods[lowering]=load(generated)
        item={'name':name,'size':size,'shapes':shapes,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ops':mods['tensor'].TENSOR_OPS,'fixtures':[]}
        for seed in range(3):
          torch.manual_seed(seed);args=[torch.randn(sh,device=a.device) for sh in shapes]
          if seed==0:args=[torch.zeros_like(x) for x in args]
          expected=reference(*args)
          for m in mods.values():torch.testing.assert_close(m.main(*args),expected,atol=2e-4,rtol=2e-4)
          check={'seed':seed,'direct_and_static':True}
          if size=='small' and not a.prepared:
            raw=subprocess.check_output([a.remora,'interpret','-f',str(path),'--']+[literal(x.cpu().numpy()) for x in args],text=True,timeout=30)
            interpreted=torch.tensor(parse_result(raw),device=a.device)
            torch.testing.assert_close(mods['tensor'].main(*args),interpreted,atol=2e-4,rtol=2e-4);check['interpreter']=True
          item['fixtures'].append(check)
        # Nonzero random fixture for timing; no compilation or transfers in timed region.
        with torch.no_grad():item['median_ms']={mode:measure(fn,args,a.device) for mode,fn in [('tensor',mods['tensor'].main),('static',mods['static'].main),('direct',reference)]}
        report['cases'].append(item)
        (a.out/(a.device+'-results.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(stem,item['ops'],item['median_ms'],flush=True)
if __name__=='__main__':main()
