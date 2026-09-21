"""Correctness-gated resident GPU timings for exact small YOLO source cases."""
import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
torch.set_grad_enabled(False);torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'tf32':False,'warmups':3,'runs':10,'timing':'Host wall time with CUDA synchronization; inputs resident, output stays resident; setup and correctness checks excluded; PyTorch eager execution','cases':[]}
def save():(a.root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def compare(actual,expected):
 actual=actual.detach().cpu().numpy()
 assert actual.shape==expected.shape,(actual.shape,expected.shape)
 diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64));limit=1e-5+1e-4*np.abs(expected.astype(np.float64))
 assert np.isfinite(actual).all() and (diff<=limit).all(),f'max difference {diff.max()}'
 return float(diff.max())
def measure(fn):
 for _ in range(3):fn();torch.cuda.synchronize()
 samples=[]
 for _ in range(10):
  torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
 return {'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}
def spp(x):return torch.cat([F.max_pool2d(F.pad(x,(k//2,)*4,value=-1e30),k,stride=1) for k in [13,9,5]]+[x],0)
def fused_conv(d,x,p,activation):
 inv=d[p+'scale']/torch.sqrt(d[p+'var']+1e-5)
 w=d[p+'w']*inv[:,None,None,None]
 bias=d[p+'bias']-(d[p+'scale']*d[p+'mean'])/torch.sqrt(d[p+'var']+1e-5)
 z=F.conv2d(x[None],w,padding=1)[0]+bias[:,None,None]
 return z*torch.tanh(torch.log(1+torch.exp(z))) if activation=='mish' else torch.maximum(z,.1*z)
def composed(d):
 x=fused_conv(d,d['x'],'p','mish');x=spp(x).repeat_interleave(2,1).repeat_interleave(2,2)
 x=fused_conv(d,x,'f','leaky');z=(F.conv2d(x[None],d['hw'])[0]+d['hbias'][:,None,None]).reshape(3,7,10,10)
 s=10;row=torch.arange(s,device=z.device,dtype=torch.float32).reshape(1,s,1);col=row.transpose(1,2)
 sig=lambda x:1/(1+2.718281828459045**(-x))
 dec=torch.cat([((col+sig(z[:,0])*1.05-.025)/s)[:,None],((row+sig(z[:,1])*1.05-.025)/s)[:,None],(2.718281828459045**z[:,2]*d['aw'][:,None,None]/608)[:,None],(2.718281828459045**z[:,3]*d['ah'][:,None,None]/608)[:,None],sig(z[:,4:])],1)
 return dec.flatten(2).transpose(1,2).reshape(300,7)
for name in ['conv_same','spp_cascade','composed_pipeline']:
 d=a.root/name;meta=json.loads((d/'metadata.json').read_text());expected=np.load(d/'reference.npy')
 with np.load(d/'inputs.npz') as data:inputs={k:torch.from_numpy(data[k]).cuda() for k in data.files}
 assert hashlib.sha256((d/'generated.py').read_bytes()).hexdigest()==meta['generated_sha256']
 spec=importlib.util.spec_from_file_location('generated_'+name,d/'generated.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 generated=lambda:module.ENTRIES['main'](*(inputs[k] for k in meta['args']))
 direct=({'conv_same':lambda:F.conv2d(inputs['x'][None],inputs['w'],padding=1)[0].flatten(1),'spp_cascade':lambda:spp(inputs['x']),'composed_pipeline':lambda:composed(inputs)})[name]
 record={'case':name,'shapes':meta['shapes'],'output_shape':meta['output_shape'],'lowering':module.LOWERING,'source_sha256':meta['source_sha256'],'correctness_max_abs':{}}
 report['cases'].append(record);save()
 # All three pipelines must agree with the saved interpreter before timing results are accepted.
 record['correctness_max_abs']['remora_pytorch']=compare(generated(),expected)
 record['correctness_max_abs']['direct_pytorch']=compare(direct(),expected)
 if 'futhark_error' in meta:
  record['remora_futhark']={'error':meta['futhark_error']}
 else:
  proc=subprocess.run([str((d/'benchmark').resolve())],cwd=d,capture_output=True,text=True,check=False)
  (d/'futhark.log').write_text(proc.stdout+proc.stderr)
  if proc.returncode:
   record['remora_futhark']={'error':proc.stderr or proc.stdout,'exit_code':proc.returncode}
  else:
   f_samples=[float(line.split()[1]) for line in proc.stdout.splitlines() if line.startswith('resident_ms ')]
   assert len(f_samples)==10
   record['correctness_max_abs']['remora_futhark']=float(next(line.split()[1] for line in proc.stdout.splitlines() if line.startswith('correctness_max_abs ')))
   record['remora_futhark']={'median_ms':statistics.median(f_samples),'min_ms':min(f_samples),'max_ms':max(f_samples),'samples_ms':f_samples}
 for label,fn in [('remora_pytorch',generated),('direct_pytorch',direct)]:
  record[label]=measure(fn);save();print(name,label,record[label]['median_ms'],flush=True)
 profile=cProfile.Profile();profile.enable();generated();torch.cuda.synchronize();profile.disable()
 profile.dump_stats(str(d/'generated.prof'))
 stats=pstats.Stats(profile)
 record['profile_top']=[{'function':str(key),'calls':v[1],'cumulative_seconds':v[3]} for key,v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:12]]
 save()
report['completed']=True;report['all_three_available']=all('error' not in r['remora_futhark'] for r in report['cases']);save()
