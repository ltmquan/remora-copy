"""Build Futhark CUDA libraries for already interpreter-validated YOLO fixtures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--remora',required=True);p.add_argument('--futhark',required=True)
p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.fixtures/'results.json').read_text());assert manifest['passed']
for name in ['conv_same','spp_cascade','composed_pipeline']:
 d=a.out/name;d.mkdir(exist_ok=True)
 case=next(c for c in manifest['cases'] if c['name']==name)
 for suffix in ['.remora','_auto.py']:
  shutil.copyfile(a.fixtures/(name+suffix),d/('source.remora' if suffix=='.remora' else 'generated.py'))
 assert hashlib.sha256((d/'generated.py').read_bytes()).hexdigest()==case['lowerings']['auto']['generated_sha256']
 stages=[]
 for label,cmd,out in [('remora',[a.remora,'futhark','-f',str((d/'source.remora').resolve())],'model.fut_soacs'),('gpu-memory',[a.futhark,'dev','--gpu-mem',str((d/'model.fut_soacs').resolve())],'model.fut_gpu_mem'),('cuda',[a.futhark,'dev','--backend=cuda','--library',str((d/'model.fut_gpu_mem').resolve())],'cuda.stdout')]:
  start=time.monotonic()
  with (d/out).open('w') as f,(d/(label+'.stderr')).open('w') as err:
   result=subprocess.run(cmd,stdout=f,stderr=err,timeout=300)
  stages.append({'stage':label,'seconds':time.monotonic()-start,'returncode':result.returncode})
  print(name,label,result.returncode,flush=True)
  if result.returncode:break
 abi=json.loads((d/'model.json').read_text())['entry_points']['main'] if not stages[-1]['returncode'] else None
 data=np.load(a.fixtures/(name+'_17_inputs.npz'))
 names=[v['name'] for v in abi['inputs']] if abi else data.files
 assert names==data.files,(names,data.files)
 expected=np.load(a.fixtures/(name+'_17_reference.npy'))
 if abi:
  for info,key in zip(abi['inputs'],names):assert info['type']=='[]'*data[key].ndim+'f32'
  assert abi['output']['type']=='[]'*expected.ndim+'f32'
 shutil.copyfile(a.fixtures/(name+'_17_inputs.npz'),d/'inputs.npz')
 np.save(d/'reference.npy',expected);expected.tofile(d/'reference.bin')
 with (d/'inputs.bin').open('wb') as f:
  for key in names:data[key].tofile(f)
 meta={'case':name,'source_sha256':hashlib.sha256((d/'source.remora').read_bytes()).hexdigest(),'generated_sha256':case['lowerings']['auto']['generated_sha256'],'lowering':case['lowerings']['auto']['selected'],'args':names,'shapes':{k:list(data[k].shape) for k in names},'output_shape':list(expected.shape),'stages':stages}
 if not abi:meta['futhark_error']=(d/(stages[-1]['stage']+'.stderr')).read_text()
 (d/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 if not abi:continue
 rank=expected.ndim;n=expected.size
 lines=['#include "model.h"','#include <stdio.h>','#include <stdlib.h>','#include <math.h>','#include <time.h>',
 'double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}',
 'void check(struct futhark_context *c,int code){if(code){fprintf(stderr,"%s\\n",futhark_context_get_error(c));exit(2);}}',
 'int main(void){FILE *in=fopen("inputs.bin","rb");if(!in)return 1;struct futhark_context_config *cfg=futhark_context_config_new();struct futhark_context *ctx=futhark_context_new(cfg);check(ctx,futhark_context_sync(ctx));']
 for i,key in enumerate(names):
  sh=data[key].shape;size=data[key].size;r=len(sh)
  lines += [f'float h{i}[{size}];if(fread(h{i},4,{size},in)!={size})return 3;',f'struct futhark_f32_{r}d *a{i}=futhark_new_f32_{r}d(ctx,h{i},'+','.join(map(str,sh))+');']
 lines+=['fclose(in);',f'float expected[{n}],host[{n}];in=fopen("reference.bin","rb");if(!in||fread(expected,4,{n},in)!={n})return 4;fclose(in);',f'struct futhark_f32_{rank}d *result=NULL;']
 call='check(ctx,futhark_entry_main(ctx,&result,'+','.join(f'a{i}' for i in range(len(names)))+'));check(ctx,futhark_context_sync(ctx));'
 free=f'check(ctx,futhark_free_f32_{rank}d(ctx,result));'
 lines += [call,f'const int64_t *shape=futhark_shape_f32_{rank}d(ctx,result);']
 for i,v in enumerate(expected.shape):lines += [f'if(shape[{i}]!={v})return 5;']
 lines += [f'check(ctx,futhark_values_f32_{rank}d(ctx,result,host));check(ctx,futhark_context_sync(ctx));',f'double maxerr=0;for(int i=0;i<{n};i++){{double diff=fabs((double)host[i]-expected[i]);if(!isfinite(host[i])||diff>1e-5+1e-4*fabs(expected[i]))return 6;if(diff>maxerr)maxerr=diff;}}', 'printf("correctness_max_abs %.12g\\n",maxerr);',free,
 'for(int i=-3;i<10;i++){check(ctx,futhark_context_sync(ctx));double start=now();',call,'double elapsed=now()-start;if(i>=0)printf("resident_ms %.9f\\n",elapsed*1000);',free,'}']
 for i,key in enumerate(names):lines += [f'check(ctx,futhark_free_f32_{data[key].ndim}d(ctx,a{i}));']
 lines += ['futhark_context_free(ctx);futhark_context_config_free(cfg);return 0;}']
 (d/'harness.c').write_text('\n'.join(lines)+'\n')
