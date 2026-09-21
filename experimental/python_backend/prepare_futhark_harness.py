"""Generate an ABI-checked harness and matching trained input stream."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'fresh_futhark');a=p.parse_args();out=a.out.resolve()
abi=json.loads((out/'model.json').read_text())
args=abi['entry_points']['main']['inputs']
graph=json.loads((ROOT/'generated_yolov4.json').read_text())
assert {p['name'] for p in args}==set(graph['inputs'])
shapes=[]
for a in args:
 ds=graph['inputs'][a['name']]
 assert a['type']=='[]'*len(ds)+'f32'
 shapes.append((a['name'],ds))
assert shapes[0][0]=='image'
(out/'inputs.json').write_text(json.dumps(shapes,indent=2)+'\n')
lines=['#include "model.h"','#include <stdio.h>','#include <stdlib.h>','#include <math.h>','#include <time.h>',
'double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}',
'void check(struct futhark_context *c,int code){if(code){char *e=futhark_context_get_error(c);fprintf(stderr,"%s\\n",e?e:"Futhark error");exit(2);}}',
'int main(int argc,char **argv){if(argc!=3)return 1;FILE *in=fopen(argv[1],"rb");if(!in)return 3;',
'double setup=now();struct futhark_context_config *cfg=futhark_context_config_new();struct futhark_context *ctx=futhark_context_new(cfg);if(!ctx)return 2;check(ctx,futhark_context_sync(ctx));']
import math
for i,(name,ds) in enumerate(shapes):
 n=math.prod(ds);rank=len(ds)
 lines += [f'float *h{i}=malloc({n}ULL*sizeof(float));if(!h{i})return 3;',f'if(fread(h{i},sizeof(float),{n}ULL,in)!={n}ULL)return 4;',f'struct futhark_f32_{rank}d *a{i}=futhark_new_f32_{rank}d(ctx,h{i},'+','.join(map(str,ds))+f');if(!a{i})check(ctx,1);']
 if i:lines.append(f'free(h{i});')
lines+=['if(fgetc(in)!=EOF)return 5;fclose(in);check(ctx,futhark_context_sync(ctx));printf("setup_seconds %.9f\\n",now()-setup);fflush(stdout);',
'float *host=malloc(22743ULL*85*sizeof(float));if(!host)return 3;']
for phase in ['resident','transfer']:
 lines += [f'printf("MEMORY_PHASE {phase}\\n");fflush(stdout);']
 lines+=['for(int run=-3;run<10;run++){struct futhark_f32_2d *result=NULL;check(ctx,futhark_context_sync(ctx));double start=now();']
 first='a0'
 if phase=='transfer':
  first='image';lines+=['struct futhark_f32_3d *image=futhark_new_f32_3d(ctx,h0,3,608,608);if(!image)check(ctx,1);']
 lines+=['check(ctx,futhark_entry_main(ctx,&result,'+','.join([first]+[f'a{i}' for i in range(1,len(shapes))])+'));', 'check(ctx,futhark_context_sync(ctx));']
 if phase=='transfer':lines+=['check(ctx,futhark_values_f32_2d(ctx,result,host));check(ctx,futhark_context_sync(ctx));']
 lines += [f'printf("{phase}_%s %d %.9f\\n",run<0?"warmup_seconds":"run_seconds",run,now()-start);fflush(stdout);']
 if phase=='resident':
  lines+=['if(run==9){const int64_t *s=futhark_shape_f32_2d(ctx,result);if(s[0]!=22743||s[1]!=85)return 6;check(ctx,futhark_values_f32_2d(ctx,result,host));check(ctx,futhark_context_sync(ctx));FILE *f=fopen(argv[2],"wb");if(!f)return 7;if(fwrite(host,sizeof(float),22743ULL*85,f)!=22743ULL*85)return 7;fclose(f);}']
 if phase=='transfer':lines+=['check(ctx,futhark_free_f32_3d(ctx,image));']
 lines+=['check(ctx,futhark_free_f32_2d(ctx,result));}']
lines+=['char *report=futhark_context_report(ctx);if(report){printf("FUTHARK_MEMORY_REPORT %s\\n",report);free(report);}', 'for(size_t i=0;i<22743ULL*85;i++)if(!isfinite(host[i]))return 8;free(host);free(h0);']
for i,(_,ds) in enumerate(shapes):lines+=[f'check(ctx,futhark_free_f32_{len(ds)}d(ctx,a{i}));']
lines+=['futhark_context_free(ctx);futhark_context_config_free(cfg);return 0;}']
(out/'harness.c').write_text('\n'.join(lines)+'\n')
print(f'Generated {len(shapes)}-input harness')
