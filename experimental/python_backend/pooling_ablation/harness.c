#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void check(struct futhark_context *c,int code){if(code){fprintf(stderr,"%s\n",futhark_context_get_error(c));exit(2);}}
int main(int argc,char **argv){
 if(argc!=2)return 1;
 struct futhark_context_config *cfg=futhark_context_config_new();
 struct futhark_context *ctx=futhark_context_new(cfg);if(!ctx)return 2;
 float *a=malloc(512*19*19*sizeof(float)),*out=malloc(2048*19*19*sizeof(float));
 for(int i=0;i<512*19*19;i++)a[i]=((i*17)%251-125)/125.f;
 struct futhark_f32_3d *x=futhark_new_f32_3d(ctx,a,512,19,19);check(ctx,futhark_context_sync(ctx));
 for(int round=1;round<=3;round++)for(int run=-10;run<100;run++){
  struct futhark_f32_3d *y=NULL;double t=now();
  check(ctx,futhark_entry_main(ctx,&y,x));check(ctx,futhark_context_sync(ctx));
  double elapsed=now()-t;
  if(run>=0)printf("run %d %d %.9f\n",round,run,elapsed);
  if(round==3 && run==99){check(ctx,futhark_values_f32_3d(ctx,y,out));check(ctx,futhark_context_sync(ctx));FILE *f=fopen(argv[1],"wb");if(!f)return 3;fwrite(out,sizeof(float),2048*19*19,f);fclose(f);}
  check(ctx,futhark_free_f32_3d(ctx,y));
 }
 check(ctx,futhark_free_f32_3d(ctx,x));futhark_context_free(ctx);futhark_context_config_free(cfg);free(a);free(out);return 0;
}
