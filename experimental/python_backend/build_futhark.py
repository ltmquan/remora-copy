"""Build the current original YOLO source through the existing Futhark toolchain."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
root=Path(__file__).resolve().parent
workspace=root.parents[2]
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=workspace/'remora/examples/darknet/yolov4.remora');p.add_argument('--out',type=Path,default=root/'fresh_futhark');a=p.parse_args()
out=a.out.resolve();out.mkdir(exist_ok=True)
source=a.source.resolve()
remora=workspace/'scheduling-investigation/bin/remora'
futhark=workspace/'scheduling-investigation/bin/futhark'
records={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'stages':[]}
for name,cmd,target in [
 ('remora',[remora,'futhark','-f',source],out/'model.fut_soacs'),
 ('gpu-memory',[futhark,'dev','--gpu-mem',out/'model.fut_soacs'],out/'model.fut_gpu_mem'),
 ('cuda-library',[futhark,'dev','--backend=cuda','--library',out/'model.fut_gpu_mem'],out/'cuda.stdout')]:
 print(name,flush=True);t=time.monotonic()
 with target.open('wb') as stdout,(out/(name+'.stderr')).open('wb') as stderr:
  r=subprocess.run(list(map(str,cmd)),stdout=stdout,stderr=stderr,cwd=workspace/'remora-copy',timeout=900)
 records['stages'].append({'stage':name,'command':list(map(str,cmd)),'exit_code':r.returncode,'seconds':time.monotonic()-t})
 (out/'build.json').write_text(json.dumps(records,indent=2)+'\n')
 if r.returncode:raise SystemExit(r.returncode)
print('fresh CUDA library generated',flush=True)
