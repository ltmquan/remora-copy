"""Bounded first experiment. Never rent GPUs or shut down the host."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
p.add_argument('--budget-minutes',type=float,default=30);p.add_argument('--trial-seconds',type=int,default=240)
p.add_argument('--output',type=Path,default=Path('results/single-image'));a=p.parse_args()
assert a.budget_minutes>0 and a.trial_seconds>0
root=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'topology.txt').open('w') as f:
 subprocess.run(['nvidia-smi','topo','-m'],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['nvidia-smi'],stdout=f,stderr=subprocess.STDOUT,check=True)
start=time.monotonic();records=[]
trials=[('smoke',608,'spatial','0,1','1,1',True)]
for size in (608,1216):
 trials += [(f'{size}-compiled',size,'single-compiled','0','1',False),(f'{size}-one',size,'spatial','0','1',False),(f'{size}-equal',size,'spatial','0,1','1,1',False),(f'{size}-unequal',size,'spatial','0,1','1,2',False)]
for name,size,mode,devices,split,eager in trials:
 remaining=a.budget_minutes*60-(time.monotonic()-start)
 if remaining<30:break
 cmd=[sys.executable,str(root/'run.py'),'--weights',str(a.weights.resolve()),'--size',str(size),'--mode',mode,'--devices',devices,'--split',split,'--min-side',str(size//2),'--output',str((a.output/f'{name}.json').resolve())]
 if eager:cmd+=['--eager-shards','--runs','2']
 timeout=min(a.trial_seconds,remaining)
 print(name,flush=True);t=time.monotonic()
 try:
  with (a.output/f'{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  status='passed' if r.returncode==0 else 'failed'
 except subprocess.TimeoutExpired:status='timeout'
 records.append({'name':name,'status':status,'seconds':time.monotonic()-t,'command':cmd})
 (a.output/'sweep.json').write_text(json.dumps(records,indent=2)+'\n')
 # Correctness or setup failure is a stop condition, not a speedup result.
 if status!='passed':break
print(json.dumps(records,indent=2))
