"""Emit a reviewable matrix; launch sequentially only with --execute."""
import argparse, itertools, json, subprocess, sys
from pathlib import Path
from run import IMPLEMENTATIONS
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sizes',type=int,nargs='+',default=[608,960,1216,1536])
    p.add_argument('--counts',type=int,nargs='+',default=[1,2,4,8]); p.add_argument('--implementations',nargs='+',choices=IMPLEMENTATIONS,default=IMPLEMENTATIONS)
    p.add_argument('--policies',nargs='+',choices=['uniform','round-robin','weighted','dynamic'],default=['uniform','round-robin','weighted','dynamic'])
    p.add_argument('--modes',nargs='+',choices=['resident','transfer'],default=['resident','transfer'])
    p.add_argument('--gpu-order',nargs='+',help='Physical GPU IDs/UUIDs in topology-aware order; each count uses a prefix')
    p.add_argument('--atol',type=float,default=1e-5); p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--trials',type=int,default=5); p.add_argument('--setup-timeout',type=float,default=1800); p.add_argument('--trial-timeout',type=float,default=600)
    p.add_argument('--images',type=int,default=128); p.add_argument('--weak-scaling',action='store_true',help='Images per GPU instead of fixed global request count')
    p.add_argument('--reference-dir',type=Path,default=ROOT/'artifacts/references')
    p.add_argument('--weights',required=True); p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--sample-memory',action='store_true'); p.add_argument('--execute',action='store_true'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--continue-on-failure',action='store_true',help='Record failures and continue with remaining cells')
    p.add_argument('--skip-failed-model',action='store_true',help='After a failure, explicitly skip remaining cells for that implementation and size in this invocation')
    p.add_argument('--resume',action='store_true',help='Preserve existing results and resume unfinished cells')
    p.add_argument('--exclude-implementations',nargs='+',choices=IMPLEMENTATIONS,default=[])
    a=p.parse_args(); cells=[]
    if min(a.counts+a.sizes+[a.images,a.trials,a.setup_timeout,a.trial_timeout])<1 or any(s%32 for s in a.sizes): p.error('Invalid matrix dimensions')
    gpu_order=a.gpu_order or [str(i) for i in range(max(a.counts))]
    if len(gpu_order)<max(a.counts) or len(set(gpu_order))!=len(gpu_order): p.error('GPU order must contain enough distinct devices')
    if any(len(v)!=len(set(v)) for v in (a.sizes,a.counts,a.implementations,a.policies,a.modes)): p.error('Duplicate matrix dimensions')
    for size,impl,count,policy,mode in itertools.product(a.sizes,a.implementations,a.counts,a.policies,a.modes):
        if count==1 and policy!='uniform': continue
        if impl=='darknet-dense' and mode=='resident': continue
        name=f'yolov4-{impl}-{size}-{count}gpu-{policy}-{mode}'
        command=[sys.executable,str(ROOT/'run.py'),'--implementation',impl,'--size',str(size),'--gpus',*gpu_order[:count],'--policy',policy,'--mode',mode,'--images',str(a.images*count if a.weak_scaling else a.images),'--atol',str(a.atol),'--rtol',str(a.rtol),'--trials',str(a.trials),'--setup-timeout',str(a.setup_timeout),'--trial-timeout',str(a.trial_timeout),'--reference-dir',str(a.reference_dir),'--weights',a.weights,'--output',str(a.out/(name+'.json'))]
        if a.sample_memory: command.insert(2,'--sample-memory')
        cells.append({'name':name,'command':command,'model_key':f'{impl}-{size}'})
    if a.limit: cells=cells[:a.limit]
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'plan.json').write_text(json.dumps({'gpu_order':gpu_order,'scaling':'weak' if a.weak_scaling else 'strong','cells':cells,'adapter_pending':['DARPA additional targets']},indent=2)+'\n')
    print(f'{len(cells)} cells; plan: {a.out / "plan.json"}; execute={a.execute}')
    if a.execute:
        failed_models={}; outcomes=[]
        if a.resume and (a.out/'execution.json').exists():
            outcomes=json.loads((a.out/'execution.json').read_text())
        previous={x['name']:x for x in outcomes}
        for cell in cells:
            if previous.get(cell['name'],{}).get('status')=='failed':
                failed_models[cell['model_key']]=cell['name']
        for cell in cells:
            output=Path(cell['command'][-1])
            if cell['name'] in previous: continue
            if a.resume and output.exists():
                saved=json.loads(output.read_text())
                status='completed' if saved.get('status')=='passed' else 'failed'
                outcomes.append({'name':cell['name'],'status':status,'recovered_existing_result':True})
                if status=='failed': failed_models[cell['model_key']]=cell['name']
                (a.out/'execution.json').write_text(json.dumps(outcomes,indent=2)+'\n')
                continue
            if output.exists(): raise FileExistsError(f'Use a new results directory: {output}')
            impl=cell['command'][cell['command'].index('--implementation')+1]
            if impl in a.exclude_implementations:
                outcomes.append({'name':cell['name'],'status':'skipped','reason':'User stopped further runs of this implementation to limit rental cost; not measured'})
                (a.out/'execution.json').write_text(json.dumps(outcomes,indent=2)+'\n')
                continue
            if a.skip_failed_model and cell['model_key'] in failed_models:
                outcomes.append({'name':cell['name'],'status':'skipped','reason':'Earlier failure for this implementation and image size; not measured','failed_cell':failed_models[cell['model_key']]})
                (a.out/'execution.json').write_text(json.dumps(outcomes,indent=2)+'\n')
                continue
            print(f'Running {cell["name"]}',flush=True)
            with output.with_suffix('.log').open('w') as log:
                # Workers have their own setup/trial deadlines and cleanup.
                result=subprocess.run(cell['command'],stdout=log,stderr=subprocess.STDOUT)
            outcomes.append({'name':cell['name'],'status':'failed' if result.returncode else 'completed','returncode':result.returncode})
            (a.out/'execution.json').write_text(json.dumps(outcomes,indent=2)+'\n')
            print(f'{outcomes[-1]["status"]}: {cell["name"]}',flush=True)
            if result.returncode:
                failed_models[cell['model_key']]=cell['name']
                if not a.continue_on_failure: raise SystemExit(f'Stopped on failure: {cell["name"]}; inspect log/result before spending more GPU time')
if __name__=='__main__': main()
