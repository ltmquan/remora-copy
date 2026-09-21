"""Record compile and interpreter agreement separately, including failures."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import torch
from run import ROOT, load
from remora_torch import compile_source
sys.path.insert(0,str(ROOT/'tests'))
from test_backend import values
out=ROOT/'evaluation/example-evidence';out.mkdir(exist_ok=True)
remora=os.environ['REMORA'];results=[]
paths=sorted((ROOT.parent/'examples/extras').glob('*.remora'))+[ROOT.parent/'examples/nbody.remora']
for path in paths:
    row={'source':str(path.relative_to(ROOT.parent)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        expression=path.name=='nbody.remora'
        source_args=['-e',path.read_text()] if expression else ['-f',str(path)]
        checked=subprocess.run([remora,'dev','--typecheck','--json']+source_args,capture_output=True,text=True,timeout=20)
        if checked.returncode:raise RuntimeError('frontend: '+checked.stderr+checked.stdout)
        code=compile_source(**({'expression':path.read_text()} if expression else {'source':path}),remora=remora,timeout=20)
        generated=out/(path.stem+'.py');generated.write_text(code);mod=load(generated)
        row.update(compiled=True,lowering=mod.LOWERING)
        import inspect
        if any(p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD) for p in inspect.signature(mod.main).parameters.values()):
            row['execution']='not run: entry requires inputs'
        else:
            raw=subprocess.check_output([remora,'interpret']+source_args,text=True,stderr=subprocess.PIPE,timeout=20)
            start=time.perf_counter();actual=mod.main()
            row["compiled_cpu_ms_single_run"]=(time.perf_counter()-start)*1000
            expected=torch.as_tensor(values(raw)[0][0],dtype=actual.dtype)
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            row.update(interpreter_agreement=True,shape=list(actual.shape))
    except Exception as e:row['error']=str(e)[:2500]
    results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(row,flush=True)
