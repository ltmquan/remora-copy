"""Compare exact YOLOv4 helper source to generated PyTorch on CPU.

No copied/reimplemented Remora helper bodies: top-level definitions are retained
verbatim from the selected source, with new small typed entry points appended.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remora_torch import compile_source


def definitions(source):
    """Extract balanced top-level forms, respecting comments/quoted strings."""
    forms, depth, start, comment, string, escape = [], 0, None, False, False, False
    for i, c in enumerate(source):
        if comment:
            if c == '\n': comment = False
            continue
        if string:
            if escape: escape = False
            elif c == '\\': escape = True
            elif c == '"': string = False
            continue
        if c == ';': comment = True; continue
        if c == '"': string = True; continue
        if c == '(':
            if depth == 0: start = i
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0: forms.append(source[start:i+1])
            if depth < 0: raise ValueError('unbalanced source')
    if depth or string: raise ValueError('unbalanced source')
    defs = [f for f in forms if re.match(r'\(def\s', f)]
    if len(defs) + sum(bool(re.match(r'\(entry\s', f)) for f in forms) != len(forms):
        raise ValueError('unexpected top-level form')
    return '\n\n'.join(defs) + '\n', defs


def reachable_definitions(forms, parsed, expression_ast):
    # Dependency discovery uses the real parser. Retain all global values so
    # initialization semantics are unchanged; only unused function definitions
    # are omitted from these small test programs.
    decls = [d['bind'] for d in parsed['progDecs'] if d['tag'] == 'Def']
    if len(decls) != len(forms): raise ValueError('source/AST definition mismatch')
    named = {d['var']: (d, f) for d, f in zip(decls, forms)}
    def references(x):
        if isinstance(x, str): return {x} & named.keys()
        if isinstance(x, list): return set().union(*(references(v) for v in x))
        if isinstance(x, dict): return set().union(*(references(v) for v in x.values()))
        return set()
    required = references(expression_ast) | {d['var'] for d in decls if d['tag'] == 'BindVal'}
    pending = list(required)
    while pending:
        name = pending.pop()
        new = references(named[name][0]) - required
        required |= new; pending.extend(new)
    return '\n\n'.join(f for d, f in zip(decls, forms) if d['var'] in required) + '\n', sorted(required)


def ty(shape):
    return '[Float' + (' ' + ' '.join(map(str, shape)) if shape else '') + ']'


def params(prefix, oc, ic, k, bn=True):
    return [(prefix+'w', (oc, ic, k, k))] + [(prefix+x, (oc,)) for x in (['mean','var','scale','bias'] if bn else ['bias'])]


def record(prefix, oc, ic, k, bn=True):
    fields = [('w', (oc,ic,k,k), 'w')]
    fields += [(name, (oc,), var) for name,var in ([('mean','mean'),('variance','var'),('scale','scale'),('bias','bias')] if bn else [('bias','bias')])]
    return '{' + ' '.join(f'({name} [{" ".join(map(str,sh))}] {prefix+var})' for name,sh,var in fields) + '}'


def cases():
    out = []
    def add(name, args, body, reference=None):
        out.append({'name':name,'args':args,'body':body,'reference':reference})
    for name,n,k,ic,oc,p,s,r in [('conv_same',4,3,2,3,1,1,0),('conv_stride2_remainder',4,3,2,3,1,2,1),('conv_1x1',3,1,2,3,0,1,0)]:
        q=(n+2*p-k)//s
        add(name,[('x',(ic,n,n)),('w',(oc,ic,k,k))],f'(@conv-2d/rem _ ({q} {k} {ic} {oc} {p} {s} {r}) x w)',
            lambda d,p=p,s=s: F.conv2d(d['x'][None],d['w'],padding=p,stride=s)[0].flatten(1))
    add('batch_norm',[('x',(2,3,3)),('mean',(2,)),('var',(2,)),('scale',(2,)),('bias',(2,))],
        '(@batch-norm-2d-apply _ (3 2) x mean var scale bias)',
        lambda d: (d['x']-d['mean'][:,None,None])/torch.sqrt(d['var'][:,None,None]+1e-5)*d['scale'][:,None,None]+d['bias'][:,None,None])
    add('activations',[('x',(3,7))], '[( @mish _ () x) (@leaky-relu _ () x) (@sigmoid _ () x)]',
        lambda d: torch.stack([d['x']*torch.tanh(torch.log(1+torch.exp(d['x']))),torch.maximum(d['x'],.1*d['x']),1/(1+2.718281828459045**(-d['x']))]))
    for activation in ['mish','leaky']:
        add('conv_bn_'+activation,[('x',(2,3,3))]+params('p',2,2,3),f'(@conv-bn-{activation} _ (2 3 2 2 1 1 0) x {record("p",2,2,3)})')
    add('residual',[('x',(2,3,3))]+params('r',1,2,1)+params('e',2,1,3),
        f'(@res-block _ (2 1 3) x {record("r",1,2,1)} {record("e",2,1,3)})')
    add('spp_cascade',[('x',(2,5,5))], '(@spp _ (5 2) x)',
        lambda d: torch.cat([F.max_pool2d(F.pad(d['x'],(k//2,)*4,value=-1e30),k,stride=1) for k in [13,9,5]]+[d['x']],0))
    add('upsample',[('x',(2,3,3))], '(@upsample-nearest _ (3 2 2) x)',
        lambda d: d['x'].repeat_interleave(2,1).repeat_interleave(2,2))
    add('route',[('x',(2,3,3)),('y',(1,3,3))], '(@route2 _ (2 1 3 3) x y)',lambda d: torch.cat([d['x'],d['y']],0))
    add('decode_anchors',[('x',(3,7,2,2)),('aw',(3,)),('ah',(3,))], '(@yolo-anchor _ (2 2) x aw ah 1.05 608.0)',lambda d: decode(d['x'],d['aw'],d['ah']))
    headargs=[('x',(2,2,2))]+params('f',2,2,3)+params('h',21,2,1,False)+[('aw',(3,)),('ah',(3,))]
    add('detection_head',headargs,f'(@yolo-head _ (2 2 2 2) x {record("f",2,2,3)} {record("h",21,2,1,False)} aw ah 1.05 608.0)')
    chainargs=[('x',(1,5,5))]+params('p',1,1,3)+params('f',1,4,3)+params('h',21,1,1,False)+[('aw',(3,)),('ah',(3,))]
    add('composed_pipeline',chainargs,f'''(let ((val c (@conv-bn-mish _ (4 3 1 1 1 1 0) x {record('p',1,1,3)}))
      (val pooled (@spp _ (5 1) c)) (val up (@upsample-nearest _ (5 4 2) pooled)))
      (@yolo-head _ (10 4 1 2) up {record('f',1,4,3)} {record('h',21,1,1,False)} aw ah 1.05 608.0))''')
    return out


def decode(z, aw, ah):
    s=z.shape[-1]; row=torch.arange(s,dtype=torch.float32).reshape(1,s,1); col=row.transpose(1,2)
    sigmoid=lambda x: 1/(1+2.718281828459045**(-x))
    return torch.cat([((col+sigmoid(z[:,0])*1.05-.025)/s)[:,None],
                      ((row+sigmoid(z[:,1])*1.05-.025)/s)[:,None],
                      (2.718281828459045**z[:,2]*aw[:,None,None]/608)[:,None],
                      (2.718281828459045**z[:,3]*ah[:,None,None]/608)[:,None],sigmoid(z[:,4:])],1)


def fixture(args, seed):
    rng=np.random.default_rng(seed); result={}
    for name,sh in args:
        if name.endswith('var'): x=rng.uniform(.1,2,size=sh)
        elif name in ('aw','ah'): x=np.array([12,19,40] if name=='aw' else [16,36,28])
        elif seed==0: x=np.zeros(sh)
        elif seed==1: x=np.linspace(-1.5,1.5,num=int(np.prod(sh))).reshape(sh)
        else: x=rng.uniform(-1,1,size=sh)
        result[name]=np.asarray(x,dtype=np.float32)
    return result


def literal(x):
    if x.ndim:
        return '['+' '.join(literal(v) for v in x)+']'
    value=format(float(x),'.9g')
    return value if '.' in value or 'e' in value.lower() else value+'.0'


def parse_result(raw):
    # Interpreter tensor results use nested square brackets and numeric atoms.
    tokens=re.findall(r'\[|\]|[^\s\[\]]+',raw); pos=0
    def value():
        nonlocal pos
        tok=tokens[pos];pos+=1
        if tok=='[':
            xs=[]
            while tokens[pos]!=']':xs.append(value())
            pos+=1;return xs
        return float(tok)
    result=np.asarray(value(),dtype=np.float32)
    if pos!=len(tokens): raise ValueError('trailing interpreter output')
    return result


def compare(actual,expected,atol,rtol):
    if actual.shape!=expected.shape:return {'passed':False,'shape_actual':list(actual.shape),'shape_expected':list(expected.shape)}
    diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64));limit=atol+rtol*np.abs(expected.astype(np.float64))
    finite=np.isfinite(actual)&np.isfinite(expected);bad=(diff>limit)|~finite
    return {'passed':not bool(bad.any()),'shape':list(actual.shape),'scalars':actual.size,'mismatches':int(bad.sum()),
            'nonfinite':int((~finite).sum()),'max_abs':float(diff.max(initial=0)),
            'max_error_over_tolerance':float((diff/limit).max(initial=0))}


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    raw=a.source.read_text();defs,forms=definitions(raw)
    parsed=json.loads(subprocess.check_output([a.remora,'dev','--parse','--json','-f',str(a.source)],timeout=a.timeout))
    manifest={'source':str(a.source.resolve()),'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),
              'definition_count':len(forms),'definitions_sha256':hashlib.sha256(defs.encode()).hexdigest(),
              'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','atol':a.atol,'rtol':a.rtol,
              'scope':'Exact helper bodies with small typed entries; not full 608x608 network inference','cases':[]}
    (a.out/'definitions.remora').write_text(defs)
    def save(): (a.out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    for case in cases():
        name=case['name']
        if a.case and name not in a.case:continue
        entry='(entry (main '+' '.join(f'({n} {ty(sh)})' for n,sh in case['args'])+')\n'+case['body']+')\n'
        body_ast=json.loads(subprocess.check_output([a.remora,'dev','--parse','--json','-e',case['body']],timeout=a.timeout))
        selected,dependency_names=reachable_definitions(forms,parsed,body_ast)
        path=a.out/(name+'.remora');path.write_text(selected+entry)
        record_result={'name':name,'retained_definitions':dependency_names,'test_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'lowerings':{},'fixtures':[]}
        manifest['cases'].append(record_result);save()
        modules={}
        for lowering in a.lowerings:
            started=time.monotonic()
            try:
                code=compile_source(source=path,remora=a.remora,lowering=lowering,timeout=a.timeout)
                generated=a.out/(name+'_'+lowering+'.py');generated.write_text(code)
                spec=importlib.util.spec_from_file_location(name+'_'+lowering,generated);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
                modules[lowering]=m
                record_result['lowerings'][lowering]={'selected':m.LOWERING,'compile_seconds':time.monotonic()-started,'generated_sha256':hashlib.sha256(code.encode()).hexdigest()}
            except Exception as e:record_result['lowerings'][lowering]={'error':str(e)}
        save()
        for seed in a.seeds:
            arrays=fixture(case['args'],seed);np.savez(a.out/f'{name}_{seed}_inputs.npz',**arrays)
            args=[literal(arrays[n]) for n,_ in case['args']]
            cmd=[a.remora,'interpret','-f',str(path),'--']+args
            trial={'seed':seed,'comparisons':{}};record_result['fixtures'].append(trial)
            started=time.monotonic()
            try:
                proc=subprocess.run(cmd,capture_output=True,text=True,timeout=a.timeout)
                (a.out/f'{name}_{seed}_interpreter.txt').write_text(proc.stdout)
                (a.out/f'{name}_{seed}_interpreter.stderr').write_text(proc.stderr)
                if proc.returncode:raise RuntimeError(proc.stderr or proc.stdout)
                expected=parse_result(proc.stdout);np.save(a.out/f'{name}_{seed}_reference.npy',expected)
                trial['interpreter_seconds']=time.monotonic()-started
                tensors={k:torch.from_numpy(v.copy()) for k,v in arrays.items()}
                for lowering,m in modules.items():
                    try:
                        actual=m.ENTRIES['main'](*(tensors[n] for n,_ in case['args'])).detach().cpu().numpy()
                        np.save(a.out/f'{name}_{seed}_{lowering}.npy',actual)
                        trial['comparisons'][lowering]=compare(actual,expected,a.atol,a.rtol)
                    except Exception as e:trial['comparisons'][lowering]={'passed':False,'error':str(e)}
                if case['reference']:
                    actual=case['reference'](tensors).numpy()
                    trial['comparisons']['independent_torch']=compare(actual,expected,a.atol,a.rtol)
            except Exception as e:trial['error']=str(e)
            save();print(name,seed,json.dumps(trial.get('comparisons') or trial),flush=True)
    failures=[(c['name'],t.get('seed')) for c in manifest['cases'] for t in c['fixtures'] if 'error' in t or any(not v['passed'] for v in t['comparisons'].values())]
    compile_failures=[(c['name'],k) for c in manifest['cases'] for k,v in c['lowerings'].items() if 'error' in v]
    manifest['passed']=bool(manifest['cases']) and not failures and not compile_failures;manifest['failed_fixtures']=failures;manifest['failed_compilations']=compile_failures;save()
    return 0 if manifest['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT.parent/'examples/darknet/yolov4.remora')
    p.add_argument('--remora',default=os.getenv('REMORA','remora'));p.add_argument('--out',type=Path,default=ROOT/'.build/yolo-differential')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,17]);p.add_argument('--lowerings',nargs='+',choices=['auto','static','general'],default=['auto','general'])
    p.add_argument('--case',nargs='+');p.add_argument('--timeout',type=float,default=120);p.add_argument('--atol',type=float,default=1e-5);p.add_argument('--rtol',type=float,default=1e-4)
    raise SystemExit(run(p.parse_args()))
