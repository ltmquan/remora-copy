"""Native tensor lowering adapted from experimental/python_backend/compiler.py.

YOLO tensor lowering from the Remora parser's JSON AST.

This is a checked library-idiom backend, not a general Remora compiler. Composite
functions are expanded from source; supported library definitions are pinned to
reviewed ASTs, so redefining a recognized name cannot silently change semantics.
"""
import hashlib
import json
import math
from pathlib import Path

INTRINSICS = {'conv-bn-mish', 'conv-bn-leaky', 'conv-bias', 'shortcut',
              'route2', 'spp', 'upsample-nearest', 'yolo-head'}

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

def static(x, env):
    t = x['tag']
    if t == 'DimN': return x['val']
    if t in ('DimVar', 'ShapeVar'): return env[x['name']]
    if t in ('Dim', 'ShapeDim'): return static(x['dim'], env)
    if t == 'Shape': return static(x['shape'], env)
    if t == 'Concat':
        out = []
        for v in x['shapes']:
            a = static(v, env)
            out.extend(a if isinstance(a, list) else [a])
        return out
    a = [static(v, env) for v in x['dims']]
    if t == 'Add': return sum(a)
    if t == 'Mul': return math.prod(a)
    if t == 'Sub': return 0 if not a else -a[0] if len(a)==1 else a[0]-sum(a[1:])
    raise ValueError(f'Unsupported static expression {t}')

class Compiler:
    def __init__(self, ast, contracts):
        declarations = [d['bind'] for d in ast['progDecs'] if d['tag'] == 'Def']
        self.defs = {b['var']: b for b in declarations}
        if len(self.defs) != len(declarations): raise ValueError('duplicate definitions')
        self.contracts = contracts
        self.verified = set()
        for b in declarations:
            if b['tag'] == 'BindVal':
                x = b['expr']
                if x['tag'] != 'Array' or any(a['tag'] != 'Base' for a in x['atoms']):
                    raise ValueError('effectful or computed global requires fallback')
        entries = [d for d in ast['progDecs'] if d['tag'] == 'Entry']
        if len(entries) != 1 or entries[0]['var'] != 'main':
            raise ValueError('native path currently requires one main entry')
        self.entry = entries[0]
        self.nodes = []
        self.shapes = []
        self.inputs = {}
        for p in self.entry['params']:
            if p['var'] in self.inputs: raise ValueError('duplicate entry parameter')
            t = p['type']
            if t['tag'] != 'TEArray' or t['elem']['tag'] != 'TEFloat':
                raise ValueError('native entries currently require float arrays')
            self.inputs[p['var']] = static(t['shape'], {})

    def verify(self, name):
        if name in self.verified: return
        contract = self.contracts[name]
        if name not in self.defs or digest(self.defs[name]) != contract['sha256']:
            raise ValueError(f'Library contract changed: {name}')
        self.verified.add(name)
        for dependency in contract['dependencies']:
            self.verify(dependency)
        for ref in contract['references']:
            if ref in self.defs and ref not in contract['dependencies'] and ref != name:
                raise ValueError(f'Shadowed library dependency: {ref}')

    def shape(self, x):
        if isinstance(x, dict):
            if 'input' in x: return self.inputs[x['input']]
            if 'ref' in x: return self.shapes[x['ref']]
            return {k:self.shape(v) for k,v in x.items()}
        if isinstance(x, list):
            if not x: return [0]
            sh=self.shape(x[0])
            if any(self.shape(v)!=sh for v in x): raise ValueError('ragged literal')
            return [len(x)]+sh
        return []

    def expect(self, x, shape):
        if self.shape(x) != shape:
            raise ValueError(f'native cell shape mismatch: {self.shape(x)} vs {shape}')

    def emit(self, op, *args):
        ref = {'ref': len(self.nodes)}
        if op == 'conv':
            x,p,stride,pad,activation,shape=args
            ic,n,oc,out,k=shape
            self.expect(x,[ic,n,n]);self.expect(p['w'],[oc,ic,k,k])
            for key in p:
                if key != 'w': self.expect(p[key],[oc])
            result=[oc,out,out]
        elif op == 'raw_conv':
            x,w,stride,pad,shape=args
            ic,n,oc,out,k=shape
            self.expect(x,[ic,n,n]);self.expect(w,[oc,ic,k,k]);result=[oc,out*out]
        elif op == 'spp':
            result=self.shape(args[0]).copy()
            if len(result)!=3:raise ValueError('pool cell rank')
            result[0]*=4
        elif op == 'upsample':
            x,scale=args;result=self.shape(x).copy()
            if len(result)!=3 or scale<=0:raise ValueError('upsample shape')
            result[1]*=scale;result[2]*=scale
        elif op == 'cat':
            xs,axis=args;result=self.shape(xs[0]).copy()
            for x in xs[1:]:
                sh=self.shape(x)
                if sh[1:]!=result[1:]:raise ValueError('cat shape')
                result[0]+=sh[0]
        elif op == 'add':
            result=self.shape(args[0]);self.expect(args[1],result)
        elif op == 'decode':
            x,aw,ah,sxy,net,classes=args;s=self.shape(x)[-1]
            self.expect(x,[3*(classes+5),s,s]);self.expect(aw,[3]);self.expect(ah,[3])
            self.expect(sxy,[]);self.expect(net,[]);result=[3*s*s,classes+5]
        else: raise ValueError(f'unknown native operation {op}')
        self.shapes.append(result)
        self.nodes.append({'op': op, 'args': args})
        return ref

    def conv(self, dims, args, activation):
        q, k, ic, oc, pad, stride, remainder = dims
        if not (0 <= remainder < stride and pad < k): raise ValueError('Invalid convolution dimensions')
        return self.emit('conv', args[0], args[1], stride, pad, activation,
                         [ic, q * stride + k + remainder - 2 * pad, oc, q + 1, k])

    def call(self, name, types, dims, args):
        if name in INTRINSICS or name == 'conv-2d/rem': self.verify(name)
        if name == 'conv-2d/rem':
            q,k,ic,oc,pad,stride,remainder=dims
            if stride<=0 or not 0<=remainder<stride or not 0<=pad<k:raise ValueError('convolution dimensions')
            return self.emit('raw_conv',args[0],args[1],stride,pad,[ic,q*stride+k+remainder-2*pad,oc,q+1,k])
        if name.startswith('conv-') and name in INTRINSICS:
            return self.conv(dims, args, {'conv-bn-mish':'mish', 'conv-bn-leaky':'leaky', 'conv-bias':'linear'}[name])
        if name == 'shortcut': return self.emit('add', *args)
        if name == 'append' and name in self.defs: raise ValueError('shadowed append')
        if name == 'route2':
            c1,c2,h,w=dims;self.expect(args[0],[c1,h,w]);self.expect(args[1],[c2,h,w])
            return self.emit('cat',args,0)
        if name == 'append':
            n,m,cell=dims;self.expect(args[0],[n]+cell);self.expect(args[1],[m]+cell)
            return self.emit('cat',args,0)
        if name == 'spp':
            n,c=dims;self.expect(args[0],[c,n,n]);return self.emit('spp',*args)
        if name == 'upsample-nearest':
            n,c,scale=dims;self.expect(args[0],[c,n,n]);return self.emit('upsample',args[0],scale)
        if name == 'yolo-head':
            s, ic, mid, classes = dims
            x, fp, hp, aw, ah, sxy, net = args
            x = self.conv([s-1,3,ic,mid,1,1,0], [x,fp], 'leaky')
            x = self.conv([s-1,1,mid,3*(classes+5),0,1,0], [x,hp], 'linear')
            return self.emit('decode', x, aw, ah, sxy, net, classes)
        if name not in self.defs:raise ValueError(f'unsupported function {name}')
        b = self.defs[name]
        env = {}
        supplied = {'BindTFun':types, 'BindIFun':dims, 'BindFun':args}
        while True:
            ps = b['params']; vals = supplied[b['tag']]
            if len(ps) != len(vals): raise ValueError(f'Arity mismatch: {name}')
            env.update({p.get('var',p.get('name')):v for p,v in zip(ps,vals)})
            if b['tag'] == 'BindFun': break
            b = b['expr']['binds'][0]
        return self.expr(b['expr'], env)

    def expr(self, x, env):
        t = x['tag']
        if t == 'Var':
            if x['name'] in env:return env[x['name']]
            b=self.defs[x['name']]
            if b['tag']!='BindVal':raise ValueError('function value')
            return self.expr(b['expr'],env)
        if t == 'Let':
            env = dict(env)
            for b in x['binds']:
                if b['tag'] == 'BindType': continue
                if b['tag'] != 'BindVal': raise ValueError(f'Unsupported local binding {b["tag"]}')
                env[b['var']] = self.expr(b['expr'], env)
            return self.expr(x['body'], env)
        if t == 'Struct': return {k:self.expr(v,env) for k,shape,v in x['fields']}
        if t == 'FieldProj': return self.expr(x['expr'],env)[x['field']]
        if t == 'Array':
            if any(a['tag']!='Base' for a in x['atoms']):raise ValueError('nonliteral array')
            values = [a['lit']['lit'] for a in x['atoms']]
            def nest(ds,vals):
                if not ds:return vals[0]
                cell=math.prod(ds[1:])
                return [nest(ds[1:],vals[i*cell:(i+1)*cell]) for i in range(ds[0])]
            return nest(x['dims'],values)
        if t in ('App', 'IApp', 'TApp'):
            groups = {'App':[], 'IApp':[], 'TApp':[]}
            while x['tag'] in groups:
                groups[x['tag']].insert(0,x['arg']); x = x['fun']
            if x['tag'] != 'Var': raise ValueError('Unsupported higher-order application')
            if x['name'] in env:raise ValueError('higher-order native call')
            return self.call(x['name'], groups['TApp'],
                             [static(a,env) for a in groups['IApp']],
                             [self.expr(a,env) for a in groups['App']])
        raise ValueError(f'Unsupported expression {t}')

    def compile(self):
        out = self.expr(self.entry['expr'], {k:{'input':k} for k in self.inputs})
        if not self.nodes or not isinstance(out,dict) or 'ref' not in out:raise ValueError('native tensor result required')
        return {'inputs':self.inputs, 'nodes':self.nodes, 'output':out, 'contracts':sorted(self.verified)}


def generate(graph):
    def code(x):
        if isinstance(x,dict):
            if 'ref' in x:return f'_n{x["ref"]}'
            if 'input' in x:return f'_args[{list(graph["inputs"]).index(x["input"])}]'
            return '{'+', '.join(repr(k)+': '+code(v) for k,v in x.items())+'}'
        if isinstance(x,(list,tuple)):return '['+', '.join(map(code,x))+']'
        return repr(x)
    root=Path(__file__).parent
    lines=[(root/'runtime.py').read_text(),(root/'native_runtime.py').read_text(),
           "LOWERING = 'native'", "NUMERICS = 'native float32; reduction reassociation permitted'",
           'NATIVE_CONTRACTS = '+repr(graph['contracts']),
           'NATIVE_OPS = '+repr([n['op'] for n in graph['nodes']])]
    params=[f'arg{i}' for i in range(len(graph['inputs']))]
    lines+=['def main('+', '.join(params+['*, device=None'])+'):',
            '    _args, _device = _prepare(['+', '.join(params)+'], '+repr([(tuple(sh),'FLOAT') for sh in graph['inputs'].values()]).replace("'FLOAT'",'torch.float32')+', device)',
            '    _ops = _NativeOps(_device)']
    for i,n in enumerate(graph['nodes']):
        lines.append(f'    _n{i} = _ops.{n["op"]}('+', '.join(map(code,n['args']))+')')
    lines+=['    return '+code(graph['output']),"ENTRIES = {'main': main}","if __name__ == '__main__':\n    _run_cli()"]
    return '\n'.join(lines)+'\n'


def compile_native_ast(ast):
    from .compiler import CompileError
    try:
        contracts=json.loads(Path(__file__).with_name('native_contracts.json').read_text())
        graph=Compiler(ast,contracts).compile()
        return '# Native lowering; parsed AST SHA256: '+digest(ast)+'\n'+generate(graph)
    except (ValueError,KeyError,TypeError,IndexError,RecursionError) as error:
        raise CompileError(f'native lowering unavailable: {error}') from error
