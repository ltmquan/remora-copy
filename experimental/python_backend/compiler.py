"""Experimental YOLO tensor lowering from the Remora parser's JSON AST.

This is a checked library-idiom backend, not a general Remora compiler. Composite
functions are expanded from source; supported library definitions are pinned to
reviewed ASTs, so redefining a recognized name cannot silently change semantics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

INTRINSICS = {'conv-bn-mish', 'conv-bn-leaky', 'conv-bias', 'shortcut',
              'route2', 'spp', 'upsample-nearest', 'yolo-head'}
COMPOSITES = {'res-block', 'csp-stage-1', 'csp-stage-x2', 'csp-stage-x4',
              'csp-stage-x8', 'panet-5conv', 'panet-3conv'}

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
    if t == 'Sub': return a[0] - sum(a[1:])
    raise ValueError(f'Unsupported static expression {t}')

class Compiler:
    def __init__(self, ast, contracts):
        self.defs = {d['bind']['var']: d['bind'] for d in ast['progDecs'] if d['tag'] == 'Def'}
        for name, expected in contracts.items():
            if name not in self.defs or digest(self.defs[name]) != expected:
                raise ValueError(f'Library contract changed: {name}; review lowering before compiling')
        if 'append' in self.defs:
            raise ValueError('Shadowing builtin append is unsupported')
        self.entry = next(d for d in ast['progDecs'] if d['tag'] == 'Entry' and d['var'] == 'main')
        self.nodes = []
        self.inputs = {p['var']: static(p['type']['shape'], {}) for p in self.entry['params']}

    def emit(self, op, *args):
        ref = {'ref': len(self.nodes)}
        self.nodes.append({'op': op, 'args': args})
        return ref

    def conv(self, dims, args, activation):
        q, k, ic, oc, pad, stride, remainder = dims
        if not (0 <= remainder < stride and pad < k): raise ValueError('Invalid convolution dimensions')
        return self.emit('conv', args[0], args[1], stride, pad, activation,
                         [ic, q * stride + k + remainder - 2 * pad, oc, q + 1, k])

    def call(self, name, types, dims, args):
        if name.startswith('conv-') and name in INTRINSICS:
            return self.conv(dims, args, {'conv-bn-mish':'mish', 'conv-bn-leaky':'leaky', 'conv-bias':'linear'}[name])
        if name == 'shortcut': return self.emit('add', *args)
        if name in ('route2', 'append'): return self.emit('cat', args, 0)
        if name == 'spp': return self.emit('spp', *args)
        if name == 'upsample-nearest': return self.emit('upsample', args[0], dims[2])
        if name == 'yolo-head':
            s, ic, mid, classes = dims
            x, fp, hp, aw, ah, sxy, net = args
            x = self.conv([s-1,3,ic,mid,1,1,0], [x,fp], 'leaky')
            x = self.conv([s-1,1,mid,3*(classes+5),0,1,0], [x,hp], 'linear')
            return self.emit('decode', x, aw, ah, sxy, net, classes)
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
        if t == 'Var': return env[x['name']]
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
            values = [a['lit']['lit'] for a in x['atoms']]
            return values[0] if not x['dims'] else values
        if t in ('App', 'IApp', 'TApp'):
            groups = {'App':[], 'IApp':[], 'TApp':[]}
            while x['tag'] in groups:
                groups[x['tag']].insert(0,x['arg']); x = x['fun']
            if x['tag'] != 'Var': raise ValueError('Unsupported higher-order application')
            return self.call(x['name'], groups['TApp'],
                             [static(a,env) for a in groups['IApp']],
                             [self.expr(a,env) for a in groups['App']])
        raise ValueError(f'Unsupported expression {t}')

    def compile(self):
        out = self.expr(self.entry['expr'], {k:{'input':k} for k in self.inputs})
        return {'inputs':self.inputs, 'nodes':self.nodes, 'output':out}

def generate(graph):
    def code(x):
        if isinstance(x,dict):
            if 'ref' in x: return f'v{x["ref"]}'
            if 'input' in x: return f'inputs[{x["input"]!r}]'
            return '{'+', '.join(repr(k)+': '+code(v) for k,v in x.items())+'}'
        if isinstance(x,(list,tuple)): return '['+', '.join(map(code,x))+']'
        return repr(x)
    lines = ['# Generated from Remora AST. Do not edit.', 'def forward(ops, inputs):']
    for i,n in enumerate(graph['nodes']):
        lines.append(f'    v{i} = ops.{n["op"]}('+', '.join(map(code,n['args']))+')')
    lines.append('    return '+code(graph['output']))
    return '\n'.join(lines)+'\n'

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('ast'); p.add_argument('--output',required=True)
    a=p.parse_args(); ast=json.loads(Path(a.ast).read_text())
    contracts=json.loads(Path(__file__).with_name('contracts.json').read_text())
    graph=Compiler(ast,contracts).compile(); graph['ast_sha256']=digest(ast)
    Path(a.output).write_text(generate(graph))
    Path(a.output).with_suffix('.json').write_text(json.dumps(graph,indent=2))
    print(f'Generated {len(graph["nodes"])} operations, {sum(n["op"]=="conv" for n in graph["nodes"])} convolutions')
