"""Source generation from Remora's monomorphic, defunctionalized typed AST.

The existing frontend owns parsing, imports, type checking, rank inference,
lambda lifting and specialization. This backend does not invoke Futhark.
"""
import builtins
import hashlib
import json
import keyword
import math
from pathlib import Path
import re
import subprocess


class CompileError(ValueError):
    """A frontend failure or an unsupported source construct."""


def key(v):
    return v['varTag']['getTag']


def ident(v):
    return '_r_' + re.sub(r'[^a-zA-Z0-9_]', '_', v['varName']) + '_' + str(key(v)).replace('-', 'n')


def typ(node):
    info = node.get('info')
    if isinstance(info, list):
        info = info[0]
    if not isinstance(info, dict) or info.get('tag') != 'A':
        raise CompileError(f"expected typed array information for {node.get('tag')}")
    return info


def dim(d):
    tag = d['tag']
    if tag == 'DimN':
        return d['val']
    if tag in ('Add', 'Mul', 'Sub'):
        ns = [dim(x) for x in d['dims']]
        return sum(ns) if tag == 'Add' else math.prod(ns) if tag == 'Mul' else (0 if not ns else -ns[0] if len(ns) == 1 else ns[0] - sum(ns[1:]))
    raise CompileError('dynamic dimensions are not supported by this static-shape backend')


def shape(s):
    tag = s['tag']
    if tag == 'Concat':
        result = sum((shape(x) for x in s['shapes']), ())
    elif tag == 'ShapeDim':
        result = (dim(s['dim']),)
    else:
        raise CompileError('unresolved shape variable; specialize index arguments before compilation')
    if any(x < 0 for x in result):
        raise CompileError(f'negative shape {result}')
    return result


def dtype(t):
    name = t['arrayTypeAtom']['tag']
    try:
        # Match the interpreter's machine Int and single-precision Float.
        return {'Int': 'torch.int64', 'Float': 'torch.float32', 'Bool': 'torch.bool'}[name]
    except KeyError:
        raise CompileError(f'unsupported tensor element type {name}') from None


def cell_shapes(t, frame=()):
    sh = frame + shape(t['arrayTypeShape'])
    if t['arrayTypeAtom']['tag'] == 'Record':
        return {k: cell_shapes(v, sh) for k, v in t['arrayTypeAtom']['fields']}
    return sh


def spec(t):
    sh = shape(t['arrayTypeShape'])
    if t['arrayTypeAtom']['tag'] == 'Record':
        return '{' + ', '.join(repr(k) + ': ' + spec_with_frame(v, sh)
                               for k, v in t['arrayTypeAtom']['fields']) + '}'
    return f'({sh!r}, {dtype(t)})'


def spec_with_frame(t, frame):
    t = dict(t, arrayTypeShape={'tag': 'Concat', 'shapes': [
        {'tag': 'ShapeDim', 'dim': {'tag': 'DimN', 'val': d}} for d in frame
    ] + [t['arrayTypeShape']]})
    return spec(t)


BINARY = {
    '+': 'torch.add', '-': 'torch.sub', '*': 'torch.mul', '/': "lambda x, y: torch.div(x, y, rounding_mode='floor')",
    '^': 'torch.pow', 'mod': 'torch.remainder', 'max': 'torch.maximum', 'min': 'torch.minimum',
    'bit-and': 'torch.bitwise_and', 'bit-or': 'torch.bitwise_or', 'bit-xor': 'torch.bitwise_xor',
    'shl': 'torch.bitwise_left_shift', 'shr': 'torch.bitwise_right_shift',
    '==': 'torch.eq', '!=': 'torch.ne', '<': 'torch.lt', '>': 'torch.gt', '<=': 'torch.le', '>=': 'torch.ge',
    'and': 'torch.logical_and', 'or': 'torch.logical_or', 'bool.==': 'torch.eq', 'bool.!=': 'torch.ne',
}
for _name in ('+', '-', '*', '^', 'max', 'min', '==', '!=', '<', '>', '<=', '>='):
    BINARY['f.' + _name] = BINARY[_name]
BINARY['f./'] = 'torch.div'
# Haskell Ord Float chooses an operand on unordered (NaN) comparisons.
BINARY['f.min'] = 'lambda x, y: torch.where(x <= y, x, y)'
BINARY['f.max'] = 'lambda x, y: torch.where(x <= y, y, x)'
BINARY['shl'] = 'lambda x, y: torch.where(y >= 0, x << y.clamp(min=0), x >> (-y).clamp(min=0))'
BINARY['shr'] = 'lambda x, y: torch.where(y >= 0, x >> y.clamp(min=0), x << (-y).clamp(min=0))'
UNARY = {
    'sqrt': 'torch.sqrt', 'f.sqrt': 'torch.sqrt', 'bit-not': 'torch.bitwise_not', 'popc': '_popc',
    'not': 'torch.logical_not', 'i->f': 'lambda x: x.to(torch.float32)',
    'bool->f': 'lambda x: x.to(torch.float32)', 'bool->i': 'lambda x: x.to(torch.int64)',
    'i->bool': 'lambda x: x != 0', 'truncate': 'lambda x: torch.trunc(x).to(torch.int64)',
    'round': 'lambda x: torch.round(x).to(torch.int64)', 'ceiling': 'lambda x: torch.ceil(x).to(torch.int64)',
    'floor': 'lambda x: torch.floor(x).to(torch.int64)', 'f.ln': 'torch.log',
    'f.tanh': 'torch.tanh', 'f.exp': 'torch.exp',
}


class Generator:
    def __init__(self, ast):
        self.ast = ast
        self.counter = 0
        self.defs = {}
        self.pending = []
        self.used = set()
        for d in ast['progDecs']:
            b = d['bind'] if d['tag'] == 'Def' else d
            if b['tag'] in ('BindVal', 'BindFun', 'Entry'):
                self.defs[key(b['var'])] = b

    def fresh(self):
        self.counter += 1
        return '_v' + str(self.counter)

    def emit(self, lines, expression):
        name = self.fresh()
        lines.append(f'{name} = {expression}')
        return name

    def ref(self, v):
        k = key(v)
        if k not in self.used:
            self.used.add(k)
            self.pending.append(self.defs[k])
        return ident(v)

    def intrinsic(self, name, t):
        if name in BINARY:
            return '(' + BINARY[name] + ')'
        if name in UNARY:
            return '(' + UNARY[name] + ')'
        # Intrinsics have already been specialized by the Remora frontend.
        params = []
        result = t
        while result['arrayTypeAtom']['tag'] == '(:->)':
            params.append(result['arrayTypeAtom']['in'])
            result = result['arrayTypeAtom']['out']
        ops = {
            'head': 'lambda x: _tree_map(lambda v: v[0], x)', 'tail': 'lambda x: _tree_map(lambda v: v[1:], x)',
            'length': 'lambda x: torch.scalar_tensor(_first_tensor(x).shape[0], dtype=torch.int64, device=_device)',
            'append': 'lambda x, y: _tree_map(lambda a, b: torch.cat((a, b), dim=0), x, y)',
            'reverse': 'lambda x: _tree_map(lambda v: torch.flip(v, (0,)), x)',
            'transpose2d': 'lambda x: _tree_map(lambda v: v.transpose(0, 1), x)',
            'sum': 'lambda x: x.sum(dtype=torch.int64)',
            'index': '_index', 'index2d': '_index2d',
            'reduce': 'lambda f, x: _reduce(f, x)',
            'reduce/zero': 'lambda f, z, x: _reduce(f, x, z)',
            'fold': 'lambda f, z, x: _reduce(f, x, z)',
            'fold-right': 'lambda f, z, x: _reduce(f, x, z, right=True)',
        }
        if name in ops:
            return '(' + ops[name] + ')'
        if name in ('reshape', 'flatten'):
            return f'(lambda x: _reshape(x, {cell_shapes(result)!r}))'
        if name == 'iota/static':
            sh = shape(result['arrayTypeShape'])
            return f'torch.arange({math.prod(sh)}, dtype=torch.int64, device=_device).reshape({sh!r})'
        raise CompileError(f'unsupported intrinsic {name!r}; no fallback or library substitution is performed')

    def expr(self, e, lines, env):
        tag = e['tag']
        if tag == 'Var':
            v = e['name']
            if key(v) in env:
                return env[key(v)]
            if key(v) in self.defs:
                b = self.defs[key(v)]
                ref = self.ref(v)
                return f'{ref}(_device)' if b['tag'] == 'BindVal' else f'functools.partial({ref}, _device)'
            return self.intrinsic(v['varName'], typ(e))
        if tag == 'App':
            args, shapes = [], []
            head = e
            while head['tag'] == 'App':
                ft = typ(head['fun'])['arrayTypeAtom']
                if ft['tag'] != '(:->)':
                    raise CompileError('application of a non-function')
                args.append(head['arg'])
                shapes.append(cell_shapes(ft['in']))
                head = head['fun']
            args.reverse()
            shapes.reverse()
            if shape(typ(head)['arrayTypeShape']):
                raise CompileError('arrays of functions are not supported; use a scalar function with lifted arguments')
            fn = self.expr(head, lines, env)
            values = [self.expr(x, lines, env) for x in args]
            partial = typ(e)['arrayTypeAtom']['tag'] == '(:->)'
            output = 'None' if partial else spec(typ(e))
            return self.emit(lines, f'_apply({fn}, [{", ".join(values)}], {shapes!r}, {output}, _device, partial={partial})')
        if tag == 'Array':
            if any(a['tag'] != 'Base' for a in e['atoms']):
                raise CompileError('non-data array survived defunctionalization (function arrays or boxes)')
            vals = [a['lit']['lit'] for a in e['atoms']]
            if any(v is None for v in vals):
                raise CompileError('frontend JSON cannot represent this nonfinite literal')
            return self.emit(lines, f'torch.tensor({vals!r}, dtype={dtype(typ(e))}, device=_device).reshape({tuple(e["dims"])!r})')
        if tag in ('EmptyArray', 'EmptyFrame'):
            return self.emit(lines, f'_empty({spec(typ(e))}, _device)')
        if tag == 'Frame':
            vs = [self.expr(x, lines, env) for x in e['exprs']]
            return self.emit(lines, f'_reshape(_tree_map(lambda *xs: torch.stack(xs), {", ".join(vs)}), {cell_shapes(typ(e))!r})')
        if tag == 'Let':
            env = env.copy()
            for b in e['binds']:
                if b['tag'] == 'BindVal':
                    env[key(b['var'])] = self.expr(b['expr'], lines, env)
                elif b['tag'] == 'BindFun':
                    local = ident(b['var'])
                    params = b['params']
                    inner_env = dict(env)
                    inner_env.update({key(p['var']): ident(p['var']) for p in params})
                    inner = []
                    result = self.expr(b['expr'], inner, inner_env)
                    lines.append(f'def {local}(' + ', '.join(ident(p['var']) for p in params) + '):')
                    lines.extend('    ' + line for line in inner + ['return ' + result])
                    env[key(b['var'])] = local
                elif b['tag'] in ('BindType', 'BindISpace'):
                    # Type/index aliases have already been substituted in typed nodes.
                    continue
                else:
                    raise CompileError(f'local {b["tag"]} survived frontend lowering')
            return self.expr(e['body'], lines, env)
        if tag == 'Struct':
            sh = shape(typ(e)['arrayTypeShape'])
            fields = []
            for name, cell, value in e['fields']:
                v = self.expr(value, lines, env)
                field_type = dict(typ(e)['arrayTypeAtom']['fields'])[name]
                fields.append(f'{name!r}: _prefix_expand({v}, {cell_shapes(field_type)!r}, {sh!r})')
            return self.emit(lines, '{' + ', '.join(fields) + '}')
        if tag == 'FieldProj':
            value = self.expr(e['expr'], lines, env)
            return self.emit(lines, f'{value}[{e["field"]!r}]')
        raise CompileError(f'unsupported AST construct {tag}; backend requires static, monomorphic data')

    def function(self, b):
        lines = []
        params = b.get('params', [])
        env = {key(p['var']): ident(p['var']) for p in params}
        value = self.expr(b['expr'], lines, env)
        lines.append('return ' + value)
        name = ident(b['var'])
        header = f'def {name}(' + ', '.join(['_device'] + list(env.values())) + '):'
        return header + '\n' + '\n'.join('    ' + x for x in lines)

    def generate(self):
        # Literal globals have no effects or evaluation errors to preserve.
        # Nonliteral global initialization still uses the general path.
        for d in self.ast['progDecs']:
            if d['tag'] == 'Def' and d['bind']['tag'] == 'BindVal':
                value = d['bind']['expr']
                if value['tag'] != 'Array' or any(a['tag'] != 'Base' for a in value['atoms']):
                    raise CompileError('nonliteral global initialization requires general lowering')
        entries = [d for d in self.ast['progDecs'] if d['tag'] == 'Entry']
        if not entries:
            raise CompileError('program contains no entry points')
        chunks = [Path(__file__).with_name('runtime.py').read_text()]
        wrappers = []
        names = []
        for e in entries:
            spec(typ(e['expr']))  # Function/box results require the general public ABI.
            name = e['var']['varName']
            ref = self.ref(e['var'])
            specs = [spec(typ(p)) for p in e['params']]
            if any(typ(p)['arrayTypeAtom']['tag'] not in ('Int', 'Float', 'Bool') for p in e['params']):
                raise CompileError('entry parameters must be tensors of Int, Float or Bool')
            public = name if name.isidentifier() and not keyword.iskeyword(name) and not name.startswith('_') and name not in set(dir(builtins)) | {'torch', 'math', 'functools', 'ENTRIES', 'LOWERING'} else '_entry_' + str(key(e['var']))
            params = [f'arg{i}' for i in range(len(specs))]
            wrappers.append(f'def {public}(' + ', '.join(params + ['*, device=None']) + '):\n'
                + '    ' + repr(f'Remora entry {name}: tensor arguments preserve autograd.') + '\n'
                + f'    _args, _device = _prepare([{", ".join(params)}], [{", ".join(specs)}], device)\n'
                + f'    return {ref}(_device, *_args)\n')
            names.append(f'{name!r}: {public}')
        while self.pending:
            b = self.pending.pop(0)
            try:
                chunks.append(self.function(b))
            except CompileError as err:
                raise CompileError(f'in {b["var"]["varName"]}: {err}') from err
        chunks.extend(wrappers)
        chunks.append('ENTRIES = {' + ', '.join(names) + '}')
        chunks.append("LOWERING = 'static'")
        chunks.append("if __name__ == '__main__':\n    _run_cli()\n")
        return '\n\n'.join(chunks)


def compile_ast(ast):
    """Compile a trusted `dev --defunctionalize --json` AST to Python source."""
    if ast.get('tag') != 'Prog':
        # `dev -e` returns an expression rather than a whole program.
        ast = {'tag': 'Prog', 'progDecs': [{'tag': 'Entry', 'var': {
            'varName': 'main', 'varTag': {'getTag': -1}}, 'params': [], 'expr': ast}]}
    generated = Generator(ast).generate()
    digest = hashlib.sha256(json.dumps(ast, sort_keys=True).encode()).hexdigest()
    return f'# Generated by remora-torch; typed AST SHA256: {digest}\n' + generated


def compile_source(*, source=None, expression=None, remora='remora', timeout=120, lowering='auto'):
    """Prefer native operations, then static tensors, then checked-AST lowering.

    `general` bypasses the frontend's monomorphization/defunctionalization
    restrictions; `static` requires the original tensor-only path.
    """
    if (source is None) == (expression is None):
        raise ValueError('provide exactly one of source or expression')
    if lowering not in ('auto', 'native', 'tensor', 'static', 'general'):
        raise ValueError('lowering must be auto, native, tensor, static, or general')
    def frontend(stage):
        cmd = [str(remora), 'dev', '--' + stage, '--json']
        cmd += ['-f', str(source)] if source is not None else ['-e', expression]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as err:
            raise CompileError(f'cannot run Remora frontend: {err}') from err
        if result.returncode:
            raise CompileError('Remora frontend failed:\n' + (result.stderr or result.stdout).strip())
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as err:
            raise CompileError('frontend did not return JSON; use a Remora binary with dev --json support') from err
    if lowering in ('auto', 'native'):
        from .native import compile_native_ast
        try:
            native = compile_native_ast(frontend('parse'))
        except CompileError:
            if lowering == 'native': raise
        else:
            frontend('typecheck')  # Never accept an unchecked native graph.
            return native
    if lowering != 'general':
        try:
            ast = frontend('defunctionalize')
            if lowering in ('auto', 'tensor'):
                from .tensor import compile_tensor_ast
                try:
                    return compile_tensor_ast(ast, require_optimization=lowering == 'auto')
                except CompileError:
                    if lowering == 'tensor': raise
            return compile_ast(ast)
        except CompileError:
            if lowering in ('static', 'tensor'):
                raise
    from .general import compile_general_ast
    return compile_general_ast(frontend('typecheck'))
