"""Compile the complete type-checked expression grammar to Python closures.

Types and index arithmetic become generated Python expressions; source ASTs are
never shipped to or walked by the runtime. No monomorphization is required.
"""
import builtins
import hashlib
import json
from pathlib import Path
import keyword

from .compiler import BINARY, UNARY, CompileError, Generator, ident, key, typ


class GeneralGenerator(Generator):
    def lookup(self, v, env):
        return env.get(key(v), f'_g_unbound({v["varName"]!r})')

    def dim_expr(self, d, env):
        tag = d['tag']
        if tag == 'DimN': return repr(d['val'])
        if tag == 'DimVar': return self.lookup(d['name'], env)
        xs = [self.dim_expr(x, env) for x in d['dims']]
        if tag == 'Add': return '(' + ' + '.join(xs or ['0']) + ')'
        if tag == 'Mul': return '(' + ' * '.join(xs or ['1']) + ')'
        if tag == 'Sub':
            return '0' if not xs else '-(' + xs[0] + ')' if len(xs) == 1 else '(' + ' - '.join(xs) + ')'
        raise CompileError(f'unknown dimension {tag}')

    def shape_expr(self, s, env):
        tag = s['tag']
        if tag == 'ShapeVar': return self.lookup(s['name'], env)
        if tag == 'ShapeDim': return '(' + self.dim_expr(s['dim'], env) + ',)'
        if tag == 'Concat': return '(' + ' + '.join([self.shape_expr(x, env) for x in s['shapes']] or ['()']) + ')'
        raise CompileError(f'unknown shape {tag}')

    def ispace(self, x, env):
        return f'_g_checked_shape(({self.dim_expr(x["dim"], env)},))[0]' if x['tag'] == 'Dim' else f'_g_checked_shape({self.shape_expr(x["shape"], env)})'

    def atom_desc(self, a, env):
        tag = a['tag']
        if tag in ('Int', 'Float', 'Bool'): return repr(tag)
        if tag == 'AtomTypeVar': return self.lookup(a['name'], env)
        if tag == '(:->)': return "('function', " + self.desc(a['in'], env) + ', ' + self.desc(a['out'], env) + ')'
        if tag in ('Pi', 'Forall'):
            p = a['param']; name = ident(p['name'])
            inner = env.copy(); inner[key(p['name'])] = name
            kind = 'type' if tag == 'Forall' else 'index'
            return f"('poly', {kind!r}, lambda {name}: {self.desc(a['body'], inner)})"
        if tag == 'Record':
            return "('record', {" + ', '.join(repr(k) + ': ' + self.desc(v, env) for k, v in a['fields']) + '})'
        if tag == 'Sigma':
            p = a['param']; name = ident(p['name'])
            inner = env.copy()
            inner[key(p['name'])] = name
            kind = 'dim' if p['tag'] == 'DimParam' else 'shape'
            return f"('box', {kind!r}, lambda {name}: {self.desc(a['body'], inner)})"
        raise CompileError(f'unknown atom type {tag}')

    def desc(self, t, env):
        return '(' + self.atom_desc(t['arrayTypeAtom'], env) + ', ' + self.shape_expr(t['arrayTypeShape'], env) + ')'

    def type_arg(self, t, env):
        if t['tag'] == 'AtomType': return self.atom_desc(t['type'], env)
        if t['tag'] == 'ArrayType': return self.desc(t['type'], env)
        raise CompileError(f'unknown type argument {t["tag"]}')

    def closure(self, params, body, kind, lines, env):
        if not params:
            return self.expr(body, lines, env)
        p = params[0]
        v = p['var'] if kind == 'value' else p['name']
        arg, fn = ident(v), self.fresh()
        inner_env = env.copy(); inner_env[key(v)] = arg
        inner = []
        value = self.closure(params[1:], body, kind, inner, inner_env)
        lines.append(f'def {fn}({arg}):')
        lines.extend('    ' + line for line in inner + ['return ' + value])
        param = self.desc(typ(p), env) if kind == 'value' else 'None'
        return f'_Objects((), [_Closure({fn}, {param}, {kind!r}, _device)])'

    def bind(self, b, lines, env):
        tag = b['tag']
        if tag == 'BindVal':
            value = self.expr(b['expr'], lines, env)
            name = self.emit(lines, value)
            env[key(b['var'])] = name
        elif tag in ('BindFun', 'BindTFun', 'BindIFun', 'Entry'):
            kind = {'BindFun': 'value', 'BindTFun': 'type', 'BindIFun': 'index', 'Entry': 'value'}[tag]
            value = self.closure(b['params'], b['expr'], kind, lines, env)
            env[key(b['var'])] = self.emit(lines, value)
        elif tag == 'BindType':
            env[key(b['var']['name'])] = self.type_arg(b['type'], env)
        elif tag == 'BindISpace':
            env[key(b['var']['name'])] = self.ispace(b['ispace'], env)
        else:
            raise CompileError(f'unknown binding {tag}')

    def atom(self, a, lines, env):
        tag = a['tag']
        if tag == 'Base':
            lit = a['lit']; val = lit['lit']
            if val is None: raise CompileError('frontend JSON cannot represent a nonfinite literal')
            return f'_g_tensor({val!r}, {lit["tag"].removesuffix("Val")!r}, _device)'
        if tag in ('Lambda', 'TLambda', 'ILambda'):
            return self.closure([a['param']], a['body'], {'Lambda': 'value', 'TLambda': 'type', 'ILambda': 'index'}[tag], lines, env)
        if tag == 'Box':
            value = self.expr(a['array'], lines, env)
            return f'_Objects((), [Box({self.ispace(a["ispace"], env)}, {value})])'
        raise CompileError(f'unknown atom {tag}')

    def expr(self, e, lines, env):
        tag = e['tag']
        if tag == 'Var':
            v = e['name']
            if key(v) in env: return env[key(v)]
            return f'_g_annotate(_g_intrinsic({v["varName"]!r}, _device), lambda: {self.desc(typ(e), env)})'
        if tag == 'Array':
            values = [self.atom(a, lines, env) for a in e['atoms']]
            return self.emit(lines, f'_g_frame({tuple(e["dims"])!r}, [{", ".join(values)}], lambda: {self.desc(typ(e), env)}, _device)')
        if tag in ('EmptyArray', 'EmptyFrame'):
            return self.emit(lines, f'_g_empty({self.desc(typ(e), env)}, _device)')
        if tag == 'Frame':
            values = [self.expr(x, lines, env) for x in e['exprs']]
            return self.emit(lines, f'_g_frame({tuple(e["dims"])!r}, [{", ".join(values)}], lambda: {self.desc(typ(e), env)}, _device)')
        if tag == 'App':
            fn = self.expr(e['fun'], lines, env)
            arg = self.expr(e['arg'], lines, env)
            ft = typ(e['fun'])['arrayTypeAtom']
            if ft['tag'] == 'AtomTypeVar':
                raise CompileError('unresolved function type in checked application')
            cell = self.shape_expr(ft['in']['arrayTypeShape'], env)
            return self.emit(lines, f'_g_apply({fn}, {arg}, {cell}, lambda: {self.desc(typ(e), env)}, _device)')
        if tag in ('TApp', 'IApp'):
            fn = self.expr(e['fun'], lines, env)
            arg = self.type_arg(e['arg'], env) if tag == 'TApp' else self.ispace(e['arg'], env)
            return self.emit(lines, f'_g_specialize({fn}, {arg}, lambda: {self.desc(typ(e), env)}, _device)')
        if tag == 'Let':
            env = env.copy()
            for b in e['binds']: self.bind(b, lines, env)
            return self.expr(e['body'], lines, env)
        if tag == 'Unbox':
            target = self.expr(e['target'], lines, env)
            ix, var, fn = ident(e['ispace']['name']), ident(e['var']), self.fresh()
            inner_env = env.copy(); inner_env[key(e['ispace']['name'])] = ix; inner_env[key(e['var'])] = var
            inner = []
            value = self.expr(e['body'], inner, inner_env)
            lines.append(f'def {fn}({ix}, {var}):')
            lines.extend('    ' + line for line in inner + ['return ' + value])
            return self.emit(lines, f'_g_unbox({target}, {fn}, lambda: {self.desc(typ(e), env)}, _device)')
        if tag == 'Struct':
            frame = self.shape_expr(typ(e)['arrayTypeShape'], env)
            fields = []
            for name, cell, value in e['fields']:
                v = self.expr(value, lines, env)
                fields.append(f'({name!r}, {self.shape_expr(cell, env)}, {v})')
            return self.emit(lines, f'_g_struct({frame}, [{", ".join(fields)}], lambda: {self.desc(typ(e), env)}, _device)')
        if tag == 'FieldProj':
            value = self.expr(e['expr'], lines, env)
            return self.emit(lines, f'_g_field({value}, {e["field"]!r}, lambda: {self.desc(typ(e), env)}, _device)')
        raise CompileError(f'unknown checked expression {tag}')

    def generate(self):
        lines, env, exports = [], {}, []
        entries = [d for d in self.ast['progDecs'] if d['tag'] == 'Entry']
        # Entry bodies (including zero-argument entries) must be delayed.
        for d in self.ast['progDecs']:
            if d['tag'] == 'Def':
                self.bind(d['bind'], lines, env)
            elif d['tag'] == 'Entry':
                name = self.fresh()
                params = d['params']
                inner_env = env.copy()
                inner_env.update({key(p['var']): ident(p['var']) for p in params})
                body = []
                result = self.expr(d['expr'], body, inner_env)
                lines.append(f'def {name}(' + ', '.join(ident(p['var']) for p in params) + '):')
                lines.extend('    ' + line for line in body + ['return ' + result])
                specs = '[' + ', '.join(self.desc(typ(p), env) for p in params) + ']'
                exports.append(f'{d["var"]["varName"]!r}: ({name}, {specs})')
                # Entries can be called from subsequent declarations.
                env[key(d['var'])] = self.emit(lines, f'_g_curry({name}, {len(params)}, device=_device)') if params else name + '()'
            else:
                raise CompileError(f'unknown declaration {d["tag"]}')
        lines.append('return {' + ', '.join(exports) + '}')
        chunks = [Path(__file__).with_name('runtime.py').read_text(), Path(__file__).with_name('general_runtime.py').read_text()]
        primitives = [repr(k) + ': (' + v + ', 2)' for k, v in BINARY.items()]
        primitives += [repr(k) + ': (' + v + ', 1)' for k, v in UNARY.items()]
        chunks.append('_G_PRIMITIVES = {' + ',\n'.join(primitives) + '}')
        chunks.append('def _g_program(_device):\n' + '\n'.join('    ' + line for line in lines))
        public_entries = []
        for d in entries:
            name = d['var']['varName']
            public = name if name.isidentifier() and not keyword.iskeyword(name) and not name.startswith('_') and name not in set(dir(builtins)) | {'torch', 'math', 'functools', 'ENTRIES', 'Box', 'ObjectArray', 'sys', 'dataclass', 'LOWERING'} else '_entry_' + str(key(d['var']))
            params = ['arg' + str(i) for i in range(len(d['params']))]
            args = '[' + ', '.join(params) + ']'
            chunks.append(f'def {public}(' + ', '.join(params + ['*, device=None']) + '):\n'
                          + f'    _device = _g_device({args}, device)\n'
                          + f'    _fn, _specs = _g_program(_device)[{name!r}]\n'
                          + f'    _args = [_g_import(v, s, _device) for v, s in zip({args}, _specs)]\n'
                          + '    return _g_export(_fn(*_args))\n')
            public_entries.append(repr(name) + ': ' + public)
        chunks.append('ENTRIES = {' + ', '.join(public_entries) + '}')
        chunks.append("LOWERING = 'general'\nif __name__ == '__main__':\n    _g_cli()\n")
        return '\n\n'.join(chunks)


def compile_general_ast(ast):
    if ast.get('tag') != 'Prog':
        ast = {'tag': 'Prog', 'progDecs': [{'tag': 'Entry', 'var': {'varName': 'main', 'varTag': {'getTag': -1}}, 'params': [], 'expr': ast}]}
    digest = hashlib.sha256(json.dumps(ast, sort_keys=True).encode()).hexdigest()
    return '# Generated by remora-torch (general lowering); checked AST SHA256: ' + digest + '\n' + GeneralGenerator(ast).generate()
