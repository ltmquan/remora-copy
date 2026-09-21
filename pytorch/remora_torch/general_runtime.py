"""Runtime for generated higher-order Remora code, not an AST interpreter.

Tensor data stays on its device. Data-dependent shapes, heterogeneous function
arrays and effects use Python control flow; they need not form one GPU graph.
"""
from dataclasses import dataclass
import sys


@dataclass
class Box:
    index: object
    value: object


@dataclass
class _Objects:
    shape: tuple
    values: list

    def __post_init__(self):
        if math.prod(self.shape) != len(self.values):
            raise ValueError('object array shape does not match element count')


@dataclass
class _Closure:
    call: object
    param: object = None
    kind: str = 'value'
    device: object = None
    result_type: object = None

    def __call__(self, arg):
        """Public scalar function interface; values are converted by parameter type."""
        if self.kind != 'value':
            result = self.call(arg)
            if self.result_type is not None:
                result = _g_annotate(result, lambda: self.result_type(arg))
            return _g_export(result)
        device = _g_device([arg], self.device)
        value = _g_import(arg, self.param, device)
        result = self.call(value)
        if self.result_type is not None:
            result = _g_annotate(result, self.result_type)
        return _g_export(result)


def _g_shape(x):
    return tuple(x.shape)


def _g_atom(x):
    if isinstance(x, torch.Tensor):
        return x
    if x.shape:
        raise ValueError('expected a scalar object')
    return x.values[0]


def _g_checked_shape(sh):
    sh = tuple(sh)
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in sh):
        raise ValueError(f'invalid Remora shape {sh}')
    return sh


def _g_empty(desc, device):
    atom, sh = desc() if callable(desc) else desc
    sh = _g_checked_shape(sh)
    if atom in ('Int', 'Float', 'Bool'):
        return torch.empty(sh, device=device, dtype={'Int': torch.int64, 'Float': torch.float32, 'Bool': torch.bool}[atom])
    if math.prod(sh):
        raise ValueError('cannot manufacture nonempty object values')
    return _Objects(sh, [])


def _g_reshape(x, sh):
    sh = _g_checked_shape(sh)
    if math.prod(sh) != math.prod(_g_shape(x)):
        raise ValueError('reshape changes the element count')
    return x.reshape(sh) if isinstance(x, torch.Tensor) else _Objects(sh, x.values)


def _g_frame(frame, values, desc, device):
    frame = _g_checked_shape(frame)
    if math.prod(frame) != len(values):
        raise ValueError('frame length mismatch')
    if not values:
        return _g_empty(desc, device)
    cell = _g_shape(values[0])
    if any(_g_shape(v) != cell for v in values):
        raise ValueError('Remora frame contains cells with different shapes; box them first')
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values).reshape(frame + cell)
    return _g_annotate(_Objects(frame + cell, [x for v in values for x in v.values]), desc)


def _g_cell(x, index, cell):
    cell = tuple(cell)
    count = math.prod(cell)
    if isinstance(x, torch.Tensor):
        # Explicit leading dimension handles zero-sized cells.
        n = math.prod(_g_shape(x)[:len(_g_shape(x)) - len(cell)])
        return x.reshape((n,) + cell)[index]
    return _Objects(cell, x.values[index * count:(index + 1) * count])


def _g_expand_cell(x, cell, frame, index):
    own = _g_shape(x)[:len(_g_shape(x)) - len(cell)]
    if own != frame[:len(own)] or _g_shape(x)[len(own):] != tuple(cell):
        raise ValueError(f'incompatible Remora frame/cell: {_g_shape(x)}, {cell}, {frame}')
    repeat = math.prod(frame[len(own):])
    return _g_cell(x, index // repeat, cell)


def _g_annotate(x, desc):
    if not isinstance(x, _Objects) or not any(isinstance(v, _Closure) and v.param is None and v.result_type is None for v in x.values):
        return x
    atom, _ = desc() if callable(desc) else desc
    if isinstance(atom, tuple) and atom[0] == 'function':
        return _Objects(x.shape, [_Closure(v.call, atom[1], v.kind, v.device, atom[2]) for v in x.values])
    if isinstance(atom, tuple) and atom[0] == 'poly':
        return _Objects(x.shape, [_Closure(v.call, None, atom[1], v.device, atom[2]) for v in x.values])
    return x


def _g_apply(fn, arg, cell, desc, device):
    cell = _g_checked_shape(cell)
    fshape = _g_shape(fn)
    ashape = _g_shape(arg)
    if len(ashape) < len(cell):
        raise ValueError('argument rank is smaller than its cell rank')
    aframe = ashape[:len(ashape) - len(cell)]
    frame = max((fshape, aframe), key=len)
    if fshape != frame[:len(fshape)] or aframe != frame[:len(aframe)]:
        raise ValueError(f'incompatible Remora prefixes {fshape} and {aframe}')
    if not frame:
        return _g_annotate(_g_atom(fn).call(arg), desc)
    result = []
    for i in range(math.prod(frame)):
        f = _g_atom(_g_expand_cell(fn, (), frame, i))
        x = _g_expand_cell(arg, cell, frame, i)
        result.append(f.call(x))
    out = _g_frame(frame, result, desc, device)
    if not result and isinstance(out, torch.Tensor) and out.is_floating_point() and isinstance(arg, torch.Tensor):
        out = out + arg.reshape(-1)[:0].sum().to(out.dtype)
    return out


def _g_specialize(fn, arg, desc, device):
    values = [_g_atom(_g_cell(fn, i, ())).call(arg) for i in range(math.prod(fn.shape))]
    return _g_frame(fn.shape, values, desc, device)


def _g_unbox(boxes, body, desc, device):
    values = []
    for box in boxes.values:
        if not isinstance(box, Box):
            raise TypeError('unbox expects a box')
        values.append(body(box.index, box.value))
    return _g_frame(boxes.shape, values, desc, device)


def _g_struct(frame, fields, desc, device):
    values = []
    for i in range(math.prod(frame)):
        record = {name: _g_expand_cell(x, cell, frame, i) for name, cell, x in fields}
        values.append(_Objects((), [record]))
    return _g_frame(frame, values, desc, device)


def _g_field(records, name, desc, device):
    return _g_frame(records.shape, [r[name] for r in records.values], desc, device)


def _g_tensor(value, atom, device):
    return torch.tensor(value, dtype={'Int': torch.int64, 'Float': torch.float32, 'Bool': torch.bool}[atom], device=device)


def _g_curry(fn, count, kind='value', args=(), device=None):
    if count == 0:
        return fn(*args)
    return _Objects((), [_Closure(lambda x: _g_curry(fn, count - 1, kind, args + (x,), device), kind=kind, device=device)])


def _g_call2(fn, x, y):
    return _g_atom(_g_atom(fn).call(x)).call(y)


def _g_fold(fn, arr, zero=None, right=False):
    cell = _g_shape(arr)[1:]
    if zero is None:
        acc, start = _g_cell(arr, 0, cell), 1
    else:
        acc, start = zero, 0
    indices = range(_g_shape(arr)[0] - 1, -1, -1) if right else range(start, _g_shape(arr)[0])
    for i in indices:
        v = _g_cell(arr, i, cell)
        acc = _g_call2(fn, v, acc) if right else _g_call2(fn, acc, v)
    return acc


def _g_concat(x, y):
    if isinstance(x, torch.Tensor):
        return torch.cat((x, y), dim=0)
    return _Objects((x.shape[0] + y.shape[0],) + x.shape[1:], x.values + y.values)


def _g_reverse(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, (0,))
    cell = math.prod(x.shape[1:])
    return _Objects(x.shape, [v for i in range(x.shape[0] - 1, -1, -1) for v in x.values[i*cell:(i+1)*cell]])


def _g_transpose(x):
    if isinstance(x, torch.Tensor):
        return x.transpose(0, 1)
    m, n = x.shape
    return _Objects((n, m), [x.values[i*n+j] for j in range(n) for i in range(m)])


def _g_index(x, idx):
    if isinstance(x, torch.Tensor):
        return _index(x, idx)
    i = int(idx.item())
    if i < 0 or i >= x.shape[0]:
        raise IndexError('Remora index out of bounds')
    return _g_cell(x, i, x.shape[1:])


def _g_dynamic_iota(sh, device):
    # Runtime shape values necessarily cross the device/host boundary.
    shape = _g_checked_shape(tuple(sh.detach().cpu().tolist()))
    data = torch.arange(math.prod(shape), dtype=torch.int64, device=device).reshape(shape)
    return _Objects((), [Box(shape, data)])


def _g_filename(x):
    return ''.join(chr(i) for i in x.detach().cpu().reshape(-1).tolist())


def _g_format(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            v = x.item()
            return '#t' if v is True else '#f' if v is False else repr(v)
        if x.numel() == 0:
            dtype = {torch.int64: 'Int', torch.float32: 'Float', torch.bool: 'Bool'}[x.dtype]
            return '(array [' + ' '.join(map(str, x.shape)) + '] ' + dtype + ')'
        return '[' + ' '.join(_g_format(v) for v in x) + ']'
    if x.shape:
        return '[' + ' '.join(_g_format(_g_cell(x, i, x.shape[1:])) for i in range(x.shape[0])) + ']'
    atom = x.values[0]
    if isinstance(atom, Box):
        index = '[' + ' '.join(map(str, atom.index)) + ']' if isinstance(atom.index, tuple) else str(atom.index)
        return f'(box {index} {_g_format(atom.value)})'
    if isinstance(atom, dict):
        return '{' + ' '.join(f'({k} {_g_format(v)})' for k, v in atom.items()) + '}'
    return '<function>'


def _g_trace(x, result, filename=None):
    text = _g_format(x) + '\n'
    if filename is None:
        sys.stderr.write(text)
    else:
        with open(_g_filename(filename), 'a') as f:
            f.write(text)
    return result


def _g_read(filename, atom, shape, device, binary=False):
    path = _g_filename(filename)
    if binary:
        import struct
        with open(path, 'rb') as f:
            data = f.read(4 * math.prod(shape))
        if len(data) != 4 * math.prod(shape):
            raise ValueError('read-file-f32bin: file is shorter than the declared shape')
        return _g_tensor(struct.unpack('<' + 'f' * math.prod(shape), data), 'Float', device).reshape(shape)
    # Remora read-file accepts arbitrary Remora expressions, not only literals.
    # Compile them using the same frontend/backend; never eval source as Python.
    import os
    from remora_torch import compile_source
    with open(path) as f:
        code = compile_source(expression=f.read(), remora=os.environ.get('REMORA', 'remora'))
    namespace = {'__name__': '_remora_read_file'}
    exec(compile(code, path + ':compiled', 'exec'), namespace)
    return _g_import(namespace['ENTRIES']['main'](device=device), (atom, shape), device)


def _g_undefined():
    raise RuntimeError('Remora undefined value evaluated')


def _g_intrinsic(name, device):
    if name in _G_PRIMITIVES:
        fn, arity = _G_PRIMITIVES[name]
        return _g_curry(fn, arity, device=device)
    def poly(nt, ni, nv, fn):
        return _g_curry(lambda *ts: _g_curry(lambda *ix: _g_curry(lambda *xs: fn(ts, ix, xs), nv, device=device), ni, 'index', device=device), nt, 'type', device=device)
    if name == 'head':
        return poly(1, 2, 1, lambda t, i, x: _g_cell(x[0], 0, _g_shape(x[0])[1:]))
    if name == 'tail':
        return poly(1, 2, 1, lambda t, i, x: x[0][1:] if isinstance(x[0], torch.Tensor) else _Objects((x[0].shape[0]-1,) + x[0].shape[1:], x[0].values[math.prod(x[0].shape[1:]):]))
    if name == 'length':
        return poly(1, 2, 1, lambda t, i, x: _g_tensor(_g_shape(x[0])[0], 'Int', device))
    if name == 'append':
        return poly(1, 3, 2, lambda t, i, x: _g_concat(*x))
    if name == 'reverse':
        return poly(1, 2, 1, lambda t, i, x: _g_reverse(x[0]))
    if name == 'transpose2d':
        return poly(1, 2, 1, lambda t, i, x: _g_transpose(x[0]))
    if name == 'flatten':
        return poly(1, 3, 1, lambda t, i, x: _g_reshape(x[0], (i[0] * i[1],) + i[2]))
    if name == 'reshape':
        return poly(1, 2, 1, lambda t, i, x: _g_reshape(x[0], i[1]))
    if name == 'sum':
        return poly(0, 1, 1, lambda t, i, x: x[0].sum(dtype=torch.int64))
    if name == 'reduce':
        return poly(1, 2, 2, lambda t, i, x: _g_fold(x[0], x[1]))
    if name == 'reduce/zero':
        return poly(1, 2, 3, lambda t, i, x: _g_fold(x[0], x[2], x[1]))
    if name in ('fold', 'fold-right'):
        return poly(2, 3, 3, lambda t, i, x: _g_fold(x[0], x[2], x[1], name == 'fold-right'))
    if name == 'index':
        return poly(1, 1, 2, lambda t, i, x: _g_index(*x))
    if name == 'index2d':
        return poly(1, 2, 2, lambda t, i, x: _g_index(_g_reshape(x[0], (math.prod(_g_shape(x[0])),)), _g_shape(x[0])[1] * x[1][0] + x[1][1]))
    if name == 'iota':
        return poly(0, 1, 1, lambda t, i, x: _g_dynamic_iota(x[0], device))
    if name == 'iota/static':
        return poly(0, 1, 0, lambda t, i, x: torch.arange(math.prod(_g_checked_shape(i[0])), dtype=torch.int64, device=device).reshape(i[0]))
    if name == 'reify-dim':
        return poly(0, 1, 0, lambda t, i, x: _g_tensor(i[0], 'Int', device))
    if name == 'reify-shape':
        return poly(0, 1, 0, lambda t, i, x: _Objects((), [Box(len(i[0]), _g_tensor(i[0], 'Int', device))]))
    if name == 'undefined':
        return poly(1, 1, 0, lambda t, i, x: _g_undefined())
    if name == 'trace':
        return poly(2, 2, 2, lambda t, i, x: _g_trace(*x))
    if name == 'trace-file':
        return poly(2, 3, 3, lambda t, i, x: _g_trace(x[1], x[2], x[0]))
    if name == 'read-file':
        return poly(1, 2, 1, lambda t, i, x: _g_read(x[0], t[0], i[1], device))
    if name == 'read-file-f32bin':
        return poly(0, 2, 1, lambda t, i, x: _g_read(x[0], 'Float', i[1], device, True))
    raise RuntimeError(f'unknown Remora intrinsic {name}')


def _g_export(x):
    if isinstance(x, torch.Tensor):
        return x
    def atom(v):
        if isinstance(v, dict):
            return {k: _g_export(a) for k, a in v.items()}
        if isinstance(v, Box):
            return Box(v.index, _g_export(v.value))
        return v
    if not x.shape:
        return atom(x.values[0])
    # Preserve shape metadata even for zero-sized object arrays.
    return ObjectArray(x.shape, [atom(v) for v in x.values])


@dataclass
class ObjectArray:
    shape: tuple
    values: list


def _g_device(args, device):
    tensors = []
    def visit(v):
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif hasattr(v, 'index') and hasattr(v, 'value'):
            visit(v.value)
        elif hasattr(v, 'shape') and isinstance(getattr(v, 'values', None), list):
            visit(v.values)
        elif isinstance(v, dict):
            for x in v.values(): visit(x)
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
    visit(args)
    if device is None and tensors and any(x.device != tensors[0].device for x in tensors):
        raise ValueError('entry tensors must be on the same device')
    return torch.device(device) if device is not None else tensors[0].device if tensors else torch.device('cpu')


def _g_import(value, desc, device):
    atom, sh = desc
    sh = _g_checked_shape(sh)
    if atom in ('Int', 'Float', 'Bool'):
        dtype = {'Int': torch.int64, 'Float': torch.float32, 'Bool': torch.bool}[atom]
        if isinstance(value, torch.Tensor):
            if value.dtype != dtype: raise TypeError(f'expected {dtype}, got {value.dtype}')
            out = value.to(device)
        else:
            out = torch.as_tensor(value, dtype=dtype, device=device)
        if tuple(out.shape) != sh: raise ValueError(f'expected shape {sh}, got {tuple(out.shape)}')
        return out
    if isinstance(value, _Objects):
        if value.shape != sh: raise ValueError('object shape mismatch')
        raw = value.values
    elif hasattr(value, 'shape') and isinstance(getattr(value, 'values', None), list):
        if value.shape != sh: raise ValueError('object shape mismatch')
        raw = value.values
    elif not sh:
        raw = [value]
    else:
        def flatten(v, shape):
            if not shape: return [v]
            if len(v) != shape[0]: raise ValueError('object shape mismatch')
            return [a for child in v for a in flatten(child, shape[1:])]
        raw = flatten(value, sh)
    def convert(v):
        if isinstance(atom, tuple) and atom[0] == 'record':
            fields = atom[1]
            if set(v) != set(fields): raise ValueError('record fields do not match declared type')
            return {k: _g_import(v[k], d, device) for k, d in fields.items()}
        if isinstance(atom, tuple) and atom[0] == 'box':
            if not (hasattr(v, 'index') and hasattr(v, 'value')): raise TypeError('expected Box(index, value)')
            index = tuple(v.index) if atom[1] == 'shape' else v.index
            if atom[1] == 'dim': _g_checked_shape((index,))
            else: _g_checked_shape(index)
            return Box(index, _g_import(v.value, atom[2](index), device))
        if isinstance(atom, tuple) and atom[0] == 'function':
            if isinstance(v, _Closure): return _Closure(v.call, atom[1], v.kind, v.device, atom[2])
            if not callable(v): raise TypeError('expected a callable function')
            return _Closure(lambda arg: _g_import(v(_g_export(arg)), atom[2], device), atom[1], device=device)
        if isinstance(atom, tuple) and atom[0] == 'poly':
            if not isinstance(v, _Closure): raise TypeError('expected a generated type/index function')
            return _Closure(v.call, None, atom[1], v.device, atom[2])
        if atom == 'Function':
            if not isinstance(v, _Closure): raise TypeError('expected a generated Remora function')
            return v
        raise TypeError(f'unsupported external value for {atom}')
    return _Objects(sh, [convert(v) for v in raw])


def _g_json(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Box): return {'box': x.index, 'value': _g_json(x.value)}
    if isinstance(x, ObjectArray): return {'shape': x.shape, 'values': [_g_json(v) for v in x.values]}
    if isinstance(x, dict): return {k: _g_json(v) for k, v in x.items()}
    if isinstance(x, _Closure): return {'function': x.kind}
    raise TypeError(f'cannot encode {type(x).__name__}')


def _g_cli():
    import argparse
    import json
    p = argparse.ArgumentParser(description='Compiled higher-order Remora/PyTorch program')
    p.add_argument('--entry', default='main', choices=sorted(ENTRIES))
    p.add_argument('--device', default='cpu')
    p.add_argument('args', nargs='*', help='JSON values; boxes use {"box": index, "value": data}')
    a = p.parse_args()
    def decode(v):
        if isinstance(v, dict):
            if set(v) == {'box', 'value'}: return Box(v['box'], decode(v['value']))
            if set(v) == {'shape', 'values'}: return ObjectArray(tuple(v['shape']), [decode(x) for x in v['values']])
            return {k: decode(x) for k, x in v.items()}
        return [decode(x) for x in v] if isinstance(v, list) else v
    print(json.dumps(_g_json(ENTRIES[a.entry](*(decode(json.loads(v)) for v in a.args), device=a.device))))


def _g_unbound(name):
    raise ValueError(f'cannot determine runtime index {name} without an existential witness')
