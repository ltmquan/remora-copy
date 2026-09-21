"""End-to-end tests using the actual Remora frontend, not a second parser."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from remora_torch import CompileError, compile_source
from remora_torch.compiler import dim

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REMORA = os.environ.get('REMORA') or shutil.which('remora')
BUILD = ROOT / '.build' / 'tests'
BUILD.mkdir(parents=True, exist_ok=True)


def values(text):
    """Read numeric test fixtures, preserving source spans for interpreter args."""
    tokens = list(re.finditer(r'\[|\]|\(|\)|[^\s\[\]()]+', text))
    i = 0
    def one():
        nonlocal i
        t = tokens[i].group(); i += 1
        if t == '[':
            out = []
            while tokens[i].group() != ']':
                out.append(one())
            i += 1
            return out
        if t == '(':
            assert tokens[i].group() == 'array'
            i += 1
            shape = one(); data = []
            while tokens[i].group() != ')':
                data.append(one())
            i += 1
            return torch.tensor(data).reshape(shape).tolist()
        if t in ('#t', '#f'):
            return t == '#t'
        return float(t) if any(c in t.lower() for c in ('.', 'e')) else int(t)
    out = []
    while i < len(tokens):
        start = tokens[i].start()
        v = one()
        out.append((v, text[start:tokens[i-1].end()]))
    return out


def module(*, file=None, expression=None, lowering="auto"):
    code = compile_source(source=file, expression=expression, remora=REMORA, lowering=lowering)
    # Real files let Dynamo inspect and report generated source locations.
    with tempfile.NamedTemporaryFile('w', suffix='.py', dir=BUILD, delete=False) as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location('remora_generated_' + Path(f.name).stem, f.name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@unittest.skipUnless(REMORA, 'set REMORA to the built Remora binary')
class BackendTests(unittest.TestCase):
    def test_prefix_broadcast_and_gradients(self):
        m = module(file=ROOT / 'examples/prefix.remora')
        rows = torch.tensor([10., 20.], requires_grad=True)
        matrix = torch.arange(6., requires_grad=True).reshape(2, 3)
        out = m.main(rows, matrix)
        torch.testing.assert_close(out, rows[:, None] + matrix)
        dr, dm = torch.autograd.grad(out.sum(), (rows, matrix))
        torch.testing.assert_close(dr, torch.full((2,), 3.))
        torch.testing.assert_close(dm, torch.ones(2, 3))

    def test_matmul_eager_compiled_and_gradients(self):
        m = module(file=ROOT / 'examples/matmul.remora')
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(3, 4, requires_grad=True)
        expected = a @ b
        backends = ['aot_eager'] + (['inductor'] if os.environ.get('REMORA_TORCH_TEST_INDUCTOR') else [])
        for fn in [m.main] + [torch.compile(m.main, backend=b, fullgraph=True) for b in backends]:
            out = fn(a, b)
            torch.testing.assert_close(out, expected)
            actual_grads = torch.autograd.grad(out.sum(), (a, b), retain_graph=True)
            expected_grads = torch.autograd.grad(expected.sum(), (a, b), retain_graph=True)
            for actual, reference in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(fn(a + 1, b - 1), (a + 1) @ (b - 1))

    def test_integer_and_float_semantics(self):
        cases = {'(/ -7 3)': -3, '(mod -7 3)': 2, '(round 2.5)': 2,
                 '(round 3.5)': 4, '(popc -1)': 64, '(+ 1 2)': 3, '(shl 8 -1)': 4, '(shr 8 -1)': 16,
                 '(f.+ 16777216.0 1.0)': 16777216.0}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                out = module(expression=expression).main()
                self.assertEqual(out.item(), expected)
                self.assertEqual(out.dtype, torch.float32 if expression.startswith('(f.') else torch.int64)

    def test_empty_frames_and_cells(self):
        for expression, expected_shape in [
            ('(+ (array [0] Int) 1)', (0,)),
            ('(+ (array [2 0] Int) 1)', (2, 0)),
            ('(@reduce/zero (Int) (0 []) + 7 (array [0] Int))', ()),
            ('(@head (Int) (1 [0]) (array [2 0] Int))', (0,))]:
            with self.subTest(expression=expression):
                fn = module(expression=expression).main
                out = fn()
                self.assertEqual(tuple(out.shape), expected_shape)
                if not expected_shape:
                    self.assertEqual(out.item(), 7)
                compiled = torch.compile(fn, backend='aot_eager', fullgraph=True)
                torch.testing.assert_close(compiled(), out)

    def test_empty_map_gradient(self):
        path = BUILD / 'empty-gradient.remora'
        path.write_text('(entry (main (x [Float 0]) (y Float)) (f.+ x y))')
        m = module(file=path)
        x = torch.empty(0, requires_grad=True)
        y = torch.tensor(3., requires_grad=True)
        dx, dy = torch.autograd.grad(m.main(x, y).sum(), (x, y))
        self.assertEqual(tuple(dx.shape), (0,))
        self.assertEqual(dy.item(), 0.)

    def test_ordered_folds(self):
        for expression, result in [
            ('(@reduce (Int) (3 []) - [1 2 3 4])', -8),
            ('(@fold (Int Int) (3 [] []) - 100 [1 2 3 4])', 90),
            ('(@fold-right (Int Int) (3 [] []) - 100 [1 2 3 4])', 98)]:
            self.assertEqual(module(expression=expression).main().item(), result)

    def test_index_lifting_and_bounds(self):
        expression = '(@index (Int) (3) [10 20 30] [2 0 1])'
        torch.testing.assert_close(module(expression=expression).main(), torch.tensor([30, 10, 20]))
        for expression in ('(@index (Int) (3) [1 2 3] -1)',
                           '(@index2d (Int) (2 2) [[1 2] [3 4]] [1 2])'):
            with self.assertRaises((RuntimeError, IndexError)):
                module(expression=expression).main()

    def test_record_arrays(self):
        m = module(file=REPO / 'tests/rec7.remora')
        torch.testing.assert_close(m.main([1., 2.], [10., 20.], [3.], [30.]), torch.tensor([1., 2., 3.]))
        torch.testing.assert_close(m.lifted([1., 2.], 9.), torch.tensor([9., 9.]))
        torch.testing.assert_close(torch.compile(m.main, backend='aot_eager', fullgraph=True)(
            [1., 2.], [10., 20.], [3.], [30.]), torch.tensor([1., 2., 3.]))

    def test_nested_record_fields(self):
        m = module(expression='(field (field {(outer [] {(inner [] [1 2])})} outer) inner)')
        torch.testing.assert_close(m.main(), torch.tensor([1, 2]))

    def test_convolution_from_existing_library(self):
        for name, image, weights, stride in (
            ('conv_remainder', torch.arange(25.).reshape(1, 1, 5, 5), torch.ones(1, 1, 2, 2), 2),
            ('conv', torch.arange(16.).reshape(1, 1, 4, 4),
             torch.stack([torch.ones(1, 2, 2), torch.full((1, 2, 2), .5)]), 2)):
            with self.subTest(name=name):
                m = module(file=REPO / 'examples/darknet' / (name + '.remora'))
                reference = torch.nn.functional.conv2d(image, weights, stride=stride)
                expected = reference.reshape(weights.shape[0], -1)
                torch.testing.assert_close(m.main(), expected)
                torch.testing.assert_close(torch.compile(m.main, backend='aot_eager', fullgraph=True)(), expected)

    def test_entry_name_does_not_shadow_runtime(self):
        path = BUILD / 'names.remora'
        path.write_text('(entry (len (x [Int 2])) (+ x 1))')
        m = module(file=path)
        torch.testing.assert_close(m.ENTRIES['len']([1, 2]), torch.tensor([2, 3]))

    def test_entry_validation(self):
        m = module(file=ROOT / 'examples/prefix.remora')
        with self.assertRaises(ValueError):
            m.main([1., 2., 3.], [[1., 2., 3.], [4., 5., 6.]])
        with self.assertRaises(TypeError):
            m.main(torch.tensor([1, 2]), torch.ones(2, 3))
        with self.assertRaises(TypeError):
            m.main()

    def test_rejections_and_atomic_output(self):
        for file in ('box0.remora', 'mono6.remora', 'typecheck0.remora'):
            with self.subTest(file=file), self.assertRaises(CompileError):
                module(file=REPO / 'tests' / file, lowering='static')
        target = BUILD / 'existing.py'
        target.write_text('keep me\n')
        p = subprocess.run([sys.executable, '-m', 'remora_torch', '--remora', REMORA,
                            '-e', '(+ #t 1)', '-o', str(target)], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('frontend failed', p.stderr)
        self.assertEqual(target.read_text(), 'keep me\n')

    def test_standalone_cli(self):
        target = BUILD / 'standalone.py'
        subprocess.run([sys.executable, '-m', 'remora_torch', '--remora', REMORA,
                        '-f', str(REPO / 'tests/entry0.remora'), '-o', str(target)], check=True)
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        result = subprocess.check_output([sys.executable, str(target), '--entry', 'sub', '10', '3'],
                                         cwd=BUILD, env=env, text=True)
        self.assertEqual(json.loads(result), 7)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda(self):
        m = module(file=ROOT / 'examples/matmul.remora')
        a, b = torch.randn(2, 3, device='cuda'), torch.randn(3, 4, device='cuda')
        for fn in (m.main, torch.compile(m.main, fullgraph=True)):
            torch.testing.assert_close(fn(a, b), a @ b)
        self.assertEqual(m.main(a.tolist(), b.tolist(), device='cuda').device.type, 'cuda')


class ShapeTests(unittest.TestCase):
    def test_dimension_subtraction(self):
        self.assertEqual(dim({'tag': 'Sub', 'dims': []}), 0)
        self.assertEqual(dim({'tag': 'Sub', 'dims': [{'tag': 'DimN', 'val': 3}]}), -3)


# Compile every positive tensor test that has inline expectations, and compare
# to BOTH the checked-in expectation and the actual Remora interpreter.
UNSUPPORTED = {'box1'}  # Nested box output is checked structurally in test_general.


@unittest.skipUnless(REMORA, 'set REMORA to the built Remora binary')
class CorpusTests(unittest.TestCase):
    pass


def corpus_test(path, lowering="auto"):
    def test(self):
        m = module(file=path, lowering=lowering)
        entry, inputs = 'main', []
        count = 0
        for line in path.read_text().splitlines():
            if line.startswith(';; =='):
                entry, inputs = 'main', []
            match = re.match(r';; entry:\s*(\S+)', line)
            if match:
                entry = match[1]
            match = re.match(r';; input \{(.*)\}', line)
            if match:
                inputs = values(match[1])
            match = re.match(r';; output \{(.*)\}', line)
            if match and not any('input @' in x for x in path.read_text().splitlines()):
                expected = values(match[1])[0][0]
                actual = m.ENTRIES[entry](*(v for v, _ in inputs))
                reference = subprocess.check_output([REMORA, 'interpret', '-f', str(path),
                    '--entry', entry, '--'] + [raw for _, raw in inputs], text=True)
                reference = values(reference)[0][0]
                with self.subTest(entry=entry, inputs=inputs):
                    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual, torch.as_tensor(reference, dtype=actual.dtype), rtol=1e-5, atol=1e-6)
                count += 1
        self.assertGreater(count, 0)
    return test


for path in sorted((REPO / 'tests').glob('*.remora')):
    if path.stem not in UNSUPPORTED and ';; output {' in path.read_text() and 'input @' not in path.read_text():
        setattr(CorpusTests, 'test_' + path.stem.replace('-', '_'), corpus_test(path))


if __name__ == '__main__':
    unittest.main()
