"""Higher-order/dynamic tests plus the entire numeric corpus on general lowering."""
import contextlib
import io
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from test_backend import module, corpus_test, REMORA, REPO, ROOT, BUILD, UNSUPPORTED
from remora_torch import CompileError


@unittest.skipUnless(REMORA, 'set REMORA to the built Remora binary')
class GeneralCorpusTests(unittest.TestCase):
    pass


for path in sorted((REPO / 'tests').glob('*.remora')):
    if path.stem not in UNSUPPORTED and ';; output {' in path.read_text() and 'input @' not in path.read_text():
        setattr(GeneralCorpusTests, 'test_' + path.stem.replace('-', '_'), corpus_test(path, lowering='general'))


@unittest.skipUnless(REMORA, 'set REMORA to the built Remora binary')
class GeneralTests(unittest.TestCase):
    def source(self, text):
        path = BUILD / 'general-program.remora'
        path.write_text(text)
        return module(file=path, lowering='general')

    def expr(self, text):
        return module(expression=text, lowering='general')

    def test_nested_box(self):
        m = module(file=REPO / 'tests/box1.remora')
        result = m.main()
        self.assertEqual(result.index, 1)
        self.assertEqual(result.value.index, 2)
        self.assertEqual(result.value.value.item(), 100)
        self.assertEqual(m.LOWERING, 'general')

    def test_runtime_shapes_multiple_calls(self):
        m = self.source('''(entry (main (n Int))
          (unbox (@s xs (@iota _ (1) [n])) (@sum _ (@s) xs)))''')
        for n in (0, 1, 8, 3):
            self.assertEqual(m.main(n).item(), n * (n - 1) // 2)
        with self.assertRaises(ValueError): m.main(-1)

    def test_shape_changing_box_result(self):
        m = self.source('(entry (main (n Int)) (@iota _ (1) [n]))')
        for n in (0, 4, 2):
            box = m.main(n)
            self.assertEqual(box.index, (n,))
            torch.testing.assert_close(box.value, torch.arange(n))

    def test_box_entry_and_gradient(self):
        m = self.source('''(entry (main (b (Sigma ($n) [Float $n])))
          (unbox ($n xs b) (f.* xs 2.0)))''')
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        out = m.main(m.Box(3, x))
        torch.testing.assert_close(out, x * 2)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), x)[0], torch.full((3,), 2.))
        with self.assertRaises(ValueError): m.main(m.Box(2, x))

    def test_records_at_entry(self):
        m = self.source('''(entry (main (p {(x [Float 2]) (y [Float])}))
          (f.+ (field p x) (field p y)))''')
        x = torch.tensor([1., 2.], requires_grad=True)
        y = torch.tensor(3., requires_grad=True)
        out = m.main({'x': x, 'y': y})
        torch.testing.assert_close(out, x + y)
        dx, dy = torch.autograd.grad(out.sum(), (x, y))
        torch.testing.assert_close(dx, torch.ones(2))
        self.assertEqual(dy.item(), 2.)
        with self.assertRaises(ValueError): m.main({'x': x})

    def test_function_arrays_and_closure_captures(self):
        m = self.expr('''(let ((val k 10)
                       (val fs [(fn ((x Int)) (+ x k))
                                (fn ((x Int)) (* x k))]))
                          (fs [[1 2 3] [4 5 6]]))''')
        torch.testing.assert_close(m.main(), torch.tensor([[11, 12, 13], [40, 50, 60]]))

    def test_function_arrays_through_array_intrinsics(self):
        expression = '''((@reverse ((-> Int Int)) (2 [])
             [(fn ((x Int)) (+ x 1)) (fn ((x Int)) (* x 2))]) [10 20])'''
        torch.testing.assert_close(self.expr(expression).main(), torch.tensor([20, 21]))

    def test_returned_closure(self):
        m = self.source('(entry (main (k Float)) (fn ((x Float)) (f.* x k)))')
        k = torch.tensor(3., requires_grad=True)
        fn = m.main(k)
        x = torch.tensor(4., requires_grad=True)
        out = fn(x)
        self.assertEqual(out.item(), 12.)
        dk, dx = torch.autograd.grad(out, (k, x))
        self.assertEqual(dk.item(), 4.); self.assertEqual(dx.item(), 3.)

    def test_returned_primitive_and_function_entry(self):
        primitive = self.expr('(+ 10)').main()
        self.assertEqual(primitive(5).item(), 15)
        self.assertEqual(self.expr('+').main()(2)(3).item(), 5)
        self.assertEqual(self.expr('head').main()('Int')(2)(())([10, 20, 30]).item(), 10)
        m = self.source('(entry (main (f (-> Int Int)) (x Int)) (f x))')
        self.assertEqual(m.main(lambda x: x + 7, 5).item(), 12)
        self.assertEqual(m.main(primitive, 8).item(), 18)

    def test_empty_function_arrays(self):
        m = self.expr('((array [0] (-> Int Int)) 3)')
        out = m.main()
        self.assertEqual(tuple(out.shape), (0,))
        self.assertEqual(out.dtype, torch.int64)

    def test_polymorphic_function_result(self):
        m = self.expr('(t-fn (&t) (fn ((x &t)) x))')
        identity = m.main()('Float')
        x = torch.tensor(5., requires_grad=True)
        out = identity(x)
        self.assertEqual(out.item(), 5.)
        self.assertEqual(torch.autograd.grad(out, x)[0].item(), 1.)

    def test_polymorphic_dynamic_closures(self):
        for file, expected in [('mono8', 10), ('mono9', 12), ('mono10', 20)]:
            m = module(file=REPO / 'tests' / (file + '.remora'))
            self.assertEqual(m.LOWERING, 'general')
            self.assertEqual(m.main().item(), expected)

    def test_effects_and_binary_read(self):
        path = BUILD / 'floats.bin'
        path.write_bytes(struct.pack('<3f', 1.5, 2.5, 3.5))
        chars = '[' + ' '.join(str(ord(c)) for c in str(path)) + ']'
        expr = f'(@read-file-f32bin _ ({len(str(path))} [3]) {chars})'
        torch.testing.assert_close(self.expr(expr).main(), torch.tensor([1.5, 2.5, 3.5]))
        path.write_bytes(b'bad')
        with self.assertRaises(ValueError): self.expr(expr).main()
        path = BUILD / 'trace.txt'; path.write_text('')
        chars = '[' + ' '.join(str(ord(c)) for c in str(path)) + ']'
        expression = f'(@trace-file (Int Int) ({len(str(path))} [2] []) {chars} [1 2] 9)'
        self.assertEqual(self.expr(expression).main().item(), 9)
        self.assertEqual(path.read_text(), '[1 2]\n')
        text = io.StringIO()
        with contextlib.redirect_stderr(text):
            self.assertEqual(self.expr('(@trace (Int Int) ([2] []) [3 4] 8)').main().item(), 8)
        self.assertEqual(text.getvalue(), '[3 4]\n')

    def test_unused_index_alias_is_not_evaluated(self):
        self.assertEqual(self.expr('(let ((ispace $d (- 3))) 7)').main().item(), 7)

    def test_literal_global_retains_tensor_lowering(self):
        path = BUILD / 'literal-global.remora'
        path.write_text('(def (val k 2.0)) (entry (main (x [Float 3])) (f.* x k))')
        m = module(file=path)
        self.assertEqual(m.LOWERING, 'static')
        torch.testing.assert_close(m.main([1., -2., 3.]), torch.tensor([2., -4., 6.]))

    def test_global_effect_is_not_pruned(self):
        path = BUILD / 'global-effect.remora'
        path.write_text('(def (val ignored (@trace (Int Int) ([] []) 42 0))) (entry (main) 7)')
        m = module(file=path)
        self.assertEqual(m.LOWERING, 'general')
        text = io.StringIO()
        with contextlib.redirect_stderr(text): self.assertEqual(m.main().item(), 7)
        self.assertEqual(text.getvalue(), '42\n')

    def test_auto_function_result(self):
        m = module(expression='(+ 10)')
        self.assertEqual(m.LOWERING, 'general')
        self.assertEqual(m.main()(3).item(), 13)

    def test_read_file_expression(self):
        path = BUILD / 'expression.txt'; path.write_text('(+ [1 2 3] 10)')
        chars = '[' + ' '.join(str(ord(c)) for c in str(path)) + ']'
        with patch.dict(os.environ, {'REMORA': REMORA}):
            out = self.expr(f'(@read-file (Int) ({len(str(path))} [3]) {chars})').main()
        torch.testing.assert_close(out, torch.tensor([11, 12, 13]))

    def test_box_values_cross_generated_modules(self):
        producer = self.source('(entry (main (n Int)) (@iota _ (1) [n]))')
        consumer = self.source('(entry (main (b (Sigma (@s) [Int @s]))) (unbox (@s x b) (@sum _ (@s) x)))')
        self.assertEqual(consumer.main(producer.main(5)).item(), 10)
        path = BUILD / 'boxed-expression.txt'; path.write_text('(@iota _ (1) [4])')
        chars = '[' + ' '.join(str(ord(c)) for c in str(path)) + ']'
        expression = f'(@read-file ((Sigma (@s) [Int @s])) ({len(str(path))} []) {chars})'
        with patch.dict(os.environ, {'REMORA': REMORA}):
            result = self.expr(expression).main()
        self.assertEqual(result.index, (4,))
        torch.testing.assert_close(result.value, torch.arange(4))

    def test_undefined_evaluation(self):
        m = self.expr('(@undefined (Int) ([2]))')
        with self.assertRaisesRegex(RuntimeError, 'undefined'): m.main()
        self.assertEqual(self.expr('(let ((fun (unused (x Int)) (@undefined (Int) ([])))) 7)').main().item(), 7)

    def test_general_empty_maps(self):
        for text, expected in [('(+ (array [0] Int) 1)', (0,)),
                               ('(@head (Int) (1 [0]) (array [2 0] Int))', (0,))]:
            self.assertEqual(tuple(self.expr(text).main().shape), expected)

    def test_complete_repository_inventory(self):
        rejected = {'issue34-flatten0', 'issue34-flatten1', 'issue34-reverse0',
                    'issue35-emptyarray0', 'issue35-emptyarray1', 'typecheck0'}
        compiled = 0
        for path in sorted((REPO / 'tests').glob('*.remora')):
            with self.subTest(file=path.name):
                if path.stem in rejected:
                    with self.assertRaises(CompileError): module(file=path, lowering='general')
                    continue
                m = module(file=path, lowering='general')
                compiled += 1
                main = m.ENTRIES.get('main')
                if main is not None and not main.__code__.co_argcount:
                    main()
        self.assertEqual(compiled, 85)

    def test_frontend_intrinsic_catalog(self):
        m = self.expr('1')
        intrinsics = re.findall(r'\(\s*"([^"]+)",', (REPO / 'src/Intrinsics.hs').read_text())
        primitives = re.findall(r'pretty \w+ = "([^"]+)"', (REPO / 'src/Primitive.hs').read_text())
        self.assertGreater(len(intrinsics), 20)
        self.assertGreater(len(primitives), 40)
        for name in intrinsics + primitives:
            with self.subTest(name=name):
                m._g_intrinsic(name, torch.device('cpu'))

    def test_library_compiles_without_entry(self):
        m = module(file=REPO / 'tests/dotlib.remora')
        self.assertEqual(m.ENTRIES, {})

    def test_file_input_fixture_programs(self):
        self.assertEqual(module(file=REPO / 'tests/infile0.remora', lowering='general').main(2, 3).item(), 5)
        torch.testing.assert_close(module(file=REPO / 'tests/infile1.remora', lowering='general').main([[1, 2], [3, 4]]), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(module(file=REPO / 'tests/infile2.remora', lowering='general').main([1., 2., 3.], [4., 5., 6.]).item(), 32.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_dynamic_cuda(self):
        m = self.source('(entry (main (n Int)) (unbox (@s x (@iota _ (1) [n])) (@sum _ (@s) x)))')
        out = m.main(torch.tensor(5, device='cuda'))
        self.assertEqual(out.device.type, 'cuda')
        self.assertEqual(out.item(), 10)
        closure = self.source('(entry (main (k Float)) (fn ((x Float)) (f.* x k)))').main(torch.tensor(3., device='cuda'))
        self.assertEqual(closure(4.).device.type, 'cuda')
        self.assertEqual(closure(4.).item(), 12.)
