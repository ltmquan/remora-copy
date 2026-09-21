"""Guard native rewrites, source validation, parameter updates, and gradients."""
import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
import torch
from test_backend import REMORA, ROOT, module
from remora_torch import CompileError
from remora_torch.native import compile_native_ast

@unittest.skipUnless(REMORA,'set REMORA')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'validation'))
        from yolo_differential import definitions, reachable_definitions
        source=ROOT.parent/'examples/darknet/yolov4.remora'
        parsed=json.loads(subprocess.check_output([REMORA,'dev','--parse','--json','-f',str(source)]))
        body='(@conv-2d/rem _ (3 3 2 3 1 1 0) x w)'
        body_ast=json.loads(subprocess.check_output([REMORA,'dev','--parse','--json','-e',body]))
        _,forms=definitions(source.read_text())
        selected,_=reachable_definitions(forms,parsed,body_ast)
        cls.path=ROOT/'.build/tests/native-conv.remora'
        cls.path.write_text(selected+'(entry (main (x [Float 2 4 4]) (w [Float 3 2 3 3])) '+body+')')
        cls.ast=json.loads(subprocess.check_output([REMORA,'dev','--parse','--json','-f',str(cls.path)]))

    def generated(self,ast=None):
        code=compile_native_ast(ast or self.ast)
        path=ROOT/'.build/tests/native-generated.py';path.write_text(code)
        spec=importlib.util.spec_from_file_location('native_generated',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_conv_gradients_updates_and_fullgraph(self):
        m=self.generated();self.assertEqual(m.LOWERING,'native')
        x=torch.randn(2,4,4,requires_grad=True);w=torch.randn(3,2,3,3,requires_grad=True)
        ref=lambda:torch.nn.functional.conv2d(x[None],w,padding=1)[0].flatten(1)
        out=m.main(x,w);torch.testing.assert_close(out,ref())
        grad=torch.randn_like(out)
        got=torch.autograd.grad(out,(x,w),grad,retain_graph=True)
        expected=torch.autograd.grad(ref(),(x,w),grad)
        for a,b in zip(got,expected):torch.testing.assert_close(a,b)
        compiled=torch.compile(m.main,backend='aot_eager',fullgraph=True)
        torch.testing.assert_close(compiled(x,w),ref())
        with torch.no_grad():w.add_(.125)
        torch.testing.assert_close(compiled(x,w),ref())
        with self.assertRaises(ValueError):m.main(torch.zeros(2,5,5),w)
        with self.assertRaises(TypeError):m.main(x.double(),w)

    def test_redefined_dependency_rejected(self):
        ast=copy.deepcopy(self.ast)
        binding=next(d['bind'] for d in ast['progDecs'] if d['tag']=='Def' and d['bind']['var']=='dot')
        binding['expr']={'tag':'Array','dims':[],'atoms':[]}
        with self.assertRaisesRegex(CompileError,'contract changed: dot'):compile_native_ast(ast)

    def test_missing_dependency_rejected(self):
        ast=copy.deepcopy(self.ast)
        ast['progDecs']=[d for d in ast['progDecs'] if not(d['tag']=='Def' and d['bind']['var']=='dot')]
        with self.assertRaisesRegex(CompileError,'contract changed: dot'):compile_native_ast(ast)

    def test_shadowed_primitive_rejected(self):
        ast=copy.deepcopy(self.ast)
        b=copy.deepcopy(next(d['bind'] for d in ast['progDecs'] if d['tag']=='Def' and d['bind']['var']=='e'))
        b['var']='f.*';ast['progDecs'].insert(0,{'tag':'Def','bind':b})
        with self.assertRaisesRegex(CompileError,'Shadowed library dependency'):compile_native_ast(ast)

    def test_effectful_global_falls_back(self):
        text=self.path.read_text()+'\n(def (val side (@trace (Int Int) ([] []) 7 0)))\n'
        path=ROOT/'.build/tests/native-effect.remora';path.write_text(text)
        # Compile only: the side effect must remain in general execution.
        m=module(file=path);self.assertEqual(m.LOWERING,'general')
        stderr=io.StringIO()
        with contextlib.redirect_stderr(stderr):m.main(torch.zeros(2,4,4),torch.zeros(3,2,3,3))
        self.assertEqual(stderr.getvalue(),'7\n')

    def test_bad_source_type_is_not_accepted(self):
        path=ROOT/'.build/tests/native-bad.remora'
        path.write_text(self.path.read_text().replace('(w [Float 3 2 3 3])','(w [Float 4 2 3 3])'))
        with self.assertRaises(CompileError):module(file=path)

    def test_folded_bn_parameters_remain_live_and_differentiable(self):
        from yolo_differential import cases, definitions, reachable_definitions, ty
        case=next(c for c in cases() if c['name']=='conv_bn_mish')
        source=ROOT.parent/'examples/darknet/yolov4.remora'
        parsed=json.loads(subprocess.check_output([REMORA,'dev','--parse','--json','-f',str(source)]))
        expr=json.loads(subprocess.check_output([REMORA,'dev','--parse','--json','-e',case['body']]))
        _,forms=definitions(source.read_text());selected,_=reachable_definitions(forms,parsed,expr)
        path=ROOT/'.build/tests/native-bn.remora'
        path.write_text(selected+'(entry (main '+' '.join(f'({n} {ty(sh)})' for n,sh in case['args'])+') '+case['body']+')')
        m=module(file=path,lowering='native')
        torch.manual_seed(83)
        args=[(torch.rand(sh)+.5 if n.endswith('var') else torch.randn(sh)*.1).requires_grad_() for n,sh in case['args']]
        out=m.main(*args);grad=torch.autograd.grad(out.sum(),args)
        self.assertTrue(all(torch.isfinite(g).all() for g in grad))
        # Finite differences independently verify differentiation through BN folding.
        eps=1e-3;weight=args[1]
        with torch.no_grad():
            weight.flatten()[0]+=eps;plus=m.main(*args).sum()
            weight.flatten()[0]-=2*eps;minus=m.main(*args).sum()
            weight.flatten()[0]+=eps
        torch.testing.assert_close(grad[1].flatten()[0],(plus-minus)/(2*eps),atol=1e-4,rtol=1e-2)
        with torch.no_grad():args[-1].add_(.3)
        self.assertFalse(torch.allclose(out,m.main(*args)))

    def test_duplicate_entry_parameters_rejected(self):
        ast=copy.deepcopy(self.ast)
        entry=next(d for d in ast['progDecs'] if d['tag']=='Entry')
        entry['params'].append(copy.deepcopy(entry['params'][0]))
        with self.assertRaisesRegex(CompileError,'duplicate entry parameter'):compile_native_ast(ast)
