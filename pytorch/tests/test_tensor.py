"""Structural rewrites: renamed programs, live inputs, lifting and non-matches."""
import unittest
import sys
import torch
from test_backend import REMORA, ROOT, module
sys.path.insert(0, str(ROOT / "validation"))

@unittest.skipUnless(REMORA,'set REMORA')
class TensorTests(unittest.TestCase):
    def program(self,text,lowering='tensor'):
        path=ROOT/'.build/tests/structural.remora';path.write_text(text)
        return module(file=path,lowering=lowering)

    def test_renamed_gemm_gradients_and_fullgraph(self):
        source=(ROOT/'evaluation/programs/probe.remora').read_text()
        for name in ['pair-product','unrelated-operation']:
            m=self.program(source.replace('pair-product',name))
            self.assertIn('matmul',m.TENSOR_OPS)
            a=torch.randn(3,4,requires_grad=True);b=torch.randn(2,4,requires_grad=True)
            for fn in [m.main,torch.compile(m.main,backend='aot_eager',fullgraph=True)]:
                out=fn(a,b);expected=a@b.T;torch.testing.assert_close(out,expected)
                actual_grads=torch.autograd.grad(out.sum(),(a,b),retain_graph=True)
                expected_grads=torch.autograd.grad(expected.sum(),(a,b))
                for x,y in zip(actual_grads,expected_grads):torch.testing.assert_close(x,y)
                torch.testing.assert_close(fn(a+1,b-1),(a+1)@(b-1).T)

    def test_modified_dot_not_gemm(self):
        source=(ROOT/'evaluation/programs/probe.remora').read_text().replace('(f.* left right)','(f.- left right)')
        m=self.program(source);self.assertNotIn('matmul',m.TENSOR_OPS)
        a=torch.randn(3,4);b=torch.randn(2,4)
        torch.testing.assert_close(m.main(a,b),(a[:,None,:]-b).sum(-1))

    def test_prefix_dot(self):
        text='''(def (fun (d (a [Float 4]) (b [Float 4])) (@reduce/zero (Float) (4 []) f.+ 0.0 (f.* a b))))
        (entry (main (a [Float 2 4]) (b [Float 2 3 4])) (d a b))'''
        m=self.program(text);a=torch.randn(2,4);b=torch.randn(2,3,4)
        torch.testing.assert_close(m.main(a,b),(a[:,None,:]*b).sum(-1))

    def test_reduction_cells_lifted_initial_and_empty(self):
        expressions=[
          '(@reduce/zero (Int) (3 [2]) (fn ((x [Int 2]) (y [Int 2])) (+ x y)) [10 20] [[1 2] [3 4] [5 6]])',
          '(@reduce/zero (Int) (3 []) + [10 20] [1 2 3])',
          '(@reduce/zero (Int) (0 []) + 7 (array [0] Int))',
          '(@reduce (Int) (2 []) + [[1 2 3] [4 5 6]])',
        ]
        for e in expressions:
            with self.subTest(e=e):
                actual=module(expression=e,lowering='tensor').main()
                expected=module(expression=e,lowering='static').main()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_subtraction_and_fold_remain_ordered(self):
        for e in ['(@reduce (Float) (2 []) f.- [1.0 2.0 4.0])',
                  '(@fold (Float Float) (2 [] []) f.+ 0.0 [16777216.0 1.0 -16777216.0])']:
            with self.subTest(e=e):
                m=module(expression=e,lowering='tensor');self.assertEqual(m.TENSOR_OPS,[])
                torch.testing.assert_close(m.main(),module(expression=e,lowering='static').main(),rtol=0,atol=0)

    def test_flat_index2d_interpreter_compatibility(self):
        import subprocess
        e='(@index2d (Int) (2 2) [[1 2] [3 4]] [0 2])'
        expected=int(subprocess.check_output([REMORA,'interpret','-e',e],text=True))
        for mode in ['static','general','tensor']:
            self.assertEqual(module(expression=e,lowering=mode).main().item(),expected)

    def test_shadowed_add_is_not_a_sum(self):
        source='''(def (fun (f.+ (a Float) (b Float)) (f.- a b)))
        (entry (main (x [Float 3])) (@reduce/zero (Float) (3 []) f.+ 0.0 x))'''
        m=self.program(source);self.assertEqual(m.TENSOR_OPS,[])
        self.assertEqual(m.main(torch.tensor([1.,2.,4.])).item(),-7.)

    def test_routing_arc_cost_against_interpreter(self):
        import json, subprocess
        path=ROOT/'evaluation/programs/route_cost.remora'
        m=module(file=path);self.assertIn('sum',m.TENSOR_OPS)
        costs=torch.tensor([[0,5,8,2],[5,0,3,7],[8,3,0,4],[2,7,4,0]])
        checks=[]
        def literal(x):
            if x.ndim:return '['+' '.join(literal(v) for v in x)+']'
            return str(x.item())
        for arcs in [torch.tensor([[0,1],[1,2],[2,0]]),torch.tensor([[0,3],[3,2],[2,0]]),torch.zeros(3,2,dtype=torch.int64)]:
            expected=costs[arcs[:,0],arcs[:,1]].sum().item()
            interpreted=int(subprocess.check_output([REMORA,'interpret','-f',str(path),'--',literal(costs),literal(arcs)],text=True))
            actual=m.main(costs,arcs).item()
            self.assertEqual(actual,expected);self.assertEqual(actual,interpreted)
            checks.append({'arcs':arcs.tolist(),'result':actual,'interpreter':interpreted,'direct':expected})
        (ROOT/'evaluation/route-results.json').write_text(json.dumps(checks,indent=2)+'\n')

    def test_neighbor_mean_against_interpreter(self):
        import json, subprocess
        from yolo_differential import literal, parse_result
        path=ROOT/'evaluation/programs/neighbor_mean.remora';m=module(file=path)
        checks=[]
        for seed in range(3):
            torch.manual_seed(seed);x=torch.randn(4,3)
            idx=torch.tensor([[0,1],[2,3]]) if seed==0 else torch.tensor([[0,0],[3,1]])
            raw_idx='['+' '.join('['+' '.join(str(v.item()) for v in row)+']' for row in idx)+']'
            raw=subprocess.check_output([REMORA,'interpret','-f',str(path),'--',literal(x.numpy()),raw_idx],text=True)
            expected=x[idx].mean(1);out=m.main(x,idx)
            torch.testing.assert_close(out,expected)
            torch.testing.assert_close(out,torch.tensor(parse_result(raw)))
            checks.append({'seed':seed,'lowering':m.LOWERING,'interpreter_and_direct_agreement':True})
        (ROOT/'evaluation/neighbor-results.json').write_text(json.dumps(checks,indent=2)+'\n')
