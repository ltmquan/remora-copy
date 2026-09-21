import copy
import json
from pathlib import Path
import unittest
from compiler import Compiler, generate

ROOT=Path(__file__).resolve().parent

class CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ast=json.loads((ROOT/'yolov4-parse.json').read_text())
        cls.contracts=json.loads((ROOT/'contracts.json').read_text())

    def test_full_graph(self):
        graph=Compiler(self.ast,self.contracts).compile()
        self.assertEqual(sum(n['op']=='conv' for n in graph['nodes']),110)
        self.assertEqual(sum(n['op']=='decode' for n in graph['nodes']),3)
        self.assertEqual(generate(graph),(ROOT/'generated_yolov4.py').read_text())

    def test_redefinition_rejected(self):
        ast=copy.deepcopy(self.ast)
        next(d['bind'] for d in ast['progDecs'] if d.get('bind',{}).get('var')=='mish')['expr']={}
        with self.assertRaisesRegex(ValueError,'Library contract changed: mish'):
            Compiler(ast,self.contracts)

    def test_entry_controls_output(self):
        ast=copy.deepcopy(self.ast)
        entry=next(d for d in ast['progDecs'] if d['tag']=='Entry')
        entry['expr']={'tag':'Var','name':'image'}
        graph=Compiler(ast,self.contracts).compile()
        self.assertEqual(graph['nodes'],[])
        self.assertEqual(graph['output'],{'input':'image'})

    def test_builtin_shadowing_rejected(self):
        ast=copy.deepcopy(self.ast)
        ast['progDecs'].append({'tag':'Def','bind':{'tag':'BindVal','var':'append','expr':{}}})
        with self.assertRaisesRegex(ValueError,'Shadowing builtin append'):
            Compiler(ast,self.contracts)

    def test_unknown_expression_rejected(self):
        ast=copy.deepcopy(self.ast)
        next(d for d in ast['progDecs'] if d['tag']=='Entry')['expr']={'tag':'Unsupported'}
        with self.assertRaisesRegex(ValueError,'Unsupported expression'):
            Compiler(ast,self.contracts).compile()

if __name__=='__main__': unittest.main()
