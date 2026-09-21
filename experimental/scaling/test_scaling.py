import unittest
from prepare import resize, WORKSPACE
from partition import assignments

class ScalingTests(unittest.TestCase):
    def test_coverage(self):
        for n in (1,3,32,101):
            for workers in (1,2,4,8):
                for policy in ('uniform','round-robin','weighted'):
                    parts=assignments(n,workers,policy,list(range(1,workers+1)))
                    self.assertEqual(sorted(i for part in parts for i in part),list(range(n)))
                    if policy=='uniform': self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
    def test_weighted(self):
        self.assertEqual(list(map(len,assignments(40,2,'weighted',[1,3]))),[10,30])
        with self.assertRaises(ValueError): assignments(4,2,'weighted',[1,0])
    def test_resizing(self):
        for repo in ('remora','remora-copy'):
            source=(WORKSPACE/repo/'examples/darknet/yolov4.remora').read_text()
            self.assertEqual(resize(source,608),source)
            changed=resize(source,960)
            self.assertIn('(image [Float 3 960 960])',changed)
            self.assertIn('[Float 56700 85]',changed)
            self.assertIn('[36.0 76.0 72.0]',changed)
            self.assertIn('(w-152 [Float 512 256 3 3])',changed)
            self.assertEqual(source.split('(entry (main')[0],changed.split('(entry (main')[0])

class ArtifactTests(unittest.TestCase):
    def test_every_size_topology_and_frontend(self):
        import json, sys, importlib.util
        from prepare import ROOT, BACKEND
        sys.path.insert(0,str(BACKEND))
        from test_topology import Node, Trace
        from direct import YOLOv4
        for size in (608,960,1216,1536):
            directory=ROOT/'artifacts'/str(size)
            for variant in ('original','modified'):
                self.assertTrue(json.loads((directory/variant/'source.json').read_text())['frontend_validated'])
            graph=json.loads((directory/'original/generated_yolov4.json').read_text())
            self.assertEqual(graph['inputs']['image'],[3,size,size])
            inputs={name:Node('input',name) for name in graph['inputs']}
            spec=importlib.util.spec_from_file_location('scaled_test',directory/'original/generated_yolov4.py')
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            self.assertEqual(mod.forward(Trace(),inputs).key,YOLOv4(directory/'yolov4.cfg')(Trace(),inputs).key)

    def test_matrix_dimensions(self):
        import subprocess,sys,json,tempfile
        from prepare import ROOT
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable,str(ROOT/'matrix.py'),'--weights','test.weights','--out',out],check=True,capture_output=True)
            plan=json.loads((__import__('pathlib').Path(out)/'plan.json').read_text())
            cells=plan['cells']; names=[c['name'] for c in cells]
            self.assertEqual(len(names),len(set(names)))
            self.assertEqual(len(names),884)
            for size in (608,960,1216,1536):
                self.assertTrue(any(f'-{size}-2gpu-' in n for n in names))
            self.assertFalse(any('darknet-dense' in n and 'resident' in n for n in names))
            for count in (1,2,4,8): self.assertTrue(any(f'-{count}gpu-' in n for n in names))
            for cell in cells:
                command=cell['command']; start=command.index('--gpus')+1; end=command.index('--policy')
                self.assertEqual(len(command[start:end]),int(cell['name'].split('gpu-')[0].split('-')[-1]))

class ReportingTests(unittest.TestCase):
    def test_memory_pass_cannot_supply_scaling_baseline(self):
        import csv,json,subprocess,sys,tempfile
        from pathlib import Path
        from prepare import ROOT
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            def record(name,size,gpus,rate,kind):
                config={'implementation':'direct-torch','size':size,'gpus':list(range(gpus)),'policy':'uniform','mode':'resident','images':32}
                (directory/(name+'.json')).write_text(json.dumps({'config':config,'status':'passed','measurement_kind':kind,'median_images_per_second':rate,'trials':[]}))
            record('single',608,1,100,'timing');record('dual',608,2,180,'timing')
            record('large',1216,2,45,'timing');record('memory',608,1,500,'memory')
            subprocess.run([sys.executable,str(ROOT/'summarize.py'),str(directory),'--output',str(directory/'table.csv')],check=True,capture_output=True)
            with (directory/'table.csv').open() as stream: rows=list(csv.DictReader(stream))
            dual=next(row for row in rows if row['image_side']=='608' and row['gpus']=='2')
            large=next(row for row in rows if row['image_side']=='1216')
            self.assertEqual(float(dual['gpu_speedup_vs_1']),1.8)
            self.assertEqual(float(large['throughput_ratio_vs_608']),.25)
            self.assertEqual(float(large['pixel_count_ratio_vs_608']),4)
            self.assertEqual(next(row for row in rows if row['kind']=='memory')['images_per_second'],'')

if __name__=='__main__': unittest.main()
