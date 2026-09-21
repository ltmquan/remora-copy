"""Run the real Remora frontend and emit Python without invoking Futhark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from compiler import Compiler,generate,digest

p=argparse.ArgumentParser();p.add_argument('--remora',required=True);p.add_argument('--source',required=True)
p.add_argument('--output-dir',default=str(Path(__file__).resolve().parent));a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
# Type validation is deliberately separate: library recognition currently uses
# the stable parsed AST, while the normal type checker validates source first.
subprocess.run([a.remora,'dev','--typecheck','--json','-f',a.source],stdout=subprocess.DEVNULL,check=True)
raw=subprocess.check_output([a.remora,'dev','--parse','--json','-f',a.source])
ast=json.loads(raw);contracts=json.loads(Path(__file__).with_name('contracts.json').read_text())
graph=Compiler(ast,contracts).compile();graph['ast_sha256']=digest(ast)
graph['source_sha256']=hashlib.sha256(Path(a.source).read_bytes()).hexdigest()
(out/'yolov4-parse.json').write_bytes(raw)
(out/'generated_yolov4.py').write_text(generate(graph))
(out/'generated_yolov4.json').write_text(json.dumps(graph,indent=2)+'\n')
print(f'Generated {len(graph["nodes"])} tensor operations from {a.source}')
