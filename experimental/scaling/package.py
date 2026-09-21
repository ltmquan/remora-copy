"""Build an upload archive with code and prepared artifacts, no credentials/results."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files=[]
 for size in (608,960,1216,1536):
  for variant in ('original','modified'):
   directory=ROOT/'artifacts'/str(size)/variant
   metadata=json.loads((directory/'source.json').read_text())
   if hashlib.sha256((directory/'model.remora').read_bytes()).hexdigest()!=metadata['source_sha256']:raise ValueError(f'Edited source: {directory}')
   if not metadata['frontend_validated']:raise ValueError(f'Unvalidated frontend: {directory}')
   for name in ('model.c','model.h','model.json','build.json'):
    if not (directory/'futhark'/name).is_file():raise FileNotFoundError(directory/'futhark'/name)
   build=json.loads((directory/'futhark/build.json').read_text())
   if len(build['stages'])!=3 or any(stage['exit_code'] for stage in build['stages']):raise ValueError(f'Incomplete CUDA generation: {directory}')
   if build['source_sha256']!=metadata['source_sha256']:raise ValueError(f'Source mismatch: {directory}')
 for path in ROOT.rglob('*'):
  if not path.is_file():continue
  rel=path.relative_to(ROOT)
  if any(part in ('results','__pycache__','.venv') for part in rel.parts):continue
  if path.suffix not in ('.py','.sh','.cpp','.md','.json','.cfg','.remora','.c','.h') and path.name!='darknet-source.tar.gz':continue
  files.append(path)
 backend=ROOT.parent/'python_backend'
 files += [p for p in backend.iterdir() if p.is_file() and p.suffix in ('.py','.json','.cfg','.cpp')]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 manifest={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
 with tarfile.open(a.output,'w:gz') as archive:
  for path in sorted(files):archive.add(path,arcname=str(Path('experimental')/path.relative_to(ROOT.parent)))
 a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(a.output,len(files),'files')
if __name__=='__main__':main()
