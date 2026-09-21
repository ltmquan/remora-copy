"""Resize the two checked-in YOLO entry points, then run the real frontend."""
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / 'python_backend'
WORKSPACE = ROOT.parents[2]

def resize(text, size):
    if size < 32 or size % 32: raise ValueError('Image side must be a positive multiple of 32')
    prefix, entry = text.split('(entry (main', 1)
    mapping = {}
    for divisor in (1,2,4,8,16,32):
        old, new = 608//divisor, size//divisor
        mapping[str(old)] = str(new)
        mapping[str(old-1)] = str(new-1)
    counts = lambda s: [3*(s//d)**2 for d in (8,16,32)]
    a,b,c = counts(608); x,y,z = counts(size)
    mapping.update({str(a):str(x),str(b):str(y),str(c):str(z),str(b+c):str(y+z),str(a+b+c):str(x+y+z),'608.0':f'{size}.0'})
    # Lex whole atoms: preserve comments, float anchors, and identifiers such as w-152.
    entry = re.sub(r';[^\n]*|[^\s()\[\]{}]+', lambda m: mapping.get(m[0],m[0]), entry)
    return prefix+'(entry (main'+entry

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sizes',type=int,nargs='+',default=[608,960,1216,1536])
    p.add_argument('--remora',type=Path,default=WORKSPACE/'scheduling-investigation/bin/remora')
    p.add_argument('--futhark',action='store_true'); p.add_argument('--sources-only',action='store_true')
    a=p.parse_args()
    for size in a.sizes:
        out=ROOT/'artifacts'/str(size); out.mkdir(parents=True,exist_ok=True)
        cfg=(BACKEND/'yolov4.cfg').read_text()
        cfg=re.sub(r'(?m)^(width|height)\s*=\s*608\s*$',lambda m:m[1]+f'={size}',cfg)
        (out/'yolov4.cfg').write_text(cfg)
        for variant,repo in [('original','remora'),('modified','remora-copy')]:
            source=WORKSPACE/repo/'examples/darknet/yolov4.remora'
            dest=out/variant; dest.mkdir(exist_ok=True)
            text=resize(source.read_text(),size); target=dest/'model.remora'; target.write_text(text)
            metadata={'size':size,'variant':variant,'upstream_source':str(source.relative_to(WORKSPACE)),
                      'upstream_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'frontend_validated':False}
            (dest/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')
            if not a.sources_only:
                if variant=='original':
                    subprocess.run([sys.executable,str(BACKEND/'build.py'),'--remora',str(a.remora),'--source',str(target),'--output-dir',str(dest)],check=True)
                else:
                    subprocess.run([str(a.remora),'dev','--typecheck','--json','-f',str(target)],stdout=subprocess.DEVNULL,check=True)
                metadata['frontend_validated']=True
                (dest/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')
            if a.futhark:
                subprocess.run([sys.executable,str(BACKEND/'build_futhark.py'),'--source',str(target),'--out',str(dest/'futhark')],check=True)
            print(size,variant,'prepared',flush=True)
if __name__=='__main__': main()
