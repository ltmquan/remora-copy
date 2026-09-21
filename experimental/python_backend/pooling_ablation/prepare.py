from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
base=root.parent
og=(base.parent/'scaling/artifacts/608/original/model.remora').read_text()
md=(base/'teammate_futhark/previous-fork.remora').read_text()
teammate=(base/'teammate_futhark/model.remora').read_text()
def definition(s,name):
 start=s.index('(def (fun (@'+name+' ');i=start;depth=0
 while i<len(s):
  if s[i]==';':i=s.index('\n',i);continue
  if s[i]=='(':depth+=1
  elif s[i]==')':
   depth-=1
   if depth==0:return s[start:i+1]
  i+=1
 raise ValueError(name)
for layout,source in [('separate',og),('cascade',md)]:
 for reduction,poolsource in [('fold',og),('zero',teammate)]:
  name=f'{layout}-{reduction}';p=root/name;p.mkdir(exist_ok=True)
  prefix=source.split('(entry (main')[0]
  prefix=prefix.replace(definition(prefix,'max-pool-2d'),definition(poolsource,'max-pool-2d'))
  # Keep only the actual SPP workload as the entry; no convolution work.
  (p/'model.remora').write_text(prefix+'\n(entry (main (image [Float 512 19 19]))\n  (@spp _ (19 512) image))\n')
  subprocess.run(['python3',str(base/'build_futhark.py'),'--source',str(p/'model.remora'),'--out',str(p)],check=True)
