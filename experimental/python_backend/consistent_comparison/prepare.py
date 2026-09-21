"""Change only warmup/sample counts in isolated copies of existing Futhark harnesses."""
from pathlib import Path
base=Path('/scratch/luu.qu/remora-python-20260911')
root=Path(__file__).resolve().parent
for variant in ('fresh_futhark','modified_futhark'):
    source=(base/variant/'harness.c').read_text()
    assert source.count('run=-3;run<10')==2
    assert source.count('if(run==9)')==1
    source=source.replace('run=-3;run<10','run=-5;run<30').replace('if(run==9)','if(run==29)')
    out=root/variant
    out.mkdir(exist_ok=True)
    (out/'harness.c').write_text(source)
