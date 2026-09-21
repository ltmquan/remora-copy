from pathlib import Path
import tarfile
root=Path(__file__).resolve().parents[1]
out=root/'single_image/single-image-experiments.tar.gz'
files=list((root/'single_image').glob('*.py'))+[root/'single_image/README.md']
files+=list((root/'python_backend').glob('*.py'))
files += [root/'python_backend'/name for name in ['yolov4.cfg','darknet_benchmark.cpp','generated_yolov4.json','generated_yolov4.py']]
files += [root/'scaling/model.py']
for size in (608,1216,1536):
 files += [root/f'scaling/artifacts/{size}/yolov4.cfg']
 files += [root/f'scaling/artifacts/{size}/original'/name for name in ['generated_yolov4.py','generated_yolov4.json']]
with tarfile.open(out,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(Path('experimental')/p.relative_to(root)))
print(out)
