from pathlib import Path
import argparse,hashlib,json
from benchmark import fixture
from direct import YOLOv4
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=root/'fresh_futhark');a=p.parse_args();out=a.out.resolve()
arrays=fixture(YOLOv4(root/'yolov4.cfg'),root/'yolov4.weights')
abi=json.loads((out/'inputs.json').read_text())
h=hashlib.sha256()
for k,v in arrays.items():h.update(k.encode());h.update(v.tobytes())
with (out/'trained-inputs.bin').open('wb') as f:
 for name,shape in abi:
  assert list(arrays[name].shape)==shape
  f.write(arrays[name].astype('<f4').tobytes())
(out/'fixture.json').write_text(json.dumps({'input_sha256':h.hexdigest(),'weights_sha256':hashlib.sha256((root/'yolov4.weights').read_bytes()).hexdigest()},indent=2)+'\n')
print(h.hexdigest())
