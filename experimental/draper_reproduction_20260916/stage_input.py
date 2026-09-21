"""Reconstruct the published input from its downloaded image and cached official weights."""
import importlib.util,pathlib,struct
p=pathlib.Path('upstream')
spec=importlib.util.spec_from_file_location('mk',p/'mk_yolov4_input.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import os
os.chdir(p)
prefix=pathlib.Path('yolov4_input.bin').open('rb').read(8*1024*1024)
assert prefix[:7]==b'b\x02\x03 f32'
shape=struct.unpack('<QQQ',prefix[7:31]);assert shape==(3,608,608),shape
image=prefix[31:31+3*608*608*4]
assert len(image)==3*608*608*4
weights=pathlib.Path('/scratch/luu.qu/remora-python-20260911/yolov4.weights').read_bytes()[20:]
shapes=m.entry_shapes();offset=0
with open('reconstructed.bin','wb') as f:
 for name,rank in m.entry_params():
  shape=shapes[name];n=m.nbytes(shape)
  if name=='image': chunk=image
  else: chunk=weights[offset:offset+n];offset+=n
  assert len(chunk)==n
  f.write(m.header(shape));f.write(chunk)
assert offset==len(weights)
reconstructed=pathlib.Path('reconstructed.bin').open('rb').read(len(prefix))
assert reconstructed[:len(prefix)]==prefix,'Published prefix differs'
print('Reconstructed bytes:',pathlib.Path('reconstructed.bin').stat().st_size,'matched published prefix:',len(prefix))
