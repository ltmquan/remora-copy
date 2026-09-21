from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageOps
root=Path(__file__).resolve().parent
p=root/'inputs';p.mkdir(exist_ok=True)
y,x=np.indices((608,608))
images={'black':np.zeros((3,608,608),np.float32),'white':np.ones((3,608,608),np.float32),'noise':np.random.default_rng(20260912).random((3,608,608),dtype=np.float32),'gradient':np.stack([x/607,y/607,(x+y)/1214]).astype(np.float32),'checkerboard':np.broadcast_to(((x//16+y//16)%2).astype(np.float32),(3,608,608)).copy()}
im=ImageOps.exif_transpose(Image.open(root/'evalbox.jpg')).convert('RGB').resize((608,608),Image.Resampling.BILINEAR)
images['evalbox']=np.asarray(im,dtype=np.float32).transpose(2,0,1)/np.float32(255)
manifest=[]
for name,a in images.items():
 data=a.astype('<f4').tobytes();(p/f'{name}.bin').write_bytes(data)
 manifest.append({'name':name,'shape':list(a.shape),'sha256':hashlib.sha256(data).hexdigest()})
(root/'inputs.json').write_text(json.dumps({'images':manifest,'photo_source':'darpa-mocha/evaluation/evalbox.jpg','photo_preprocessing':'EXIF orientation, RGB, bilinear resize to 608x608 without letterboxing, CHW float32 divided by 255','noise_seed':20260912},indent=2)+'\n')
