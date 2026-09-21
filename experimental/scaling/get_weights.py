"""Retrieve and verify the exact trained checkpoint used in the first comparison."""
import argparse,hashlib,urllib.request
from pathlib import Path
SHA='e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/yolov4.weights'));a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists():
 temporary=a.output.with_suffix('.download')
 urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights',temporary)
 with temporary.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=SHA:raise ValueError('Checkpoint digest mismatch')
 temporary.replace(a.output)
with a.output.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!=SHA:raise ValueError('Existing checkpoint digest mismatch')
print(a.output,SHA)
