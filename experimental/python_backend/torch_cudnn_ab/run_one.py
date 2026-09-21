"""Run the existing benchmark with an explicit, recorded cuDNN switch."""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

enabled = os.environ['TEST_CUDNN'] == '1'
torch.backends.cudnn.enabled = enabled
base = Path('/scratch/luu.qu/remora-python-20260911')
sys.path.insert(0, str(base))
runpy.run_path(str(base/'benchmark.py'), run_name='__main__')
assert torch.backends.cudnn.enabled == enabled
out = Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.json')
d = json.loads(out.read_text())
d['cudnn_enabled'] = torch.backends.cudnn.enabled
out.write_text(json.dumps(d, indent=2)+'\n')
