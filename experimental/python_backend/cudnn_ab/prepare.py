"""Change only forward algorithm selection in an isolated Darknet copy."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
p = root / 'src/convolutional_layer.c'
s = p.read_text()
old = 'CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle(),'
assert s.count(old) == 1
s = s.replace(old, 'CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle(),')
# Keep the existing candidate filters, precision policy and workspace policy.
p.write_text(s)

