# YOLOv8x in Remora

This follows `../darknet/yolov4.remora`: `main` accepts the prepared image and
trained float32 parameters, and returns decoded dense predictions. Every network
operation and the anchor/stride/DFL construction is in `yolov8x.remora`. The Python
builder is not needed to regenerate or interpret this saved source. This is the
expanded, shape-specialized model, not a Python callback or embedded MLIR.

- Input `image`: float32 RGB NCHW `[1,3,640,640]`, normalized to `[0,1]`.
- Other arguments: 206 fused trained weight/bias tensors, with descriptive names.
- Output: float32 `[1,84,8400]`: center x/y, width/height in image pixels, then
  80 class probabilities. Candidate order is 80x80, 40x40, then 20x20 grids.
- File decoding, image preparation and NMS are outside this entry, as in the
  existing YOLOv4 example. This is not the complete SEI instrumentation/TensorRT
  benchmark script, nor a raw-JPEG-to-boxes command.

## Compile once, run without Python

The generic MLIR toolchain uses Python during compilation. Its native runner
bundle does not use Python, PyTorch, Ultralytics, Racket or a model-specific C++
program at inference time. The CUDA driver and compatible OS/GPU are required.

From the research workspace, given the named model/input tensors in an NPZ file:

```sh
remora-mlir/.venv/bin/python remora-mlir/aot.py \
  remora-copy/examples/yolov8/yolov8x.remora \
  --inputs yolov8-inputs.npz --target cuda --cuda-target sm_89 \
  --output-dir yolov8-app
./yolov8-app/run.sh
```

The entry argument names are the NPZ keys. Preparing a checkpoint for a new model
is a one-time conversion, like unpacking YOLOv4 weights. An already built bundle
contains the converted weights and needs no converter. The evaluation's existing
frozen weights can be packaged with `eval-programs/03_yolov8/standalone/build.py`;
that script only imports saved tensors and invokes the generic compiler.

`run.sh` reads `inputs/*.npy`, calls the copied native `iree-run-module` executable
and writes `output.npy`. To use another prepared image, replace `inputs/image.npy`
with a tensor of the same shape/dtype. Copy the bundle to relocate it; paths are
relative to the bundle. One invocation at a time per bundle, because input/output
filenames are shared. Build separately for another GPU architecture or CPU.

Runtime guards are compiled into the module and fail the process on invalid
inputs. The runner is a shared language runtime, not YOLOv8 application logic.
The source also passes the original Remora frontend; the native packaging above
is for the MLIR backend. No claim is made that every other Remora backend is tested.

Validation and central source copy: `eval-programs/03_yolov8/standalone/` in the
research workspace. Its `verify.py` is optional test code invoking a separate
original native model; it is never called by the standalone application.

## Validated standalone release

The new source and native application passed all11 inputs on one L40S on
23 September2026 (job10553642): 7,761,600 dense values checked, zero tolerance
violations. See the central `standalone/REPORT.md` and per-case evidence.
The ready-made `standalone/bundle/run.sh` requires Linux x86-64 and a compatible
NVIDIA GPU/driver; the bundled build targets L40S (`sm_89`). It is not a macOS
executable. No Python is required to run this bundle.
