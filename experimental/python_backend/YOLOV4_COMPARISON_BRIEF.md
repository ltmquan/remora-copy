# YOLOv4 backend comparison on NVIDIA H200

We ran the original `remora/examples/darknet/yolov4.remora` through
Remora → Futhark → CUDA and Remora → Python → PyTorch/tinygrad. We also ran
the modified `remora-copy/examples/darknet/yolov4.remora` through Futhark,
independent cfg-derived Python models in PyTorch and tinygrad, and native
Darknet CUDA/cuBLAS with dense decoding to match Remora's output format.
The Futhark compiler itself was not modified.

For Remora → Python, we wrote a small, specialized compiler prototype, not just
a manual translation. It reads the Remora code through the existing parser and
generates Python calls to PyTorch/tinygrad operations, but it currently only
handles the patterns used in this YOLOv4 program.

All runs used one H200, batch 1, 608×608 input, the same trained checkpoint and
deterministic image, and FP32 (TF32 disabled in the Python frameworks).
Timings are medians of 10 synchronized calls after warm-up; startup and
compilation are excluded. GPU-resident timing keeps inputs and outputs on the
GPU; transfer-inclusive timing includes image upload and dense-output download,
with weights retained. Memory was measured in separate runs.

## Results

| Pipeline | GPU-resident ms | Transfer-inclusive ms | Peak GPU footprint, MiB | Peak live device allocation, MiB |
|---|---:|---:|---:| ---: |
| Remora (original source) → Futhark → CUDA | 910.605 | 913.194 | 30,044 | 14,545.235 |
| Remora (modified `remora-copy` source) → Futhark → CUDA | 481.714 | 482.692 | 30,048 | 1,916.006 |
| Remora → Python → PyTorch, compiled / CUDA-graph mode | 9.277 | 10.462 | 12,480 | — |
| Direct PyTorch, compiled / CUDA-graph mode | 9.428 | 10.672 | 12,462 | — |
| Remora → Python → tinygrad, baseline JIT | 152.349 | 156.433 | 920 | — |
| Direct tinygrad, baseline JIT | 152.378 | 156.469 | 920 | — |
| Remora → Python → tinygrad, im2col JIT † | **36.476** | **40.956** | **2,812** | — |
| Direct tinygrad, im2col JIT † | **36.470** | **41.087** | **2,812** | — |
| Native Darknet CUDA/cuBLAS + dense decode | — | 9.312 | 1,974 | — |

- † Optimized tinygrad was confirmed in a separate allocation on the same
  physical H200. Its repeated Remora/tinygrad baseline was 152.343 ms resident,
  confirming a **4.18× speedup**. Other rows are from one comparison allocation.
- GPU footprint is the sampled process peak during setup/inference, including
  context, libraries, workspaces and caches. Sampling at approximately 20 ms
  can miss brief peaks. Live-allocation values are Futhark runtime counters.
  **— means not measured**, not zero.
- Every implementation passed comparison against Darknet over 1,933,155 output
  values, using `abs(diff) <= 1e-5 + 1e-4 * abs(reference)`. Remora and direct
  optimized tinygrad outputs were bitwise identical. This is numerical
  validation on one deterministic image, not a COCO accuracy evaluation.
- NMS is excluded. Darknet has cuDNN disabled; PyTorch uses cuDNN. This is not
  a best-possible C++ baseline. Small Remora/direct timing differences should
  not be treated as significant with ten samples and fixed execution order.

## Tinygrad optimization

Both tinygrad versions use TinyJit with BEAM=0. Baseline uses ordinary tinygrad
convolution over sliding-window views. The optimized version explicitly packs
image patches into a contiguous matrix (**im2col**) and computes convolution
with matrix multiplication. It also precomputes fixed batch-normalization
weights and materializes convolution outputs. BN folding and output
materialization alone barely helped; the matrix formulation provided the gain.

The optimization uses only tinygrad operations and preserves FP32 precision.
It trades extra buffers for faster execution:
**about 4.18× faster, with 3.06× greater GPU footprint** (920 → 2,812 MiB).
PyTorch remains faster in this experiment. This is a YOLO-specific backend
prototype, not a general Remora-to-Python compiler.

## What live allocation tells us

**Peak live allocation** is the largest amount of memory held by active runtime
buffers at once under the current execution schedule. Once a buffer is no
longer needed, the allocator can retain its storage for reuse. That storage is
no longer live, but may still contribute to the process footprint.

Intuition: clearing your desk reduces active workspace, but does not return
your reserved office space. Likewise, fewer live buffers need not reduce the
GPU memory held by the process.

The modified Remora source combines `reduce/zero` and cascaded 5×5 pooling.
It gives a **1.89× speedup** and an **86.8% reduction in Futhark's peak live
allocation**, yet the observed process footprint remains around 29.3 GiB.
So the live reduction has **not demonstrated a practical GPU-capacity saving**
in this configuration. The exact remaining footprint has not been isolated;
the runtime caches allocations and can use managed memory.

Live allocation can identify large intermediates and opportunities for better
memory reuse or release. A lower live peak suggests potential savings, not a
guarantee that the program fits on a smaller GPU. **Process footprint is the
primary capacity metric; live allocation is a secondary diagnostic.**

Full methods and raw evidence: [RESULTS.md](RESULTS.md).
