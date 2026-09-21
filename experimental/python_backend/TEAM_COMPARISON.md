# YOLOv4 on one NVIDIA H200 — Remora backend comparison

Batch 1, 608×608, trained YOLOv4 weights, FP32; TF32 disabled for the Python
frameworks. Latencies are medians of 10 synchronized calls after warm-up,
excluding startup/compilation. All paths pass numerical comparison against
Darknet on the same deterministic image (1,933,155 output values).

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

Live-allocation values are Futhark runtime high-water counters, distinct from
the sampled process footprint. **— means not measured**, not zero. Comparable
live-allocation counters have not yet been collected for the other backends.

GPU-resident: input and weights already on GPU, dense output retained there.
Transfer-inclusive: image upload and dense output download included; weights
remain resident. Darknet's API includes transfers, so its resident entry is
unmeasured. Dense decode matches Remora's output contract; NMS is excluded.

Unmarked rows: job 10275526. †: job 10275644, on the same physical H200 in a
separate allocation. That allocation repeated baseline Remora/tinygrad at
152.343 ms resident and 156.424 ms transfer-inclusive, confirming a 4.18×
resident speedup; its Darknet control measured 9.729 ms including dense decode.
Small differences between Remora and direct models are not statistically
meaningful with this sample count and fixed execution order.

## What changed in tinygrad

Both versions use TinyJit with BEAM=0. Baseline uses tinygrad's ordinary conv2d,
which expresses convolution as multiplication and reduction over sliding-window
views. The optimized version explicitly packs patches into a contiguous matrix
(im2col), then computes convolution with matrix multiplication. It also
precomputes fixed batch-normalization weights and materializes convolution
outputs. BN folding and output materialization alone produced essentially no
speedup; explicit matrix lowering produced the substantial gain.

The network connectivity, weights, FP32 precision and output contract remain
the same. No PyTorch/cuDNN calls are introduced. The result is about 4.18× faster
with about 3.06× greater GPU footprint (920 → 2,812 MiB). Remora and direct
optimized outputs are bitwise identical. This is a YOLO-specific backend
prototype, not a general Remora-to-Python compiler.

## How to interpret memory

The table uses **sampled peak GPU process footprint**, measured in separate
processes at a requested 20 ms interval. It includes setup, CUDA context,
libraries, caches and inference workspaces, excluding the extra changed-input
correctness test. Sampling can miss brief peaks. NVIDIA describes this as
GPU-context memory usage: [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/).

It is a useful cross-framework capacity metric, but there is no single memory
number that answers every question. Useful complementary metrics are:

- Peak live allocation: memory occupied by active tensors/buffers.
- Peak allocator reservation: memory held by an allocator, including cached blocks.
- Steady-state footprint: memory held after initialization and warm-up.

PyTorch explicitly distinguishes allocated and reserved memory, and notes that
unused cached allocations still appear in nvidia-smi:
[PyTorch memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management).
We have not collected matching live-allocation and reservation counters for
every backend, so those should not be mixed into the common footprint column.

Futhark's internal peak live allocation falls from 14,545.235 MiB to
1,916.006 MiB (86.8% lower) with the modified source, while the driver-reported
process peak stays near 30,048 MiB. The runtime caches allocations and can use
managed memory; the exact breakdown of the remaining footprint is unresolved.
This is not evidence of an 86.8% reduction in total process memory.

## Main findings and limits

- Modified Remora/Futhark is about 1.89× faster than the original. The modified
  source combines reduce/zero with cascaded 5×5 spatial pyramid pooling.
- Remora introduces no apparent inference penalty relative to the direct
  Python models at this measurement resolution. Backend lowering dominates.
- PyTorch is the strongest measured speed option; tinygrad uses less memory,
  with im2col providing a substantial speed/memory tradeoff.
- Native Darknet has cuDNN disabled; PyTorch uses cuDNN. This is not a
  best-possible C++ or Tensor Core comparison. Numerical agreement on one
  deterministic image is not a COCO mAP evaluation.
- Useful future additions: matched live/reserved memory counters, latency
  variation across independent runs, and separately reported cold-start cost.
  None is required to use the measured table above.

Full methods, source provenance, raw measurements and validation are in
[RESULTS.md](RESULTS.md) and [evidence](evidence).
