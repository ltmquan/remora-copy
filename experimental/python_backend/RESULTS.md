# YOLOv4 Python-backend experiment — 2026-09-11

The original Remora program can run through Python/PyTorch or Python/tinygrad
without Futhark. The Remora-generated and independently cfg-derived models
produce identical outputs within each framework in the completed main matrix.
PyTorch is the more promising replacement in this experiment: it reduces the
Futhark gap to a small gap against native Darknet in the fresh matched run. This does
not establish that every Remora program benefits, or that the fastest possible
Darknet/cuDNN configuration has been beaten.

## Latest timing and memory comparison

**New measurements**, batch 1, 608×608, trained checkpoint, FP32 with TF32
disabled. Both allocations used the same physical H200 on d4052:
`GPU-6aa8d328-b2be-0104-bb56-1da50e9d16d9`, driver 570.86.15.
Unmarked rows are job **10275526**; optimized tinygrad rows (†) are the
confirmation job **10275644**. Medians use ten synchronized calls with setup,
compilation and warm-up excluded. Memory comes from separate instrumented runs.

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

The confirmation allocation also reran baseline tinygrad: **152.343 / 156.424 ms**
for Remora and **152.350 / 156.430 ms** for direct. Its native Darknet control
was **9.729 ms** including dense decode. Thus the approximately **4.18× resident
speedup** of the new tinygrad lowering is confirmed against a baseline in the
same allocation, not inferred solely across jobs. Its memory footprint is
approximately **3.06× larger**. PyTorch and native Darknet remain substantially
faster in these configurations. Small Remora/direct differences are not
statistically meaningful with ten samples and fixed run order.

### Memory definition and Futhark's allocation counter

The table reports the largest **per-process GPU footprint** sampled by
`nvidia-smi --query-compute-apps=pid,used_gpu_memory` at a requested 20 ms interval,
from setup through inference, excluding the extra replacement-input correctness
check. It includes context, libraries, workspaces and allocator caches; it is
not just live tensor storage. Whole-process peaks including validation and
per-phase samples are retained separately. Short transient peaks can be missed.
NVIDIA defines this metric as memory used by the GPU context in its
[nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/).

Futhark's internal report measures something different:

| Source | Peak live device allocation, bytes | MiB |
|---|---:|---:|
| Original | 15,251,784,580 | 14,545.235 |
| Modified | 2,009,077,508 | 1,916.006 |

That is an **86.8% reduction in Futhark's live-allocation counter**, while the
observed driver-reported process footprint is essentially unchanged. The
generated runtime caches freed allocations and automatically enables managed
memory when supported. The exact breakdown of the additional driver footprint
has not been isolated; the internal counter must not be presented as a
comparable process-memory saving. The input/parameter arrays alone total about
250 MiB; the dense output is another 7.37 MiB.

### Modified source and correctness

The modified source was freshly compiled from
`remora-copy/examples/darknet/yolov4.remora`, SHA-256
`334346d83578f49a52507fab11d0cfed0da269b10f20ab012ade4f6ca98a6e89`.
It combines `reduce/zero` with cascaded 5×5 pooling for SPP. It is approximately
**1.89× faster** than the original in this new run. Both have the same 542-input
ABI/fixture, trained checkpoint, and dense output contract. The original
Remora source and its generated Python graph remain unchanged.

Every path passes against fresh Darknet over **1,933,155 values**, with no
nonfinite values, at `abs(diff) <= 1e-5 + 1e-4 * abs(Darknet)`.
The modified Futhark maximum absolute difference is **6.869e-6**; optimized
tinygrad's is **4.530e-6**. Remora and direct optimized outputs are bitwise
identical. Replacement-input checks and signed-data GPU convolution tests
also pass. This remains numerical validation on a deterministic image, not
COCO detection-quality evaluation.

Darknet's current API includes transfers, so GPU-resident time is unmeasured.
Its dense decode adapts the head output to Remora's 22,743×85 result; no path
includes NMS. Darknet uses CUDA/cuBLAS with cuDNN disabled; PyTorch uses cuDNN.
These results do not establish a best-possible native C++ performance limit.

### Tinygrad diagnosis and implementation

The earlier `BEAM=2` runs were externally limited to **480 seconds each** and
exited with status **124** before first inference completed. BEAM=2 is search
width, not a time or iteration limit. The new 180-second diagnostic reused
21 cached searches and spent **164.648 seconds finishing five new searches**,
then timed out during the next. No per-candidate compilation timeout was logged.
This demonstrates expensive tuning work rather than slow steady-state calls.

The separate kernel profile attributes **52.8%** of baseline GPU execution time
to twelve large 3×3 convolutions. `tinygrad_optimized.py` adds contiguous patch
packing and matrix multiplication, plus once-only BN folding. BN folding or
output materialization alone stays near 152 ms; Winograd reaches 44.428 ms;
the matrix lowering is the fastest validated variant. All remain FP32, BEAM=0,
and use tinygrad operations without PyTorch/cuDNN calls.

Run the improvement with `benchmark.py --backend tinygrad --model remora
--mode jit --tinygrad-opt im2col --weights yolov4.weights --output PREFIX`.
Use `--model direct` for the independent implementation. The baseline remains
available for reproducibility. See [TINYGRAD_ANALYSIS.md](TINYGRAD_ANALYSIS.md)
for the search explanation, sweep and kernel evidence.

Evidence: [timing/memory matrix](evidence/memory_matrix),
[matrix validation](evidence/memory_matrix/validation.json),
[optimized confirmation](evidence/tiny_optimized),
[optimized validation](evidence/tiny_optimized/validation.json), and
[executed source snapshot](evidence/tiny_optimized/source).
The CPU build completed in 2:17; the three sequential H200 jobs completed in
5:57, 5:35 and 1:59. No GPU jobs from this work remain running.

## Earlier six-path comparison including original Futhark

**All six rows below are newly measured**, sequentially in Slurm job
**10272589**, on H200 node **d4052** (GPU UUID
`GPU-6aa8d328-b2be-0104-bb56-1da50e9d16d9`). The job completed in 2:03.
608×608, batch 1, FP32, the same trained checkpoint and deterministic image;
medians of 10 synchronized calls. Setup, compilation, and warm-up are excluded.
Futhark and Python use three warm-ups per timing mode; Darknet uses one.

| Implementation | GPU-resident median ms | Transfer-inclusive median ms |
|---|---:|---:|
| Remora → Futhark → CUDA, original source | **910.642** | **911.403** |
| Remora → Python → PyTorch, compiled / CUDA-graph mode | **9.226** | **9.998** |
| Direct PyTorch, compiled / CUDA-graph mode | **9.325** | **10.125** |
| Remora → Python → tinygrad, TinyJit/CUDA, BEAM=0 | **152.314** | **155.713** |
| Direct tinygrad, TinyJit/CUDA, BEAM=0 | **152.362** | **155.518** |
| Native Darknet CUDA/cuBLAS + dense decode | — | **9.023** |

The Futhark row was freshly compiled from `remora/examples/darknet/yolov4.remora`,
not `remora-copy` or an older generated binary. Its source SHA-256 matches the
Python translation: `05882890a3a6b1b14470ab2a91593185df9535ff49dfb94b3a06f2172b8876d7`.
All 542 input names and ranks were checked against the Python graph; the trained
fixture hash is `ccd50c843493cfdc36d179764a89948945b94ac83e969aa19a4e069e6ceb0dba`.
Fresh Remora/Futhark/CUDA code generation took 91.136 seconds locally, followed
by a CPU-node C build (job 10272588). Futhark runtime setup took 14.147 seconds;
its first warm-up took 2.648 seconds. None contributes to the table's timings.

GPU-resident measurements retain inputs/weights and the dense output on the GPU;
transfer-inclusive measurements upload the image and download the dense output,
with weights retained. Both include host dispatch and synchronization. Darknet's
current prediction API includes transfers, so this harness has no GPU-resident
measurement. Its prediction API alone measured **7.726 ms**; the dense decode
adapter brings the result to the same 22,743×85 output contract as Remora.
No row includes NMS. Darknet uses cuBLAS with cuDNN disabled; PyTorch uses cuDNN.
PyTorch folds BN during setup; tinygrad uses the existing JIT configuration.

All five implementations pass against fresh Darknet across **1,933,155 values**
with no nonfinite values, at `abs(diff) <= 1e-5 + 1e-4 * abs(Darknet)`.
Futhark's maximum absolute error is **4.709e-6**; the largest across all paths is
**6.735e-6**. Remora and direct outputs are bitwise identical within each Python
framework. The Python captured paths also pass the changed-input check.

Remora/PyTorch is approximately **98.7× faster** than Remora/Futhark GPU-resident
and **91.2× faster** including transfers, while remaining about **10.8% slower**
than this Darknet reference including dense decode. These are ten-sample results
with fixed run order, not evidence that small Remora/direct differences matter.
See [raw fresh results](evidence/fresh), [validation](evidence/fresh/validation.json),
and [fresh build provenance](evidence/fresh/futhark-build.json).

Earlier experiments follow for historical context; their numbers are not used
in the fresh table above.

## Earlier CUDA-graph experiment

The final `torch.compile(mode="reduce-overhead")` run, **job 10271760**, used
setup-time BN folding and separate warm-up for fresh-image inputs.

| Implementation | GPU-resident median ms | Transfer-inclusive median ms |
|---|---:|---:|
| Remora → PyTorch, compiled / CUDA-graph mode | **9.300** | **10.371** |
| Direct PyTorch, compiled / CUDA-graph mode | **9.320** | **10.137** |
| Native Darknet + dense decode, same allocation | — | **9.173** |

Remora has no meaningful GPU-resident penalty compared with the direct model
in these samples. Transfer-inclusive Remora/PyTorch remains about **13% slower**
than this native C++ reference; native C++ remains faster in these measurements.
The small differences between the two Python models should not be interpreted
as statistically significant with ten samples and fixed run order.

Both outputs pass the Darknet tolerance and are bit-for-bit identical to one
another. An untimed replacement-image test also passes against eager execution;
the changed image changes the output (maximum difference 8.3679), verifying
that capture consumes new data. Compilation/capture warm-up was 15.683 seconds
for Remora and 13.620 seconds for direct PyTorch, using the existing cache;
fresh-input signature warm-up, including any additional compilation, was kept
out of the transfer timing. These are separate measurements from the main
matrix below, with a fresh C++ reference in each allocation.

## BN folding and tinygrad tuning

**Job 10271579** measured eager PyTorch with BN folded once: **10.344 ms**
GPU-resident / **11.076 ms** transfer-inclusive for Remora, and **10.388 ms** /
**11.143 ms** for the direct model. Darknet in that allocation measured
**8.847 ms** with dense decode. All three numerical comparisons pass.

Both tinygrad `BEAM=2` attempts hit their **480-second timeout during first
inference/kernel search** (exit 124), so there is no tuned inference result.
The direct attempt could reuse the first attempt's partial search cache, but
also did not finish within its bound. This is a tuning-budget limitation, not
proof that tuned tinygrad cannot improve. The completed tinygrad timings below
are **CUDA runtime, TinyJit, BEAM=0, ALLOW_TF32=0**; they do not cover the NV
runtime, reduced precision, larger batches, or multiple GPUs.

## Earlier Python-only comparison

Slurm job **10271432**, H200 node **d4054**, GPU UUID
`GPU-bc9577c4-73a0-4061-a86a-82e8fa20fc5e`, driver 570.86.15.
608×608, batch 1, trained YOLOv4 weights, deterministic image, FP32,
TF32 disabled in both frameworks, 3 warm-ups and 10 measured calls.
All rows in this table were measured sequentially in this allocation.

| Implementation | GPU-resident median ms | Including transfers / dense decode, median ms |
|---|---:|---:|
| Remora → PyTorch, eager | 11.947 | 12.712 |
| Direct PyTorch, eager | 12.208 | 12.934 |
| Remora → PyTorch, compiled + BN folded once | 9.882 | 10.636 |
| Direct PyTorch, compiled + BN folded once | 9.843 | 10.619 |
| Remora → tinygrad, TinyJit/CUDA, BEAM=0 | 152.295 | 155.782 |
| Direct tinygrad, TinyJit/CUDA, BEAM=0 | 152.323 | 155.654 |
| Native Darknet CUDA/cuBLAS + dense decoding | — | 9.108 |

Darknet's prediction API alone measured **7.690 ms**, including input upload
and head-output download. Its final dense decoding is on the CPU. Python dense
decoding runs on the GPU, and its transfer-inclusive measurement downloads the
complete dense array. Neither includes image preprocessing, thresholding, or NMS.
Darknet has cuDNN disabled; PyTorch uses cuDNN 9.10.2. This is the native C++
reference used in the preceding project comparison, not a best-possible
cuDNN/Tensor Core C++ baseline.

The source's per-call BN folding is retained in eager rows. Compiled rows fold
it once before compilation, as Darknet does. Initial compilation warm-up was
60.379 seconds for Remora/PyTorch and 30.071 seconds for direct PyTorch; the
shared compiler cache and fixed execution order make those cold-start numbers
unsuitable for comparing frontend compile speed. GPU synchronization is inside
all inference timing boundaries.

The initial matrix did not separately warm the fresh-input alias signature:
the first transfer-inclusive compiled call triggered ~3.5 seconds of additional
compilation. The table reports the median of the recorded samples, so that
outlier is not the median, but these initial transfer samples are not uniformly
steady-state. Later runs use explicit fresh-input warm-up; raw samples are
retained rather than silently deleted.

## Correctness

The main matrix passed all dense-output comparisons against native Darknet:
**1,933,155 values per output**, no nonfinite values, no failures under
`abs(diff) <= 1e-5 + 1e-4 * abs(Darknet)`.

- Synthetic weights: maximum absolute difference at most 2.3842e-7.
- Trained weights, PyTorch eager: 4.3511e-6.
- Trained weights, PyTorch compiled: 6.4373e-6.
- Trained weights, tinygrad JIT: 6.7353e-6.
- Remora versus direct within each framework/mode: maximum difference **0**.

The checkpoint's SHA-256 matches the repository's Git LFS pointer:
`e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562`.
Trained weights were tested with a deterministic image; this is numerical
validation, not a COCO mAP or real-image detection-quality measurement.

Additional tests passed: original-source Remora type checking, five compiler
checks (including rejecting modified recognized helpers), independent topology
comparison, and four GPU operator comparisons covering signed strided
convolution/BN/Mish, negative pooling padding, upsampling order, and anchor decode.

## Relation to the previous Futhark results

Historical Futhark reduce/reduce-zero results used synthetic weights and older
binaries. They are superseded for this comparison by the freshly compiled,
original-source, trained-weight matrix above.

The PyTorch result benefits from recognizing Remora's convolution idiom and
calling a library convolution. It should not be interpreted as evidence that
merely emitting Python for the existing scalarized Futhark-style reductions
would give the same gain.

## Implementation and scope

`build.py` invokes the actual Remora parser and type checker. `compiler.py`
expands composite source functions and emits 150 operations, including all 110
convolutions. Checked library idioms lower to native tensor operations. The Python paths use no
Futhark IR, Futhark compiler, or Futhark runtime in execution.
`direct.py` builds an independent model from Darknet cfg. Runtime math routines
are shared to isolate translation and connectivity, while C++ Darknet provides
an independent numerical reference.

This is a **YOLO-specific experimental backend**. It is not a general
rank-polymorphic Remora-to-Python compiler. Recognized helper definitions are
checked against their AST hashes, and unsupported/redefined constructs fail
rather than silently substituting a fixed YOLO network. The existing Futhark
backend and `remora-copy` were not modified.

## Reproduction and evidence

See [README.md](README.md) for build/run commands and bounded Slurm scripts.
Raw timing arrays, stdout/stderr, environment versions, and validation summaries
are under [evidence](evidence). `summarize.py evidence` regenerates timing tables.
Remote working directory: `/scratch/luu.qu/remora-python-20260911`.
The successful GPU jobs completed in 4:14 (main), 16:21 (bounded tuning),
and 1:03 (final graph mode), with no jobs left running.
All jobs used at most one H200 at a time; dependency installation ran on CPU nodes.
The initial setup attempts (10271238, 10271366) ended before full-network timing
because of a C++ declaration and a tinygrad device-selection compatibility issue;
10271432 is the successful main matrix.
