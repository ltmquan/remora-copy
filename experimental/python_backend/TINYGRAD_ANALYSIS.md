# Tinygrad investigation

## What the previous timeout means

The two earlier full-network `BEAM=2` attempts were launched by `tune.sbatch`
with GNU `timeout 480s`. Both returned **124**, after printing “first inference
started” but before “first_eager_seconds”. Thus the experiment stopped unfinished
startup/search after eight minutes per process. It did not measure a 480-second
steady-state inference, and the logs do not report an out-of-memory failure.
The semaphore warnings appeared during forced process shutdown.

In the exact installed tinygrad 0.14.0 source:

- `codegen/opt/search.py:beam_search` retains the best `amt` schedules per round.
  With BEAM=2, each of those schedules can generate many next candidates.
- Each candidate goes through code generation/compilation and GPU trial timing.
  Search continues until progress stops; width 2 does not cap total candidates,
  rounds, kernels, or elapsed time for the network.
- `_try_compile` has its own default `BEAM_TIMEOUT_SEC=10` alarm for a candidate.
  That limit is separate from our external 480-second whole-process budget.
- Finished search results are cached per AST/device/search configuration.
  The earlier direct-model attempt could reuse completed searches, but it still
  failed to finish within its own budget. The logs do not tell us which kernel
  was active at termination, so the new diagnostic logs search entry/exit events.

The new diagnostic uses four compiler workers and a clearly bounded 180-second
budget. It reuses the existing cache; its startup duration is not a cold-cache
comparison. DEBUG=2 profiling is separate from uninstrumented latency runs.

## Why an alternative lowering could help

`mixin/op.py:conv2d` expresses ordinary convolution using padded pooling views,
broadcasted multiplication, and channel/kernel reductions. This lets tinygrad
fuse operations, but the generated schedule also has to handle convolution's
indexing and data reuse. The YOLO model has 110 convolutions totaling about
128.389 GFLOP (multiply and add counted separately), before activations/decode.

`codegen/opt/postrange.py:_apply_tc_opt` skips the float-input CUDA Tensor Core
path when `ALLOW_TF32=0`. Our strict FP32 comparisons use that setting, matching
PyTorch's disabled TF32. These tests therefore do not measure tinygrad's
Tensor Core performance with TF32, FP16, or BF16.

The opt-in variants in `tinygrad_optimized.py` preserve the original Remora
model's connectivity and use tinygrad tensor operations throughout:

- **fold:** prepare fixed BN weights once.
- **materialize:** additionally make each convolution output contiguous, to
  test whether explicit intermediate boundaries improve scheduling.
- **im2col:** pack padded/strided patches into a contiguous matrix and use matrix
  multiplication, making patch reuse and reduction layout explicit. This trades
  extra intermediate storage and copying for a simpler multiplication kernel.
- **winograd:** use tinygrad's built-in Winograd transform for eligible 3×3,
  stride-one convolutions. It changes floating-point operation order, so full
  output validation is required even though tensor dtype remains FP32.

Local NumPy-reference tests pass for im2col and Winograd on signed inputs,
strided padding, odd output sizes, and replacement inputs with cached weights.
GPU full-network results are recorded separately; only candidates passing the
same Darknet tolerance are eligible for the final controlled comparison.

## Measured findings

The cache before this investigation held 21 completed BEAM=2 searches and 4,396
CUDA compiler entries (the latter also include earlier baseline work). In job
10275537, the diagnostic reused those 21 searches, completed five new searches
in 28.452, 65.113, 30.019, 21.105 and 19.959 seconds, and was stopped during
search 27 by the external 180-second budget. No per-candidate compilation
timeouts were logged. The next unfinished kernel shape was
`[256, 38, 38, 256, 3, 3]`. This is observed cumulative tuning cost, not an
inference taking minutes or evidence that the compiler had hung.

The separate baseline DEBUG=2 profile recorded 221 kernels and 154.523 ms of
GPU kernel time for the first eager execution. Twelve large 3×3 convolutions,
in two repeated schedule shapes, account for 81.598 ms (**52.8%**) of that time.
Those kernels achieved only about 0.5 TFLOP/s by tinygrad's operation estimate.
This localizes a substantial bottleneck to convolution execution. The counters
are a diagnostic, not hardware performance-counter measurements; host code
compilation is excluded from their summed GPU time.

All tested lowerings pass the full 1,933,155-value Darknet comparison at the
unchanged tolerance. In the sweep (job 10275537):

| Remora → tinygrad variant | Resident ms | Transfer-inclusive ms |
|---|---:|---:|
| Baseline | 152.247 | 156.533 |
| BN folded once | 152.295 | 156.804 |
| Materialized convolution outputs | 152.269 | 156.805 |
| Explicit im2col + matrix multiplication | **36.465** | **40.978** |
| Winograd | 44.428 | 48.969 |

The matrix lowering is the fastest validated candidate and is confirmed for
both Remora and direct tinygrad in job 10275644 (see RESULTS.md). Its full-output
maximum absolute error is 4.530e-6. It improves execution without BEAM tuning,
TF32, reduced precision, or calling PyTorch/cuDNN. The larger patch buffers
increase memory footprint, so this is a speed/memory tradeoff. BN folding alone
does not explain the improvement. The matrix formulation and layout change
are supported by the measured ablation; the profile alone would not establish
which hardware-level bottleneck dominates.

Evidence: [kernel summary](evidence/tiny_investigation/kernel-profile-summary.json),
[BEAM summary](evidence/tiny_investigation/beam-summary.json),
[raw profile](evidence/tiny_investigation/profile.log),
[raw search log](evidence/tiny_investigation/beam2.log), and
[full-output validation](evidence/tiny_investigation/validation.json).
