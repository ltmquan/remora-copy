# Controlled Darknet cuDNN comparison

One Slurm-allocated H200; same source, FP32 with TF32 disabled, checkpoint, 608×608 ramp image, compiler flags and harness. Each variant runs in three processes, alternating order; each process has five warmups and thirty timed single-image calls. Setup/search time is excluded.

“Search” replaces the forward convolution algorithm heuristic with measured cuDNN algorithm search. The existing workspace and candidate filters are retained. This does not claim an exhaustive optimization of Darknet.

| Variant | Total ms | Prediction + transfers ms | CPU decode ms | Total medians by round, ms |
|---|---:|---:|---:|---|
| off | 9.364 | 7.687 | 1.578 | 9.364, 9.211, 9.441 |
| heuristic | 11.311 | 9.777 | 1.536 | 11.229, 11.311, 11.783 |
| search | 11.559 | 9.978 | 1.630 | 11.525, 11.700, 11.559 |

Each displayed aggregate is the median of the three process medians; component medians need not sum exactly. All nine outputs passed the full 1,933,155-value comparison against off/round1 at atol=1e-5, rtol=1e-4. NMS is excluded; dense decode is included. This test does not include PyTorch and must not replace the prior matched PyTorch/Darknet allocation selectively.
The cuDNN heuristic path was about 21% slower than the CUDA/cuBLAS path in this controlled test, and measured forward-algorithm search was about 23% slower. Both were slower in every round. The difference is mainly in prediction/transfers rather than CPU decoding. This rules out different allocations as the sole explanation of the earlier gap and does not support heuristic selection alone as the fix; kernel-level profiling would be needed to isolate the cause. These conclusions are specific to this fork, cuDNN version, FP32/TF32-off policy and batch-one input.
