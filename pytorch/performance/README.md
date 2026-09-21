# GPU performance measurements

**Follow-up:** [Native-first optimization results](NATIVE.md) now include a
validated full-network run at 26.056 ms. This page preserves the earlier
general-backend measurements.

This harness compares exact small YOLO helper sources through Remora→Futhark
CUDA, the current Remora→PyTorch compiler, and direct PyTorch. It also reruns
full 608×608 YOLO with the existing Futhark CUDA build and direct PyTorch.
Results are accepted only when output comparisons pass.

The earlier 120-second limit was in `validation/check_device.sbatch`, around
the full-network probe process. It is not a timeout inside either compiler
runtime. This benchmark uses smaller inputs with no per-inference timeout;
the shared-cluster allocation still has a 15-minute Slurm wall limit.

## Method

- One GPU, float32, TF32 disabled, sequential execution in one Slurm allocation.
- Three warm-ups and ten timed runs; report the median and all samples.
- Host wall time bracketed by CUDA synchronization; input tensors resident,
  output remains on the GPU. Compilation, initialization, and validation are
  excluded. Futhark frees the returned array outside the timed region.
- PyTorch measurements execute eagerly. “Remora→PyTorch” refers to the
  generated compiler output, not `torch.compile`/Inductor optimization.
- Small inputs are seed 17 from the passing interpreter differential suite.
  All available implementations must agree with the saved interpreter output
  at `atol=1e-5, rtol=1e-4`, including shape and finite-value checks.
- The direct composed pipeline is independently written with native tensor
  operations. It was also checked against CPU interpreter outputs for seeds
  0, 1, and 17 before cluster submission.
- Small tests disable cuDNN autotuning. The reused full-network direct harness
  enables cuDNN autotuning and validates that a changed image changes output.

Small shapes: convolution input `[2,4,4]`, weights `[3,2,3,3]`; pooling input
`[2,5,5]`; composed pipeline input `[1,5,5]`, conv/BN/Mish → pooling → upsampling
→ detection head, output `[300,7]`. These measure small-workload overhead and
are not throughput estimates for full YOLO.

The current Futhark compiler cannot generate CUDA for the small composed case:
`Cannot allocate memory block mem_21179 : mem@f32 in kernel block.` That row is
reported as unavailable, not assigned an invented timing. The convolution and
pooling cases compile through all three implementations.

## Reproduce

From `remora-copy/pytorch`, after generating the correctness fixtures:

```sh
python performance/prepare.py --remora /absolute/path/to/remora \
  --futhark /absolute/path/to/futhark \
  --fixtures .build/yolo-differential-final --out .build/performance-small
```

Copy the prepared case directories and `performance/{benchmark.py,
full_baselines.py,run.sbatch}` into an isolated cluster directory, then submit
`sbatch run.sbatch`. The script reuses the existing cluster Python environment,
checkpoint, and full-network reference binaries read-only. New artifacts stay
under `/scratch/luu.qu/remora-copy/pytorch/performance-20260912`.

The full Futhark baseline's recorded source hash must match the exact selected
YOLO source before it runs. Its output is compared to full direct PyTorch over
1,933,155 values before the collected baseline timings are reported.

## Measured small workloads

Tesla V100-SXM2-32GB, node `d1022`, GPU UUID
`GPU-0fcb89b2-1d05-5bde-55d9-d790827a1625`, driver `545.23.08`,
PyTorch `2.8.0+cu128`, Futhark CUDA libraries linked with CUDA 12.3.
Slurm job `10299357`. All times below are median milliseconds.

| Workload | Remora→Futhark | Current Remora→PyTorch | Direct PyTorch |
| --- | ---: | ---: | ---: |
| conv_same | 0.014 | 152.084 | 0.041 |
| spp_cascade | 0.088 | 1178.158 | 0.132 |
| composed_pipeline | Build failure | 3279.070 | 1.195 |

All available outputs pass comparison against the Remora interpreter.
Maximum observed absolute error is `2.38418579102e-7`. The current compiler
selected general lowering for all three workloads. Raw samples, errors, and
source hashes are in [results.json](results.json).

## Profile diagnosis

A separate, untimed `cProfile` run of the composed workload records 26,650
`_g_apply` calls, 205,659 `_g_cell` calls, 18,748 `torch.stack` calls, and
135,363 tensor reshapes. Python iterates through lifted array cells and creates
many tiny tensor operations. This explains why the general backend cannot
currently scale to full YOLO. The profile itself adds overhead, so its elapsed
time is not used for the benchmark table. Numeric indexing did not call
`.item()` in this workload; that runtime branch handles non-tensor objects.

## Full 608×608 YOLO baselines

On the same V100 allocation, the existing Remora→Futhark build measured
**1012.758 ms**, and direct PyTorch eager measured
**23.222 ms**, median GPU-resident latency.
Direct PyTorch's transfer-inclusive median was
**25.331 ms**.

The two full-network outputs agree within `atol=1e-5, rtol=1e-4` over
**1,933,155 values**. Maximum absolute error is `5.8114528656e-06`.
This comparison is Futhark versus independent direct PyTorch; the new general
compiler still has no completed full-network output or full-network timing.

Direct PyTorch first inference took 2.849 seconds,
including cuDNN tuning, versus 23.222 ms
steady state. Thus a subsecond expectation is reasonable for steady-state
native GPU execution. Increasing the previous test limit would not address
the new compiler's measured overhead on much smaller inputs.

See [full-baselines.json](full-baselines.json), [full-direct.json](full-direct.json),
and [full-futhark.log](full-futhark.log) for samples and provenance. The full
Futhark C source was reused read-only and relinked into the isolated benchmark
directory against CUDA 12.3. No original compiler/repository files were changed.

The final Slurm allocation `10299357` completed successfully in **4 minutes
4 seconds**, exit `0:0`, peak host RSS `1729832K`. The H200 request `10299258`
was canceled while pending because its estimated wait was about 40 minutes.
An initial V100 attempt `10299351` failed after 17 seconds before timing Futhark;
retrying with CUDA 12.3 produced the complete results above. That first attempt
did not preserve the Futhark error text, so the precise startup failure is not
established, although changing the toolkit resolved it. No allocation remains
running for this benchmark.
