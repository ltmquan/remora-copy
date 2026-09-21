# Eight-H100 setup checks — 11 September 2026

The rental is configured: eight H100 80 GB GPUs in one machine, NV18 connectivity between every pair, 208 logical CPUs, about 1.7 TiB RAM, CUDA 12.8 / driver 570.148.08, Python 3.13.15, PyTorch 2.8.0+cu128, tinygrad 0.14.0 and NumPy 2.5.3. All source edits are in remora-copy.

All nine variants passed 608² smoke checks on one and eight GPUs. Both PyTorch variants also passed on two and four GPUs. Round-robin, calibrated weighted and dynamic request assignment passed on eight GPUs. These are whole-image replicas, not tensor/model parallelism; NVLink bandwidth has not been benchmarked.

At 1536², Remora → PyTorch and Remora → tinygrad-im2col passed on one and eight GPUs. Darknet exceeded the selected numerical bound on 748 / 12,337,920 values, so that case is flagged and has no accepted timing. At 960², original Remora → Futhark timed out during the 300-second setup/check window; GPU memory was observed at 81,087 MiB with heavy UVM activity. Modified remora-copy → Futhark passed at about 1.30 seconds/image. The original timeout is not a per-image runtime measurement.

The normal ramp input passed the original strict cross-backend checks at 608². The all-zero stress input exposed small rounding differences, so we generated independent FP64 reference outputs for all four sizes. Later cross-backend checks explicitly use atol=2e-5, rtol=2e-4 and retain violation counts under the original 1e-5/1e-4 bound. Execution remains FP32. Earlier failures remain in the evidence; the 1536² Darknet failure has not been hidden by widening the bound again.

Separate 608² transfer-inclusive memory passes ran on eight GPUs for at least three seconds per worker, collecting 20–24 samples each. This table shows the largest per-GPU warmed process footprint, not the sum across GPUs and not the entire process-startup peak. Raw JSON also contains the sampled simultaneous sum. Memory-pass throughput is deliberately null.

| Implementation | Max per-GPU footprint (MiB) | Framework counter† (MiB) |
|---|---:|---:|
| Remora → PyTorch | 12,486 | 508.6 |
| Direct PyTorch | 12,468 | 509.8 |
| Remora → tinygrad, baseline | 918 | — |
| Direct tinygrad, baseline | 918 | — |
| Remora → tinygrad, im2col | 2,810 | — |
| Direct tinygrad, im2col | 2,810 | — |
| Original Remora → Futhark | 30,044 | 14,553.7 |
| Modified remora-copy → Futhark | 30,048 | 1,924.5 |
| Native Darknet + dense decode | 1,974 | — |

† These counters are not interchangeable working-set measurements. Futhark reports peak live buffers over its context lifetime. PyTorch reports allocator-active bytes after a per-trial reset; CUDA graph replay reuses previously captured storage, so this counter does not expose all captured temporary liveness. That explains the roughly 0.5 GiB PyTorch counter alongside a roughly 12 GiB footprint. Tinygrad/Darknet counters were not measured. The process footprint is the common comparison here.

The scalable native rows use a shared Python process scheduler: Darknet runs native C++/CUDA through ctypes, and Futhark runs its generated C library through ctypes. These timings include that host harness; they are not timings of a standalone C++ executable. All use the same SHA256-verified trained checkpoint and dense decoded output.

The full 884-cell matrix has **not** been run. Its dimensions remain 608²/960²/1216²/1536² × 1/2/4/8 GPUs × implementation × assignment policy × timing mode. Start the sustained sweep with uniform assignment, keep failure cells visible, and use separate memory runs. Local raw evidence is under `results/rental-20260911/`; the prepared matrix and setup scripts are alongside this file.
