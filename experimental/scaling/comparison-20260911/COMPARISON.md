# YOLOv4 on the 8× H100 rental

## What we ran

YOLOv4 with official weights on one **8× H100 80GB** machine. We tested **608, 960, 1216, and 1536 square images** on **1/2/4/8 GPUs**, distributing whole images uniformly across batch-one model replicas.

**Each timing trial processes 128 images total across all GPUs**, repeated five times. Tables show median **images/sec**; runtime for 128 images is **128 ÷ images/sec**. Compilation, setup, and warmup are excluded.

**R** = GPU resident; **T** = includes image upload and dense output download. Inputs alternate between synthetic ramp and zero images. We check numerical agreement, not detection accuracy. Memory is measured separately after warmup.

**Results:** 172 timing cases passed, 10 failed, 90 skipped; all 38 memory cases passed. Further Futhark and partition-strategy runs were cut for cost.

**Fail** = numerical mismatch; **Timeout** = 600-second setup/check limit; **Skip** = unmeasured after a related failure; **Cut** = canceled for cost; **—** = unavailable.

## Remora → PyTorch

Original Remora → Python → PyTorch/CUDA. We wrote a small, specialized compiler prototype for these Remora operations, rather than manually translating the network; it is not yet a general Remora compiler.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 105.05 | 88.27 | 209.60 | 174.42 | 415.54 | 350.73 | 831.47 | 662.92 |
| 960×960 | 62.61 | 49.57 | 125.01 | 98.32 | 249.44 | 194.31 | 498.86 | 359.99 |
| 1216×1216 | 40.37 | 32.05 | 80.77 | 63.15 | 160.58 | 123.53 | 319.04 | 236.48 |
| 1536×1536 | 25.66 | 11.89 | 51.32 | 23.63 | 102.30 | 45.49 | 203.58 | 88.81 |

**Notes.** Near-linear GPU scaling. The main exception is 1536: transfers reduce single-GPU throughput from 25.66 to 11.89 images/sec; the cause is not yet isolated.

## Direct PyTorch

An independently written Python YOLOv4 network using the same PyTorch runtime, weights, and inputs. This is the control for overhead from the Remora frontend.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 103.28 | 87.88 | 207.68 | 173.99 | 412.16 | 344.10 | 813.39 | 679.06 |
| 960×960 | 62.39 | 49.55 | 124.71 | 98.33 | 246.57 | 194.91 | 494.42 | 373.83 |
| 1216×1216 | 40.24 | 32.06 | 80.43 | 63.24 | 160.52 | 123.46 | 318.68 | 238.14 |
| 1536×1536 | 25.47 | 11.86 | 50.99 | 23.27 | 101.85 | 45.39 | 203.44 | 89.65 |

**Notes.** Closely matches the Remora path at every size; there is no clear frontend performance penalty.

## Remora → tinygrad baseline

Remora-generated Python using baseline tinygrad convolutions and warmed TinyJit execution.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 6.61 | 6.42 | 13.13 | 12.76 | 26.26 | 25.53 | 52.50 | 50.66 |
| 960×960 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |
| 1216×1216 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |
| 1536×1536 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |

**Notes.** Good GPU scaling at 608, but slow per-image execution. Larger sizes fail numerical validation; remaining configurations were skipped.

## Direct tinygrad baseline

An independent Python YOLOv4 network using the same baseline tinygrad runtime.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 6.57 | 6.37 | 13.13 | 12.77 | 26.26 | 25.53 | 52.53 | 51.01 |
| 960×960 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |
| 1216×1216 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |
| 1536×1536 | Fail | Skip | Skip | Skip | Skip | Skip | Skip | Skip |

**Notes.** Matches the Remora baseline in both speed and larger-size validation failures.

## Remora → tinygrad im2col

Remora-generated Python with optimized tinygrad convolutions: contiguous image patches (im2col) become matrix multiplications, with cached batch-normalization-folded weights. This trades extra storage for speed; timing uses warmed TinyJit without BEAM search.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 26.87 | 23.84 | 53.69 | 47.55 | 107.32 | 94.09 | 213.55 | 181.76 |
| 960×960 | 13.73 | 11.70 | 27.46 | 23.50 | 54.90 | 46.51 | 109.68 | 91.81 |
| 1216×1216 | 11.26 | 10.08 | 22.50 | 19.96 | 45.00 | 39.99 | 89.97 | 79.61 |
| 1536×1536 | 6.14 | 5.30 | 12.28 | 10.59 | 24.55 | 21.12 | 49.03 | 41.66 |

**Notes.** About 4.1× faster than baseline at 608, and passes all sizes. Memory rises from the baseline’s 918 MiB to 2,810 MiB at 608.

## Direct tinygrad im2col

The independent Python network using the same optimized tinygrad runtime.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 26.70 | 23.73 | 53.42 | 47.51 | 106.81 | 93.63 | 213.36 | 186.32 |
| 960×960 | 13.73 | 11.63 | 27.45 | 23.25 | 54.87 | 46.57 | 109.68 | 91.60 |
| 1216×1216 | 11.26 | 10.07 | 22.51 | 20.08 | 45.01 | 40.03 | 89.96 | 79.16 |
| 1536×1536 | 6.14 | 5.34 | 12.28 | 10.63 | 24.55 | 21.06 | 49.03 | 41.83 |

**Notes.** Almost identical throughput to the optimized Remora path; also passes all sizes.

## Original Remora → Futhark

Original Remora → Futhark → CUDA, invoked through Python/ctypes. Futhark managed-memory paging can still transfer data in resident mode.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 1.02 | 1.02 | 2.03 | 2.03 | 4.07 | 4.07 | 8.14 | 8.13 |
| 960×960 | Timeout | Skip | Skip | Skip | Skip | Skip | Skip | Skip |
| 1216×1216 | Cut | Cut | Cut | Cut | Cut | Cut | Cut | Cut |
| 1536×1536 | Cut | Cut | Cut | Cut | Cut | Cut | Cut | Cut |

**Notes.** Eight GPUs at 608 are still about 13× slower than one PyTorch GPU. At 960, setup/checks time out near GPU memory capacity; this is not a measured inference duration.

## Modified remora-copy → Futhark

The modified YOLOv4 Remora source from remora-copy, compiled through Futhark to CUDA. “Modified” refers to the Remora program, not the Futhark compiler.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | 2.08 | 2.07 | 4.15 | 4.14 | 8.30 | 8.28 | 16.60 | 16.56 |
| 960×960 | 0.82 | 0.82 | 1.64 | 1.63 | 3.26 | 3.27 | 6.51 | 6.53 |
| 1216×1216 | Cut | Cut | Cut | Cut | Cut | Cut | Cut | Cut |
| 1536×1536 | Cut | Cut | Cut | Cut | Cut | Cut | Cut | Cut |

**Notes.** Roughly doubles original-source throughput at 608 and successfully runs 960. Still substantially slower than PyTorch; larger tests were cut.

## Darknet + dense decode

Native Darknet CUDA, built without cuDNN, invoked through Python/ctypes. Dense decoding matches the other paths’ output before thresholding/NMS; the wrapper includes host copies and exposes only transfer-inclusive timing.

| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 608×608 | — | 102.65 | — | 203.34 | — | 385.51 | — | 691.63 |
| 960×960 | — | Fail | — | Skip | — | Skip | — | Skip |
| 1216×1216 | — | Fail | — | Skip | — | Skip | — | Skip |
| 1536×1536 | — | Fail | — | Skip | — | Skip | — | Skip |

**Notes.** Leads the transfer-inclusive comparison at 608, but larger sizes fail the chosen numerical tolerance. This is not a standalone optimized C++/cuDNN baseline.

## Across implementations: throughput and image size

**One GPU, transfer-inclusive.** Cells show **images/sec (megapixels/sec)**. Megapixels/sec = images/sec × width × height ÷ 1,000,000. If runtime were proportional to image area, this value would stay constant across sizes.

| Implementation | 608×608 | 960×960 | 1216×1216 | 1536×1536 | 608: seconds / 128 images |
|---|---:|---:|---:|---:|---:|
| Remora → PyTorch | 88.27 (32.63) | 49.57 (45.68) | 32.05 (47.39) | 11.89 (28.05) | 1.45 |
| Direct PyTorch | 87.88 (32.49) | 49.55 (45.66) | 32.06 (47.40) | 11.86 (27.97) | 1.46 |
| Remora → tinygrad baseline | 6.42 (2.37) | Skip | Skip | Skip | 19.93 |
| Direct tinygrad baseline | 6.37 (2.35) | Skip | Skip | Skip | 20.11 |
| Remora → tinygrad im2col | 23.84 (8.81) | 11.70 (10.78) | 10.08 (14.90) | 5.30 (12.51) | 5.37 |
| Direct tinygrad im2col | 23.73 (8.77) | 11.63 (10.72) | 10.07 (14.89) | 5.34 (12.60) | 5.39 |
| Original Remora → Futhark | 1.02 (0.38) | Skip | Cut | Cut | 125.93 |
| Modified remora-copy → Futhark | 2.07 (0.77) | 0.82 (0.75) | Cut | Cut | 61.77 |
| Darknet + dense decode | 102.65 (37.94) | Fail | Fail | Fail | 1.25 |

**Notes.** Generated and direct Python paths closely match. PyTorch beats optimized tinygrad; Darknet leads at 608 only. Megapixels/sec varies with size, so runtime is not simply proportional to pixel count. Failed and skipped cells are not included in a ranking.

## Memory

Transfer-inclusive, separately sampled after warmup. MiB below is the **maximum per-GPU process footprint**; the 8-GPU column is not the total across eight GPUs. The CSV also contains simultaneous totals and allocator counters. Each worker replays for at least three seconds. Sampling can miss short peaks and excludes setup.

| Implementation | Side | 1 GPU peak MiB | 8 GPUs: max per-GPU MiB |
|---|---:|---:|---:|
| Remora → PyTorch | 608 | 12,486 | 12,486 |
| Remora → PyTorch | 960 | 28,526 | 28,526 |
| Remora → PyTorch | 1216 | 78,290 | 78,290 |
| Remora → PyTorch | 1536 | 45,222 | 45,222 |
| Direct PyTorch | 608 | 12,468 | 12,468 |
| Direct PyTorch | 960 | 28,696 | 28,696 |
| Direct PyTorch | 1216 | 78,258 | 78,258 |
| Direct PyTorch | 1536 | 45,184 | 45,184 |
| Remora → tinygrad baseline | 608 | 918 | 918 |
| Direct tinygrad baseline | 608 | 918 | 918 |
| Remora → tinygrad im2col | 608 | 2,810 | 2,810 |
| Remora → tinygrad im2col | 960 | 5,200 | 5,200 |
| Remora → tinygrad im2col | 1216 | 7,632 | 7,632 |
| Remora → tinygrad im2col | 1536 | 11,384 | 11,384 |
| Direct tinygrad im2col | 608 | 2,810 | 2,810 |
| Direct tinygrad im2col | 960 | 5,200 | 5,200 |
| Direct tinygrad im2col | 1216 | 7,632 | 7,632 |
| Direct tinygrad im2col | 1536 | 11,384 | 11,384 |
| Darknet + dense decode | 608 | 1,974 | 1,974 |

**Notes.** PyTorch’s 1216 memory spike is not a smooth pixel-count trend; its reserved allocator counters show the same pattern. The cause remains unresolved. Tinygrad im2col memory grows smoothly with size.

**Live allocation** counts buffers the backend considers active; process footprint also includes reserved pools and other CUDA memory. Lower live allocation can therefore leave footprint unchanged. It helps compare variants within one backend, but our counters differ: Futhark includes setup, PyTorch trial counters miss some CUDA-graph temporary liveness, and tinygrad/Darknet have no equivalent here.

No new Futhark memory tests ran after the cutoff. Earlier 608 measurements are in [RENTAL_CHECKS.md](../RENTAL_CHECKS.md).

## Measurement details and files

PyTorch 2.8.0+cu128; tinygrad 0.14.0; CUDA 12.8. Darknet fork: 58bb1326f8aefe19805b33901101697de5f09336, without cuDNN. Timed Python uses FP32; validation uses an independent FP64 reference and the fixed bound abs(error) ≤ 2e-5 + 2e-4 × abs(reference). Memory trials start with eight total requests, adding replays to reach three seconds per worker. Sampling can miss short peaks; no memory-run throughput is reported.

The broad partition sweep and additional DARPA adapters were not completed. Small timing differences between generated and direct Python are not established speed advantages.

[Timing CSV](timing.csv) · [Memory CSV](memory.csv) · [Cross-implementation CSV](cross-implementation.csv) · [Raw results](../results/full-comparison-20260911/)

## Conclusions: patterns and exceptions

- **Image size:** runtime increased in every measured size comparison, but not proportionally to pixels. From 608→1536, pixels grow 6.38×; single-GPU resident runtime grows 4.09× for Remora/PyTorch and 4.38× for optimized tinygrad.
- **Size-dependent exceptions:** optimized tinygrad takes only 1.22× longer from 960→1216, then 1.83× longer from 1216→1536, despite both steps adding about 1.60× pixels. Kernel-level causes remain untested.
- **Transfers:** PyTorch’s 1216→1536 resident runtime grows 1.57×, but transfer-inclusive runtime grows 2.70×. Upload/download and host-copy costs need separate investigation.
- **GPU count:** more GPUs helped every measured case. Resident 1→8 speedups are 7.88–8.00×; Darknet transfer-inclusive reaches only 6.74×. More throughput does not imply lower single-image latency or better cost efficiency.
- **Memory:** PyTorch peaks near 78 GB at 1216, then drops near 45 GB at 1536; its allocator behavior remains unexplained. Replication leaves per-GPU memory unchanged, so more GPUs do not solve one image’s memory demand.
- **Correctness:** passing 608 did not predict passing larger sizes for baseline tinygrad or Darknet. These numerical mismatches do not establish worse detection accuracy.
- **Backend choice:** generated Python matches direct Python. Good GPU scaling does not rescue Futhark’s slow per-image execution in this workload.

These conclusions cover uniform, batch-one replicas on this machine. Model/tensor partitioning and real-image accuracy remain untested.
