# Remora YOLOv4: GPU backend experiments

## Executive Summary

We start with the YOLOv4 implementation in the Remora repository (`examples/darknet/yolov4.remora`, commit `72510c69006b90d1d5bcd1c472a69212b89b3116`, August 17, 2026) and compile it through Futhark to generate CUDA code. We evaluate the Futhark-generated CUDA code against the Darknet C++ native implementation of YOLOv4. We optimized the Remora YOLOv4 code so that Futhark can generate more efficient CUDA code and the runtime performance doubled. The optimized Remora was ~50x slower than the native Darknet code.

We also consider the question of how much more we can optimize the Remora code. To that end, we developed a preliminary compiler that's good enough to compile the Remora YOLOv4 program to Python. This Python route tests whether existing high-level GPU tools (we use both PyTorch and tinygrad) can bring Remora close to native C++ performance, providing a practical baseline for future Remora compiler/backend and MLIR work. This led to a surprising result that using PyTorch, we were able to improve upon the performance of the Darknet C++ code. An important point is that the compiler from Remora to Python emits high-level PyTorch/tinygrad operations rather than translating each source line or function separately, allowing those frameworks to optimize execution. If we don't do this, the performance is much worse and this is an important point to keep in mind for future compilation/MLIR work. The experiments also showed that using Futhark resulted in significantly increased GPU memory usage. This should also be addressed in future work.

To test correctness, we used 7 images, including a real photo from the DARPA evaluation repo, plus black, white, random-noise, gradient, and checkerboard images. All experiments that have performance numbers associated with them were tested for correctness and produced almost identical results (details below).

## Pipelines

We ran experiments across nine pipelines. **Remora (OG)** is the original YOLOv4 source; **Remora (MD)** is the modified version (explained below). **tinygrad (OP)** denotes our optimized tinygrad implementation.

1. **Remora (OG) → Futhark:** original source compiled through Futhark.
2. **Remora (MD) → Futhark:** modified source compiled through Futhark.
3. **Remora (OG) → Python → PyTorch:** generated Python using PyTorch operations.
4. **Direct PyTorch:** independently written Python network; no Remora compilation.
5. **Remora (OG) → Python → tinygrad:** generated Python using tinygrad operations.
6. **Direct tinygrad:** independent Python network using tinygrad.
7. **Remora (OG) → Python → tinygrad (OP):** generated Python using the optimized operations.
8. **Direct tinygrad (OP):** independent Python network using the same optimized operations.
9. **Darknet C++:** native implementation of YOLOv4.

Remora (MD) makes two changes: dot products use `reduce/zero` with identity `0.0` and pooling reuses three successive 5×5 max pools instead of separate 5×5, 9×9, and 13×13 pools. The successive pools cover the same effective neighborhoods with the program’s padding and preserve output order. Reduction reassociation may change rounding. **All Remora → Python results use Remora (OG).** The OG/MD comparison measures both changes together. A teammate variant also changes both pooling reductions to `reduce/zero` while retaining separate windows. It measured 481.929 ms resident, 482.852 ms transfer-inclusive, and 30,050 MiB, effectively tied with Remora (MD), and produced bitwise-identical outputs on all seven inputs. We therefore retain the current Remora (MD) version ([comparison](python_backend/teammate_futhark/RESULTS.md)).

We optimized tinygrad because its initial performance was slower than expected. tinygrad (OP) packs convolution patches into contiguous matrices, multiplies them by the weights, and caches batch-normalization-folded weights. It uses more memory but runs faster. PyTorch uses compiled execution; this was not an exhaustive tuning study of either framework.

**All nine pipelines use CUDA to execute on NVIDIA GPUs.** PyTorch and tinygrad are higher-level interfaces to CUDA, not alternatives to GPU execution. Compilation and cached replay reduce repeated launch overhead; the tables omit those implementation details from pipeline names.

## Remora → Python compilation

The prototype reads the Remora AST, expands composite functions, and recognizes supported computations such as convolution with normalization and activation. It emits high-level PyTorch/tinygrad operations rather than translating each source line or function separately, allowing those frameworks to optimize execution.

```text
Remora (OG) → parser AST → recognized tensor computations
            → Python calls such as ops.conv(...)
            → runtime.py → PyTorch/tinygrad → CUDA GPU
```

Recognition currently uses known YOLO library calls checked against reviewed AST hashes; the compiler supports this specific YOLOv4 program, not arbitrary Remora code. We checked generated graph connectivity and parameter shapes against the independently constructed network, and compared full numerical outputs. All nine pipelines passed on the original synthetic input.

## What the output includes

All paths return dense candidate boxes and class scores. **NMS (non-maximum suppression)** removes lower-scoring overlapping detections; it is excluded from every pipeline. Darknet’s output is decoded into the same format as Remora’s before timing ends. This is network-plus-decoding performance, not a complete detection-service benchmark.

**cuDNN** is NVIDIA’s library of optimized neural-network operations. It is disabled for both Darknet C++ and PyTorch in the results below because controlled on/off tests were faster without it in this FP32 configuration. tinygrad and Futhark use their own CUDA implementations.

However, six additional inputs, including a real photo, exposed mismatches: Remora (OG) → Futhark failed the tolerance on all six, while other paths had smaller failures on some inputs. Generated and direct Python outputs were bitwise identical for each backend on all six. The mismatches against Darknet need investigation; correctness across inputs is not established. Applying confidence filtering (0.25) and class-wise NMS (IoU 0.45) to these six inputs kept the same candidates and classes across all nine pipelines, although scores and box coordinates still differed slightly. 

## Runs and measurements

**H200 comparison:** 608×608, batch one, official YOLOv4 weights, and a deterministic image. Each timed call processes one image. Every pipeline uses three rounds of 30 timed calls per measured mode, after five warmups in each round and mode. We report the median of the three round medians. All runs use the same image.

**R** = GPU-resident input/output. **T** = includes image upload and dense-output download, retaining weights. Setup, compilation, and warmup are excluded. Futhark managed-memory paging can still move data in R mode. Memory is sampled separately during setup/inference, excluding the extra correctness-check phase; it is the process footprint, including reserved memory, and sampling may miss short peaks.

Timed arithmetic is FP32, with TF32 disabled in the PyTorch/Darknet on/off tests. The output check uses `abs(error) ≤ 1e-5 + 1e-4 × abs(reference)` against Darknet. This does not measure real-image detection accuracy.

## H200 results

Times are milliseconds per image; lower is better. Darknet C++ and PyTorch have cuDNN disabled. “—” means not measured.

| Pipeline | R, ms | T, ms | GPU footprint, MiB |
|---|---:|---:|---:|
| Remora (OG) → Futhark | 911.004 | 911.595 | 30,044 |
| Remora (MD) → Futhark | 481.906 | 482.714 | 30,048 |
| Remora (OG) → Python → PyTorch | 7.973 | 8.862 | 2,156 |
| Direct PyTorch | 8.049 | 8.927 | 2,130 |
| Remora (OG) → Python → tinygrad | 152.237 | 156.294 | 920 |
| Direct tinygrad | 152.243 | 156.301 | 920 |
| Remora (OG) → Python → tinygrad (OP) | 36.487 | 40.778 | 2,812 |
| Direct tinygrad (OP) | 36.474 | 40.721 | 2,812 |
| Darknet C++ | — | 9.364 | 1,974 |

Notes: Remora (MD) improves Futhark runtime by about 1.89×. tinygrad (OP) improves GPU-resident runtime by about 4.17× over tinygrad, with roughly 3× the footprint. Generated Python closely matches direct Python. The cuDNN-off transfer-inclusive times are 8.862 ms for Remora (OG) → Python → PyTorch, 8.927 ms for direct PyTorch, and 9.364 ms for Darknet C++.

The Darknet and PyTorch controlled checks both found cuDNN-off faster. These results apply to this batch-one, FP32/TF32-off configuration, not all cuDNN workloads.
