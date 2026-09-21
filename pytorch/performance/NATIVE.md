# Native PyTorch optimization

The automatic compiler now prioritizes native PyTorch operations. Full YOLO
runs in **26.056 ms**, versus **23.075 ms** for direct PyTorch and **1015.581 ms**
for Remora→Futhark CUDA on the same V100 allocation. This is about **39× faster
than Futhark** and **13% slower than direct PyTorch** in this run.

The native full-network output is **bit-for-bit equal to direct PyTorch** over
**1,933,155 float32 values** for the tested checkpoint/image. A zero image changes
the output, confirming that inputs remain live. This is evidence for that
fixture, not a promise of bitwise agreement for every input or device.

## What changed

The optimizer adapts the reviewed-library approach from
`remora-copy/experimental/python_backend/compiler.py`. It checks parsed AST
hashes and transitive dependencies before replacing library functions with
native operations, and expands composite functions from their actual source.
The recognized full network becomes 110 `conv2d` calls plus native pooling,
concatenation, upsampling, residual addition, BN folding, and decoding.

The pipeline is now `auto: native → static → general`. Unsupported idioms,
changed definitions, higher-order arguments, nonliteral global initialization,
and incompatible cell shapes retain the existing fallbacks. Explicit
`--lowering native` reports an error instead of falling back. Accepted native
programs still pass the Remora typechecker. No runtime AST interpreter or
hard-coded network output is used.

Parameters are not cached or detached. Batch-normalization folding remains
part of the differentiable graph and responds to in-place parameter updates.
Generated modules are standalone PyTorch and can be captured by `torch.compile`;
the small full-graph test uses `aot_eager`, while timings below are **eager**.
Full-network Inductor optimization has not been measured here.

Native float32 operations permit PyTorch reduction reassociation. Contract
matching protects source definitions, but does not make floating-point
reassociation exact or promise all nonfinite-value edge cases. General lowering
remains available for ordered interpreter-style execution. Recognition currently
covers the reviewed YOLO tensor library, not arbitrary equivalent user code.

## Latency

Tesla V100-SXM2-32GB, node `d1027`, UUID
`GPU-ac9a87f5-69c3-a1e1-d83d-5193220f7809`, driver `545.23.08`.
PyTorch `2.8.0+cu128`; Futhark CUDA linked with CUDA 12.3. Inputs are GPU-resident;
median of ten synchronized host-wall-time measurements after three warm-ups.
Compilation, startup, and validation are excluded. All times are milliseconds.

| Workload | Previous general backend | Native backend | Direct PyTorch, same run | Futhark, same run |
| --- | ---: | ---: | ---: | ---: |
| conv_same | 152.084 | 0.060 | 0.041 | 0.014 |
| spp_cascade | 1178.158 | 0.153 | 0.134 | 0.091 |
| composed_pipeline | 3279.070 | 1.398 | 1.201 | Build limitation |
| Full YOLO, 608×608 | Did not finish previous probe | 26.056 | 23.075 | 1015.581 |

Previous general-backend measurements used a different V100 of the same model,
node `d1022`. Native/direct/Futhark columns were measured together on `d1027`.
The small inputs are convolution `[2,4,4]`, pooling `[2,5,5]`, and a composed
pipeline from `[1,5,5]` to `[300,7]`; they are not full-YOLO throughput estimates.
The existing Futhark compiler still cannot generate CUDA for the small composed
case, as documented in the earlier performance report.

Full native first-inference latency was **2.668 seconds**, including cuDNN
initialization/tuning; steady-state latency is the 26.056 ms above.

## Validation and evidence

- All **42 CPU interpreter fixtures** and **42 CUDA replays** pass, including
  33 fixtures using native lowering and nine using static/general fallbacks.
- Full YOLO native versus direct: exact agreement over 1,933,155 values.
- Futhark versus direct full YOLO: passes `atol=1e-5, rtol=1e-4`.
- Regression suite: **193 tests, 191 passed and two CUDA tests skipped locally**.
  An additional duplicate-parameter guard test also passes.
- Tests cover changed/missing/shadowed dependencies, effectful globals, invalid
  source types, parameter updates, autograd through BN folding, finite-difference
  gradient validation, and a small full-graph `torch.compile` invocation.
- Wheel packaging was checked to include native code and the contract manifest.

See [native full results](native-evidence/native-full-results.json),
[byte-for-byte output check](native-evidence/bitwise-check.json),
[small timings](native-evidence/results.json),
[CPU comparisons](native-evidence/cpu-results.json),
[CUDA comparisons](native-evidence/device-results.json), and
[regression log](native-evidence/regressions.log). The raw allocation log and
baseline reports are alongside these files. The baseline-stage JSON's old
“not timed” note predates the subsequent native stage; the final native result
is in `native-full-results.json`.

The current generator reproduces the exact generated-module hash recorded in
the GPU results. The source SHA-256 is
`334346d83578f49a52507fab11d0cfed0da269b10f20ab012ade4f6ca98a6e89`.
Slurm job `10304988` completed successfully, exit `0:0`, in 2 minutes 52 seconds.
No allocation remains running. New local work stays in `remora-copy/pytorch`.

## Use

From `remora-copy/pytorch`:

```sh
export REMORA=/absolute/path/to/remora
python -m remora_torch -f ../examples/darknet/yolov4.remora -o .build/yolov4.py
# Or require native lowering explicitly:
python -m remora_torch --lowering native -f ../examples/darknet/yolov4.remora -o .build/yolov4.py
```

The module exposes `main`, `ENTRIES`, `LOWERING`, `NATIVE_OPS`, and
`NATIVE_CONTRACTS`. `validation/native_validation.py` regenerates auto-lowered
modules and replays the saved interpreter fixtures. `performance/native_run.sbatch`
runs device validation, small benchmarks, full baselines, and `native_full.py`
in an isolated cluster directory. Remote evidence is under
`/scratch/luu.qu/remora-copy/pytorch/native-performance-20260912`.
