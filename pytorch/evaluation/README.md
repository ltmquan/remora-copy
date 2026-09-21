# General-program evaluation, 2026-09-15

The compiler has a general lowering path, but **full Remora conformance and fast
execution of arbitrary programs have not been established**. The earlier full
YOLO result measured a reviewed-library optimization. It did not establish
coverage of DARPA-MOCHA's other applications.

## What was tested

- The regression suite after the indexing fix: **203 tests, 201 passed, two CUDA
  skips** locally. See `regressions.log`. All nine structural tests also pass in
  `structural-tests.log` (including later shadowing, routing and neighbor-mean additions).
  The existing corpus tests compare checked-in numeric expectations and the
  actual Remora interpreter, through automatic and forced general lowering.
- **Seven original examples**: convolution-1d, FFT, Life, matmult, row-col-sums,
  Rule 30, and N-body. All compile and agree with the interpreter on their
  checked-in inputs (`example-evidence/results.json`). N-body is supplied as an
  expression, matching its source format. All seven currently select `general`;
  this establishes correctness on these fixtures, not fast GPU execution.
- **Four parameterized kernel ports**, three seeds each, two sizes. Small
  fixtures agree with the Remora interpreter, ordered static lowering, and
  independent direct PyTorch expressions. Medium fixtures compare tensor/static/
  direct PyTorch; they are not interpreter runs. See `evidence/cpu-results.json`.
- **Fixed-degree graph neighbor mean**, three seeded fixtures including repeated
  neighbors, agrees with the interpreter and direct PyTorch (`neighbor-results.json`).
- **Routing arc costs**, three routes, agree exactly across the interpreter,
  compiled code, and direct integer indexing/summation (`route-results.json`).
- Rewrite tests cover renamed functions, changed dot-product bodies, shadowed
  addition, non-square GEMM, prefix broadcasting, vector reduction cells,
  lifted initializers, empty reductions, ordered subtraction/folds, live inputs,
  gradients, and `torch.compile(..., backend='aot_eager', fullgraph=True)`.

## DARPA-MOCHA mapping and gaps

This checkout has no `.remora` files. The applications are Python/framework or
C++ programs. Kernel ports below are newly written Remora, not existing complete
Remora applications. Source-file SHA256s are in `source-provenance.json`.

| Target | Evidence in this backend | Still unvalidated |
| --- | --- | --- |
| 3_yolov4 | Prior full network fixture, see `../performance/NATIVE.md` | Entire evaluation driver/data coverage; YOLOv8 |
| 15_transformers | Representative linear projection and RMS normalization math; model configuration references RMSNorm/linear | HF model execution, attention/KV cache, tokenization, training |
| 41_or-tools | Arc-cost accumulation corresponding to `vrp_problem.cc` | Vehicle-routing solver and its search |
| 46_pyg | Fixed-degree neighbor mean, a GraphSAGE aggregation component | Complete GCN/GraphSAGE/GAT operators, sampling, training |
| 49_pygod | Representative feature reconstruction L2 error, motivated by its GAE feature reconstruction objective | GAE training, detector-specific score normalization, complete nomination |
| 65_hnsw | Squared L2 distance matching the loop in `include/bruteforceGPU.h` | HNSW construction/traversal, top-k and full queries |

The transformer and reconstruction kernels express representative mathematics;
they are not claimed to be literal source translations of framework internals.
These measurements are **not DARPA full-application scores** or line coverage.

## General optimizations added

`--lowering tensor` recognizes typed additive reductions and pure dot products,
then lowers them to `torch.sum` and `torch.matmul`. A lifted dot-product callback
can become a whole matrix multiplication. Recognition uses operations and bound
variable identities, not YOLO function names or source hashes. `auto` now tries
native, tensor, static, general, in that order.

Floating sums/products may reassociate. Tests use finite float32 values and
explicit tolerances (`atol=rtol=2e-4` for the kernel sweep). Choose `static` or
`general` for ordered reductions. This pass does not prove arbitrary programs
are algebraically equivalent or optimize every way to write matrix multiplication.

Testing the existing examples exposed an indexing discrepancy. `index2d` now
matches `Interpreter/Intrinsics.hs`: select flattened address `n*i+j`, rejecting
out-of-range linear addresses. Convolution, Life and Rule 30 depend on this.
Futhark's implementation checks coordinates separately, so out-of-axis indices
can still differ between those two upstream execution paths.

## CPU timing

Two CPU threads; float32; medium shapes use 128 rows/features and 96 projection
outputs. Times are medians of ten synchronized invocations after three warmups,
with inputs already resident, no compilation/transfers, gradients disabled.
These short microbenchmarks are indicative, not statistically robust rankings.

| Kernel | Structural tensor (ms) | Ordered static (ms) | Direct PyTorch (ms) |
| --- | ---: | ---: | ---: |
| Squared L2 | 0.265 | 0.643 | 0.0107 |
| Linear projection | 0.0118 | 3.325 | 0.0103 |
| RMS normalization | 0.358 | 0.712 | 0.0181 |
| Reconstruction error | 0.230 | 0.722 | 0.0121 |

The projection is about 281× faster than static lowering and within 16% of
this direct PyTorch measurement. Other kernels retain substantial lifting and
Python overhead: this is a concrete remaining optimization target.

The original examples also record single-run CPU execution times (not warmed
benchmark medians) in `example-evidence/results.json`. In particular, N-body
currently takes about 3.10 seconds for its built-in small input. Correctness
coverage does not imply that this fallback is an efficient implementation.
No new Futhark timing comparison was run for these general kernels.

## GPU timing and correctness

Slurm job **10362199 completed in 19 seconds** on a Tesla V100-SXM2-32GB,
PyTorch 2.8.0+cu128. All 24 fixtures (four kernels × two sizes × three seeds)
agree with direct PyTorch and ordered static lowering on GPU. The Remora
interpreter comparisons were performed on CPU for the small fixtures.
Raw evidence: `evidence/cuda-results.json`.

| Medium kernel | Structural tensor (ms) | Ordered static (ms) | Direct PyTorch (ms) |
| --- | ---: | ---: | ---: |
| Squared L2 | 0.554 | 2.613 | 0.0410 |
| Linear projection | 0.0547 | 3.392 | 0.0287 |
| RMS normalization | 1.102 | 3.825 | 0.0779 |
| Reconstruction error | 0.747 | 4.162 | 0.176 |

These are synchronized host wall times for small resident tensors; Python and
launch overhead matter substantially. Projection improves about 62× over static,
but still takes about 1.9× the direct PyTorch time. None of these kernel times
represent complete application inference or training.

## Reproduce

From `remora-copy/pytorch`, with `REMORA` set to the built frontend:

```sh
python -m unittest discover -s tests
python evaluation/audit_examples.py
python evaluation/run.py
```

`evaluation/run.py --prepared --device cuda --out evaluation/evidence` executes
already generated programs without a frontend. `check.sbatch` requests one V100,
two CPU cores, 8 GB, and five minutes. Job 10362199 was submitted with a dependency
on the already-running job 10362144 to avoid an overlapping GPU allocation.
