# Remora → Python → PyTorch / tinygrad (experimental)

See [RESULTS.md](RESULTS.md) for the completed H200 measurements and conclusions.

This backend lowers the **original** `examples/darknet/yolov4.remora`, without
Futhark. It uses Remora's normal parser and type checker, expands the source's
CSP/residual/PANet functions, and emits straight-line Python calls to tensor
operations. Both runtimes execute the same generated Python. `direct.py` is an
independent YOLOv4 implementation whose topology and parameter shapes come from
`yolov4.cfg`, and runs in both frameworks.

This is a **YOLO library-idiom prototype**, not a complete replacement backend
for arbitrary Remora. Convolution, SPP, upsampling, and detection idioms map to
native tensor operations. Their underlying library ASTs are checked against
`contracts.json`; redefining a recognized helper is rejected rather than
silently ignored. Generic rank-polymorphic array programs and arbitrary changed
library definitions require further lowering work. Composite functions are
expanded from source; the network is not substituted with a hard-coded model.
`compiler.py` consumes the real frontend's AST, not a regex-based network parser.

## Build

From the `remora-copy` repository root, using your built Remora binary
and reading the original benchmark source from the sibling `remora` checkout:

```sh
python3 experimental/python_backend/build.py \
  --remora /absolute/path/to/remora \
  --source ../remora/examples/darknet/yolov4.remora
cd experimental/python_backend
python3 -m unittest test_compiler test_topology -v
```

`build.py` invokes `remora dev --typecheck --json`, then `--parse --json`.
It never invokes Futhark. Generated artifacts include source/AST hashes,
parameter shapes, 150 operations, and 110 convolutions. The checked-in parser
snapshot allows compiler tests without installing Haskell dependencies.

## Run

Use a CUDA-enabled PyTorch installation compatible with the GPU driver,
tinygrad, and NumPy. Explorer's tested environment is recorded with results;
The tested primary packages are PyTorch 2.8.0+cu128, tinygrad 0.14.0, and
NumPy 2.5.3. `setup.sbatch` creates a clean environment on a CPU node.
CUDA 12.8 PyTorch wheels are needed for its 570 driver.

```sh
python benchmark.py --backend torch --model remora --output results/torch-remora-eager-synthetic
python benchmark.py --backend torch --model direct --output results/torch-direct-eager-synthetic
DEV=CUDA python benchmark.py --backend tinygrad --model remora --mode jit --output results/tinygrad-remora-jit-synthetic
DEV=CUDA python benchmark.py --backend tinygrad --model direct --mode jit --output results/tinygrad-direct-jit-synthetic
python compare.py results
```

Add `--weights /path/to/yolov4.weights` for trained parameters. The repository
contains a Git LFS pointer, not the checkpoint; the expected checkpoint SHA-256
is `e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562`.
The checkpoint comes from the [official Darknet YOLOv4 release](https://github.com/AlexeyAB/darknet/releases/tag/yolov4).

`--fold-bn` prepares convolution weights once, matching Darknet inference;
the default preserves the source's per-call folding operations. `--mode compile`
enables full-graph `torch.compile`; `--mode jit` enables TinyJit. These modes
report warm-up/compilation separately. Weights are fixed across timed calls;
tinygrad/JIT may hoist operations depending only on fixed weights.

`run.sbatch` runs one H200 in `gpu-short`, four CPU threads, 32 GiB host memory,
and a 45-minute limit. Individual benchmark cases also have timeouts. Dependency
installation runs in a separate CPU allocation, not a GPU allocation or the
login node. Adapt the scratch directory and Darknet path for another account.

## What is compared

- Batch 1, 608×608 image, FP32; PyTorch matmul and cuDNN TF32 explicitly disabled; tinygrad
  `ALLOW_TF32=0` explicitly set.
- Identical synthetic fixture matching the earlier C++ harness, plus the trained
  checkpoint using a deterministic image. Trained-weight numerical agreement
  does not measure COCO mAP or real-image detector quality.
- Full dense output: 22,743 rows × 85 attributes, anchor-major, fine-to-coarse.
  No image preprocessing, thresholding, or NMS.
- Setup, first eager inference, compilation/capture warm-ups, and steady-state
  calls are separate. Every timing waits for GPU completion.
- Python GPU-resident timing excludes input upload/output download. A separate
  transfer-inclusive timing includes image upload and dense output download.
- The native C++ Darknet API includes image upload, head download, and CPU dense
  decoding. It folds BN once at setup. This is the prior CUDA/cuBLAS reference,
  **cuDNN disabled**; PyTorch can use cuDNN. Report this implementation difference
  with speed comparisons, rather than calling it a pure language comparison.
- `compare.py` checks every scalar with `abs(diff) <= 1e-5 + 1e-4*abs(reference)`,
  reports failures and nonfinite values, and also compares Remora versus direct
  within each framework. It records failures instead of hiding them.
- Ten timed calls after three warm-ups are exploratory measurements from one
  allocation, not a statistical study across nodes and cluster loads.

Files: `compiler.py`/`build.py` (frontend and code generation), `runtime.py`
(tensor operations), `generated_yolov4.py` (Remora-derived model), `direct.py`
(independent model), `benchmark.py` (fixtures and timing), `compare.py`
(numerical validation), `darknet_benchmark.cpp` (native reference).

Relevant runtime documentation: [PyTorch CUDA precision and synchronization](https://docs.pytorch.org/docs/stable/notes/cuda.html),
[tinygrad tensor operations](https://docs.tinygrad.org/tensor/ops/),
[tinygrad CUDA runtime](https://docs.tinygrad.org/runtime/).

## Additional experiments

`tune.sbatch` compares setup-time BN folding in eager PyTorch and tinygrad
`BEAM=2` kernel search, with an eight-minute per-case timeout. `graphs.sbatch`
compares `torch.compile(mode="reduce-overhead")` with the same native reference.
Submit follow-up jobs with Slurm `--dependency=afterok:JOBID` to keep GPU use
sequential. `summarize.py evidence` produces timing tables from retained records.

The current harness separately warms fresh-image uploads because they can
trigger a new torch.compile alias guard. It also checks an all-zero replacement
image against an eager evaluation after timing, to verify capture consumes new
inputs. Initial experiment records predate these harness refinements; the
experiment report identifies those differences. No model operations or
precision settings changed during these timing refinements.

## Fresh Futhark-inclusive comparison

`build_futhark.py` compiles the current original Remora source to Futhark IR,
GPU memory IR, and a CUDA library using the workspace toolchain.
`prepare_futhark_harness.py` checks the generated ABI against the Python model
and generates GPU-resident and transfer-inclusive timing loops.
`futhark_setup.sbatch` builds the C executable on a CPU node using the CUDA
link-time driver stub and exports the same trained fixture in ABI order.
`fresh_comparison.sbatch` reruns native Darknet, fresh Remora/Futhark/CUDA,
compiled PyTorch (Remora/direct), and tinygrad JIT (Remora/direct) sequentially
on one H200. Submit the GPU job with `--dependency=afterok:CPU_BUILD_JOB_ID`.
The fixture fingerprint and source hashes are retained with the new evidence.

## Memory and modified-source comparison

`memory_comparison.sbatch` reruns the original and modified Futhark programs,
compiled PyTorch, tinygrad JIT, and native Darknet. It takes uninstrumented
latency measurements first, then launches separate processes through
`measure_memory.py`. The common memory metric is **sampled per-process GPU
footprint in MiB**, including context, libraries, allocator caches and setup
workspaces, excluding the extra replacement-input correctness check. The whole-process
peak (including that check) is retained separately. It is not interchangeable with live tensor allocation; the 20 ms
sampling request can miss transient peaks. Host RSS and raw samples are also
retained. Futhark additionally prints its own peak device-allocation report.

From the MOCHA workspace:

```sh
python3 remora-copy/experimental/python_backend/build_futhark.py \
  --source remora-copy/examples/darknet/yolov4.remora \
  --out remora-copy/experimental/python_backend/modified_futhark
python3 remora-copy/experimental/python_backend/prepare_futhark_harness.py \
  --out remora-copy/experimental/python_backend/modified_futhark
```

Upload the generated C/header, harness and input manifest, then submit
`memory_setup.sbatch` on a CPU node and `memory_comparison.sbatch` with an
`afterok` dependency. Both sources use the same trained fixture. The modified
source includes `reduce/zero` and cascaded 5×5 SPP, without modifying the
original Remora program.

## Tinygrad investigation

The earlier `BEAM=2` attempts used an explicit **480-second process timeout**.
Exit status 124 means the external timeout stopped the process before first
inference completed. BEAM is per-kernel schedule search; width 2 retains two
candidate schedules per round and does not bound the total compilation time.
Tinygrad also has a separate per-candidate compilation timeout (10 seconds by
default in the installed version). These are different limits.

`investigate_tinygrad.sbatch` records a per-kernel `DEBUG=2` diagnostic, tests
opt-in FP32 convolution variants, and logs BEAM entry/exit events under a
180-second diagnostic budget. Diagnostic latencies are not benchmark results.
`tinygrad_optimized.py` implements the experimental lowerings using tinygrad
operations only. Full-network trained output validation is required before
using any variant's speed result.

After choosing a validated variant, it can be selected directly:

```sh
DEV=CUDA BEAM=0 python benchmark.py --backend tinygrad --model remora \
  --mode jit --tinygrad-opt im2col --weights yolov4.weights \
  --output tiny-optimized/remora-trained
```

`--tinygrad-opt` accepts `baseline`, `fold`, `materialize`, `im2col`, and
`winograd`. The default remains the recorded baseline. Every experimental
variant folds fixed BN parameters once. `im2col` uses contiguous patches and
matrix multiplication; `materialize` adds convolution-output boundaries;
`winograd` selects tinygrad's built-in transform for eligible 3×3 convolutions.
