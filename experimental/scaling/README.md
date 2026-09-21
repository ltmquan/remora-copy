# Larger inputs and multiple GPUs

Prepared under `remora-copy` only. Local frontend/topology checks pass; Linux builds and GPU smoke tests require the rental SSH connection. No rental GPU measurements yet.

The first executable experiment is whole-image data parallel inference: one model replica per GPU, processing independent requests. It compares Remora-generated Python and independent direct Python on PyTorch, baseline tinygrad, and im2col tinygrad, both Remora→Futhark source variants, and native Darknet with dense decode. This measures throughput, not acceleration of one image. Every request still computes the full dense YOLO output. Two deterministic input patterns alternate across unique request IDs; this is a compute benchmark, not dataset detection accuracy.

Dimensions: image sides 608, 960, 1216, 1536; 1/2/4/8 GPUs by default; resident or transfer-inclusive; uniform contiguous, round-robin, measured-speed-weighted, or dynamic chunk assignment. Strong scaling fixes the total number of requests; `--weak-scaling` fixes requests per GPU. The same two resident input buffers are held on every GPU for every policy, so dynamic scheduling does not get a different input-memory budget.

`prepare.py` changes spatial dimensions in copies of the original and modified Remora entry points, preserving helpers, weights and float anchors. It runs the actual Remora typechecker and generates Python from the original source AST. `--futhark` also emits CUDA library sources for both versions. Artifacts stay in this directory. Generated C needs compilation on the Linux rental.

```bash
python3 prepare.py --sizes 608 960 1216 1536
# Optional, slower: generate both Futhark variants too.
python3 prepare.py --sizes 608 960 --futhark
```

On the rented host, upload the `experimental/scaling` and `experimental/python_backend` directories, including prepared artifacts. Supply the official YOLOv4 checkpoint from the earlier experiment. Do not upload `.git`, caches, old evidence, or credentials. Run `bash bootstrap.sh` with Python 3.13 installed (`PYTHON_BIN` can specify its path); this creates a local venv with the earlier Torch/tinygrad versions. CUDA 12.8-compatible drivers are required. Record the machine using `preflight.py`; check GPU topology and other active processes before running.

```bash
.venv/bin/python model.py --size 608 --weights /path/to/yolov4.weights
.venv/bin/python run.py --implementation remora-torch --size 608 --gpus 0 1 \
  --weights /path/to/yolov4.weights --policy uniform --images 32 \
  --output results/smoke-2gpu.json
.venv/bin/python matrix.py --sizes 608 --counts 1 2 --implementations remora-torch direct-torch \
  --weights /path/to/yolov4.weights --out results/first-plan
# Inspect plan.json; add --execute to launch sequentially. Failures stop the sweep.
```

Generate a direct eager PyTorch reference for every size before running the matrix. Both input patterns must pass full-output FP32 checks (`atol=1e-5`, `rtol=1e-4`) after JIT warmup; failed validation produces no accepted timing. This reference is independent of the Remora-generated topology, but uses the same tensor primitives; a native Darknet reference is still needed as an additional check at new sizes.

Workers warm up separately, calibrate, then start each trial against a common monotonic deadline. Throughput uses the last completed worker, never the sum of worker rates. Setup includes compilation and validation and is reported separately. Explicit defaults: 1,800 seconds for setup, 600 seconds per trial. A timeout is a failed cell, not a runtime sample. The controller terminates workers on failure. Tinygrad BEAM defaults to zero; record any override.

Use a separate `--sample-memory` pass for process GPU footprint: per-GPU peaks and the simultaneous sum, sampled during warmed inference. Its timings are excluded from summary throughput. Short peaks may be missed. PyTorch live/reserved peaks reset per trial; Futhark live peaks cover the context lifetime, including setup. These scopes are recorded and must remain visible in comparisons. Tinygrad/Darknet live counters are null, not zero. Summing asynchronous per-GPU peaks is not the simultaneous machine peak.

Still needed on the rental: compile the prepared CUDA libraries and native wrapper, validate the new execution adapters on GPUs, and cross-check new sizes with native Darknet. Additional DARPA target adapters are not implemented yet. Tensor/model/spatial partitioning is also a separate phase; cutting a YOLO image into tiles without handling convolution halos and boundary semantics changes the computation.

# Prime Intellect

Use **On-Demand GPUs → Single-Node → GPU Count 2 or 4**. Single-node means one machine, not one GPU. Given the currently offered choices, 2× H100 is the first choice for memory headroom and continuity with H200; 4× A6000 provides more scaling points but only 48 GB per GPU. Confirm actual NVLink topology rather than assuming it from the GPU label. We will measure a fresh one-GPU baseline on the same rented host.

Availability lookup (does not rent anything):

```bash
prime availability list --gpu-count 2 --no-group-similar
```

After provisioning, share the SSH user/host/port or connection command. Keep private keys and API tokens on your machine. We can then upload, configure, and run the smoke test before a larger sweep. Export results before terminating an ephemeral instance; persistent storage depends on provider/datacenter compatibility.

Sources: [Prime availability](https://docs.primeintellect.ai/cli-reference/check-gpu-availability), [Prime provisioning](https://docs.primeintellect.ai/cli-reference/provision-gpu), [PyTorch DDP input sharding](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html), [tinygrad explicit sharding](https://docs.tinygrad.org/tensor/properties/).

# Size-scaling experiment on the eight-GPU rental

The full default plan contains 884 cells, so begin with a smoke test rather than launching it immediately. Stage 1: 608², uniform assignment, 1, 2, 4 and 8 GPUs across implementations. Stage 2: repeat at 960², 1216² and 1536². Stage 3: compare assignment policies, followed by separate memory passes. Large sizes may exceed H100 memory, especially original Futhark; record the failure and keep it visible rather than substituting a smaller input.

Image sizes are independent of global request count (default 128). All sizes retain FP32, the same trained weights and anchor dimensions. The output grows with image area: 22,743 / 56,700 / 90,972 / 145,152 rows, each with 85 values. Compare images/s, pixel-count ratio, throughput relative to 608², and multi-GPU speedup relative to the matching 1-GPU run. This avoids mistaking more work per image for worse implementation scaling.

On the rental, after `bootstrap.sh`:

```bash
.venv/bin/python get_weights.py
bash setup_darknet.sh
for side in 608 960 1216 1536; do
  .venv/bin/python model.py --size "$side" --weights artifacts/yolov4.weights
done
.venv/bin/python matrix.py --sizes 608 --policies uniform --weights artifacts/yolov4.weights --out results/smoke-plan
# After reviewing the smoke plan, rerun with --execute.
.venv/bin/python summarize.py results/smoke-plan --output results/smoke.csv
```

`setup_darknet.sh` builds the pinned native implementation with CUDA and cuDNN disabled, matching the prior baseline's library choice. This is not a claim about maximum achievable cuDNN-enabled Darknet performance. It also builds the prepared Futhark libraries. The host needs the CUDA development toolkit, make and GCC/G++; these were verified on the rental. The source archive is the project Darknet fork, not the upstream repository.

`package.py --output /tmp/remora-h100.tar.gz` bundles code and generated model artifacts with a SHA256 manifest; it excludes weights, old measurements, environments and credentials. `preflight.py` captures GPU UUIDs, topology, driver/runtime versions and running processes. No instance provisioning or GPU execution is performed locally.

Every cell uses images of one fixed size, so homogeneous GPUs may show little benefit from weighted or dynamic assignment. These policies establish a baseline for later heterogeneous request sizes; there is no assumption that more scheduling complexity is faster. Actual CUDA UUIDs must differ across workers or the run fails.

Futhark's existing CUDA configuration may use managed memory. At large sizes, host paging can turn an apparent capacity problem into a severe slowdown rather than an immediate allocation error. Record host RAM and GPU footprint along with runtime; do not interpret successful allocation as proof that the working set fits in GPU memory.

For the current eight-GPU rental, use the updated [handoff](HANDOFF.md). The earlier Prime discussion above describes the previously considered offers; the current host has eight H100 80 GB GPUs. Set `--gpu-order` after inspecting topology. The native build discovers CUDA compute capabilities on the new host rather than assuming sm_90.

```bash
# Quick correctness/dispatch smoke before longer measurements:
.venv/bin/python matrix.py --sizes 608 --counts 1 2 4 8 --images 16   --implementations direct-torch remora-torch --policies uniform   --weights artifacts/yolov4.weights --out results/eight-gpu-smoke
# First scaling sweep, still dry-run unless --execute is supplied:
.venv/bin/python matrix.py --policies uniform --weights artifacts/yolov4.weights   --out results/eight-gpu-size-sweep
```

The H100 zero-image stress check exposed small cross-backend rounding differences: native Darknet exceeded the original `1e-5 + 1e-4*abs(reference)` bound on 42 of 1,933,155 values; the ramp image passed. An FP64 diagnostic found Darknet's largest normalized zero-image error was 1.572 times that bound, versus 0.888 for PyTorch. For cross-backend runs we can explicitly use `--atol 2e-5 --rtol 2e-4`, applied uniformly across implementations. Defaults remain strict. Every worker records both the selected tolerance and violation counts under the original strict bound, so this difference remains visible. GPU execution stays FP32; FP64 is diagnostic only.

On the rental, the remaining cross-backend validation uses `model.py --precision float64 --reference-dir artifacts/references-fp64` and passes that same directory to the matrix. These more accurate reference outputs are generated separately from timed FP32 execution. The earlier FP32-reference smoke results remain preserved, including failures. Summary speedups only match runs using the same reference definition.

All scalable backends use the same Python process scheduler. The Darknet row runs the native C++/CUDA implementation through a thin ctypes wrapper (with full dense decode and result copying), and Futhark uses its generated C library through ctypes. These are explicitly host-harness-inclusive measurements; do not label them as a timing of a standalone C++ executable.

Futhark now uses its runtime PTX cache, with separate files for each model size/source variant and visible GPU selection. The runtime checks source and compilation options before accepting the cache. Results record whether a candidate cache file existed before context creation; this is not a claim that the runtime accepted it. Compilation is still excluded from inference timing and included in setup time.

Memory passes replay the workload for at least three seconds per worker (`--memory-min-seconds` in run.py) so fast GPU cases produce multiple samples. Extra memory-only replays are recorded separately and do not count as scheduled request IDs. Memory runs expose no throughput value, and insufficient samples or missing GPU processes cause failure. This improves sampling coverage but still cannot guarantee capture of every brief allocation peak.
