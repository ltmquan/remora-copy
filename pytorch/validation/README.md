# YOLO correctness validation

**Follow-up:** [The native optimization](../performance/NATIVE.md) now passes
full-network output validation. The timeout below describes the earlier general
backend and is retained as historical evidence.

The CPU differential suite passes: **14 cases × 3 fixtures × 2 compiler modes =
84 comparisons**, with zero shape, finite-value, or tolerance failures. There
are 16,734 scalar comparisons across both modes. Maximum absolute difference
from the Remora interpreter is **1.1920928955078125e-7**. The maximum error uses
0.1481% of the allowed `1e-5 + 1e-4 * abs(reference)` tolerance.

These are small typed instances of the helper definitions extracted verbatim
from `remora-copy/examples/darknet/yolov4.remora`, not full 608×608 inference.
The source SHA-256 is
`334346d83578f49a52507fab11d0cfed0da269b10f20ab012ade4f6ca98a6e89`.
The real Remora parser discovers dependencies; unused function definitions are
omitted, while global value definitions are retained. Each fixture runs through
the actual Remora interpreter independently of the generated Python module.

| Cases | Auto-selected backend |
| --- | --- |
| Same-padding convolution, stride-2/remainder convolution, 1×1 convolution | General |
| Batch normalization, Mish/leaky-ReLU/sigmoid | Static tensor |
| Convolution + BN + Mish/leaky-ReLU, residual block | General |
| Cascaded spatial pyramid pooling | General |
| Nearest upsampling, route concatenation | Static tensor |
| Anchor decoding, detection head, composed conv/BN/Mish/SPP/upsample/head | General |

Fixtures include zeros, signed linearly spaced inputs, and seeded random signed
inputs, with positive variances and nonuniform anchors. Every case also runs
with forced general lowering. For the ten cases where auto selects general,
these modes generate identical code; they are not independent implementations.

Nine cases additionally compare the interpreter to a separate direct-PyTorch
reference: **27 comparisons pass**, maximum absolute difference
`9.5367431640625e-7`. In particular, cascaded Remora pooling agrees with separate
13×13, 9×9, and 5×5 PyTorch pools on these fixtures.

The final composed pipeline returns `[300, 7]` and exercises 2,100 output values
per fixture. It is a small composition, not the complete YOLO topology.

## Reproduce

From `remora-copy/pytorch`, using Python with PyTorch and NumPy:

```sh
export REMORA=/absolute/path/to/remora
python validation/yolo_differential.py --out .build/yolo-differential-final --timeout 90
python validation/check_device.py .build/yolo-differential-final \
  --device cuda --output .build/device-results.json
```

`cpu-results.json` contains compilation modes, source/module hashes, and all
comparison metrics. `.build/yolo-differential-final/` additionally retains the
generated modules, source fixtures, input arrays, interpreter stdout/stderr,
reference arrays, and compiled outputs. These larger artifacts are ignored by
git and can be regenerated with the command above.

The device runner refuses to proceed unless the CPU manifest passes. The Slurm
script requests one H200, two CPU cores, 16 GB host RAM, and ten minutes. It
reuses the existing cluster Python environment and reference assets read-only.
A full-network probe, when supplied, has an additional two-minute process limit.
No computation is performed on the login node.

## Compiler change and regression check

Pure literal global values can now use static lowering. Nonliteral global
initialization still falls back to general lowering to preserve effects.
Remora's specialization introduces nonliteral index-grid globals in the YOLO
convolution path, so the current compiler still selects general lowering there.
The regression suite after this change ran **186 tests: 184 passed and two CUDA
tests skipped on the local CPU host**. The new regression checks literal-global
backend selection; the existing effectful-global check also passes.

The complete YOLO source successfully compiles with the current auto mode,
selecting general lowering and accepting 542 parameters. Its generated hash is
recorded in `full-network-provenance.json`. Earlier performance results from
`experimental/python_backend` concern a separate YOLO-specific compiler and
must not be attributed to this compiler.

## GPU fixture results

All **84 CUDA comparisons pass**, with zero mismatches against the saved Remora
interpreter outputs. Maximum absolute difference is `2.2351741790771484e-7`;
the worst error consumes 0.6433% of the allowed tolerance. See
`gpu-results.json` for every comparison.

Execution used an NVIDIA H200 on Explorer node `d4055`, PyTorch `2.8.0+cu128`,
and Slurm job `10279463`. TF32 was disabled. The local CPU run used PyTorch
`2.14.0`; device checks therefore also exercise a different PyTorch version.

## Full-network limit and performance gate

The direct PyTorch implementation produced a finite full-network reference
using the existing YOLOv4 checkpoint. The generated general backend started
inference but did **not return an output before the probe's 120-second process
limit**, which includes setup and reference generation. Exit status was 124;
the saved stage is `compiled_inference`. This is an incomplete comparison,
not a numerical mismatch and not a measured inference latency.

Consequently, **full-network equivalence is unverified and no new three-way
performance comparison is reported**. The small CPU/CUDA tests establish
agreement only for their tested helpers, compositions, shapes, and fixtures.
The general backend's host-controlled tensor execution remains a practical
obstacle for this network; a scalable lowering needs validation before timings
can meaningfully compare this compiler, Futhark CUDA, and direct PyTorch.

The allocation finished after **2 minutes 53 seconds**, with Slurm exit status
`124:0` and peak host RSS `3790340K` (about 3.62 GiB). It is no longer running.
The `full-network-results.json`, `full-network-exit-status.txt`, and
`device-10279463.log` files preserve the incomplete probe and successful device
fixture evidence. Remote artifacts remain under
`/scratch/luu.qu/remora-copy/pytorch/yolo-correctness-20260911`.
