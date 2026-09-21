# Remora → PyTorch compiler

Compile Remora to Python/PyTorch without generating Futhark IR. This backend
implements lowering for checked expressions, closures, type/index polymorphism,
function arrays, existential boxes, runtime shapes, records and intrinsics.
**This is not a claim of complete language conformance or competitive performance
on every program.** See [general-program evaluation](evaluation/README.md) for
measured coverage, compatibility findings, and remaining application gaps.
All implementation changes are isolated in `remora-copy/pytorch`.

Four lowering paths serve different programs, with native PyTorch operations preferred:

| Mode | Frontend and generated code | Execution |
| --- | --- | --- |
| `native` | Reviewed library definitions → native convolution, pooling, concatenation, upsampling and decoding | CPU/CUDA, autograd, eligible for `torch.compile` |
| `tensor` | Structural recognition of additive reductions and dot/matrix products, independent of source names | Native `torch.sum` / `torch.matmul`, CPU/CUDA and autograd |
| `static` | Existing specialization/defunctionalization → tensor operations and `vmap` | CPU/CUDA, autograd, eligible for full-graph `torch.compile` |
| `general` | Type checking → generated Python closures and PyTorch operations | Higher-order values, runtime shapes, effects; Python controls execution |
| `auto` (default) | Try `native`, then `tensor`, then `static`, then `general` | Prefers native tensor operations and retains language coverage |

The general path **does not interpret a serialized AST** or call the Remora
interpreter. Source constructs become generated Python functions, bindings,
type/index expressions and calls to an embedded value runtime. It bypasses the
frontend passes that reject some valid higher-order/existential programs.

Language coverage and GPU optimization are different: dynamic shape values,
heterogeneous closures and file effects cannot all be one static GPU graph.
General lowering can operate on CUDA tensors, but may execute host loops and
synchronize to obtain runtime dimensions or select a function/box. It is a
correctness path, not a claim of competitive full-network GPU performance.

## Install and compile

From this directory, with Python 3.10+ and a suitable PyTorch installation:

```sh
python -m pip install -e .
export REMORA=/absolute/path/to/your/built/remora
mkdir -p .build
remora-torch -f examples/matmul.remora -o .build/matmul.py
python .build/matmul.py '[[1,2,3],[4,5,6]]' \
  '[[1,2,3,4],[5,6,7,8],[9,10,11,12]]'
# [[38.0, 44.0, 50.0, 56.0], [83.0, 98.0, 113.0, 128.0]]
```

Without installing the package, use `python -m remora_torch` from this directory.
Code generation imports only standard-library modules; execution requires
PyTorch. `--remora PATH` overrides `REMORA` and the default `remora` on `PATH`.
The frontend binary must support `dev --typecheck --json` and, for static
lowering, `dev --defunctionalize --json`.

```sh
# Runtime shape, automatic general lowering:
python -m remora_torch -f examples/dynamic.remora -o .build/dynamic.py
python .build/dynamic.py 6
# 15

# Closures in an array:
python -m remora_torch --lowering general \
  -f examples/higher_order.remora -o .build/higher_order.py
python .build/higher_order.py 10 '[[1,2,3],[4,5,6]]'
# [[11.0, 12.0, 13.0], [40.0, 50.0, 60.0]]

# An expression:
python -m remora_torch -e '(+ [10 20] [[1 2 3] [4 5 6]])' -o .build/prefix.py
python .build/prefix.py
# [[11, 12, 13], [24, 25, 26]]
```

`--lowering native` requires a supported native tensor graph. Every accepted
program is checked by the Remora typechecker before source is returned.
`--lowering static` fails if the tensor path cannot represent a program;
`--lowering general` directly compiles the checked AST. Generated modules expose
`LOWERING` so callers can inspect the selected mode. `-o` replaces output
atomically only after successful compilation; omitting it writes Python to stdout.
The frontend timeout applies to each invoked frontend stage.

The native optimizer follows the experimental YOLO compiler's reviewed-library
approach. It checks AST hashes and transitive dependencies, expands composite
functions from their actual source, and rejects shadowed dependencies or altered
helper definitions. It currently targets the reviewed YOLO tensor library;
arbitrary equivalent user functions are not recognized by that path. The separate
`tensor` path recognizes typed sums and dot/matrix products structurally, without
YOLO names or hashes. It does not yet recognize arbitrary equivalent algorithms.

Native and structural tensor floating-point operations use PyTorch numerical behavior, including
reassociated reductions. They are validated with tolerances on finite float32
inputs, not promised to be bitwise identical to ordered Remora reductions or to
preserve every nonfinite-value edge case. Select `general` for ordered
interpreter-style execution. Native folded weights are never cached: parameters
remain live and differentiable, including after in-place updates.

No Futhark executable is invoked. The existing Haskell frontend still has its
existing build dependencies, including the Futhark Haskell library.

## Python API, devices and gradients

Each source entry is in `module.ENTRIES['source-name']`. Python-safe names that
do not shadow runtime names also have direct wrappers such as `module.main`.
Arguments are positional, with optional keyword-only `device`.

```python
import sys
sys.path.insert(0, '.build')
import matmul
import torch

a = torch.randn(2, 3, device='cuda', requires_grad=True)
b = torch.randn(3, 4, device='cuda', requires_grad=True)
c = matmul.main(a, b)
c.sum().backward()

# For a module produced by static lowering:
compiled = torch.compile(matmul.main, fullgraph=True)
c = compiled(a, b)
```

Use CPU tensors without CUDA. Devices are inferred from tensor arguments,
including tensors inside boxes and records. Mixed devices are rejected unless
an explicit `device` requests a transfer. No-argument entries default to CPU;
use `main(device='cuda')` to create constants on CUDA.

Python numbers/lists become tensors of the declared type. Tensor arguments must
have the declared dtype and shape; no silent dtype conversion occurs. Floating
operations preserve autograd through executed tensor operations, including
boxed values and closures. Shape witnesses are discrete control data and have
no gradient.

General modules also expose `Box(index, value)` and `ObjectArray(shape, values)`.
A dimension witness is an integer; a shape witness is a tuple. Object arrays
store a flat row-major list and retain their shape even when empty. Records are
Python dictionaries. Generated scalar closures are callable; value closures
accept the declared parameter type and use their creation device. Type closures
accept an atom descriptor such as `'Float'`; index closures accept a dimension
or shape. Function parameters can also receive Python callables whose inputs
and outputs follow the declared Remora types.

For example, an entry with argument `(Sigma ($n) [Float $n])` accepts
`module.Box(3, torch.tensor([1., 2., 3.]))`. Witness/data mismatches are rejected.
Boxes and object arrays can pass between generated modules.

Generated program CLI arguments use **JSON**, not Remora value syntax. General
modules encode boxes as `{"box": witness, "value": data}` and object arrays as
`{"shape": [...], "values": [...]}`; the same forms are accepted as inputs.
Static modules provide `--compile` for `torch.compile(fullgraph=True)`.
General modules do not promise full-graph capture and have no `--compile` flag.

## Semantics

- Remora frames agree by **prefix**. `[2]` lifted against `[2,3]` broadcasts by
  row, rather than using PyTorch's normal trailing-axis broadcasting.
- The general compiler handles every checked expression/atom/binding variant:
  value/type/index abstraction and application, arrays/frames, local and global
  bindings, records/projection, boxes/unboxing and entries. Imports remain the
  existing frontend's responsibility.
- All intrinsics currently declared in `src/Intrinsics.hs` are lowered,
  including static/dynamic iota, reification, structural array operations,
  indexing, reductions/folds, scalar operations, tracing and file reads.
- `Int` is `torch.int64` (the interpreter's machine Int on a 64-bit host),
  `Float` is `torch.float32`, and `Bool` is `torch.bool`. This differs from the
  Futhark backend's i32 ABI. Floating rounding can differ across devices and
  compilers.
- Folds and reductions retain source order, including non-associative operators
  and non-identity initial values. `sum` uses a native integer tensor reduction.
- General lowering evaluates global values in declaration order on each entry
  invocation, preserving effects even for an otherwise unused value. Functions
  and entry bodies remain delayed until called.
- `undefined` compiles and raises when evaluated. Out-of-bounds indices,
  negative dimensions, invalid input types and nonrectangular unboxed results
  raise errors. Boxes permit varying shapes; ordinary arrays must be rectangular.
- Empty existential arrays cannot provide a witness for a result dimension that
  depends on that witness. Such an underdetermined result reports an error;
  fixed result shapes and empty function arrays are supported.

There is no predefined-network substitution or fallback to the Remora
interpreter. `auto` selects another compiler lowering, not an interpreter.

## File intrinsics and deployment

Generated numerical/higher-order programs embed their runtime and need only
PyTorch. Binary reads (`read-file-f32bin`, little-endian float32), `trace` and
`trace-file` also use the embedded runtime and ordinary host file operations.
Relative filenames are resolved against the execution working directory.

Remora's **`read-file` accepts arbitrary Remora expressions**, not just data
literals. Supporting that behavior requires a runtime compiler: install
`remora-torch` and set `REMORA` on the execution host. The file's expression is
compiled and executed on the requested device, and its result is checked against
the declared type/shape. Source text is never evaluated directly as Python.
Only programs using this intrinsic have this additional deployment dependency.

## Validation and remaining work

```sh
REMORA=/absolute/path/to/remora python -m unittest discover -s tests -v
# Also check real CPU Inductor forward/backward:
REMORA_TORCH_TEST_INDUCTOR=1 REMORA=/absolute/path/to/remora \
  python -m unittest discover -s tests -v
```

The general path compiles all **85 type-correct files** in the existing 91-file
corpus, including the library without entries. The remaining six are intentional
frontend-rejected type errors. Tests compare numeric fixtures against both
checked-in expectations and the Remora interpreter, through auto and forced
general lowering. Additional tests cover higher-order/existential behavior,
I/O, global effects, external value validation, autograd, and static graph capture.
The final local run passed 183 tests, with 2 CUDA tests skipped.
See [VALIDATION.md](VALIDATION.md) for the recorded results.

This is coverage of the repository's current language implementation and tests,
not a formal proof for every well-typed program. CUDA execution and performance
remain unverified locally. No native matmul/convolution pattern optimizations
have been added: generic convolution still follows the source's indexing and
dot products, and ordered reductions may produce large graphs. The separate
YOLO-specific prototype's measurements do not apply to this compiler.

## Files

- `compiler.py`: original tensor lowering and automatic frontend selection.
- `general.py`: checked-AST source generation for higher-order/dynamic programs.
- `runtime.py`: embedded tensor lifting and static program interface.
- `general_runtime.py`: closures, boxed/object arrays, dynamic operations and I/O.
- `__main__.py`: compiler CLI; all these modules are under `remora_torch/`.
- `examples/`: matrix multiplication, prefix lifting, runtime shapes and closures.
- `tests/`: differential corpus, runtime, effects and gradient tests.

PyTorch references: [vmap](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html)
and [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile).

## YOLO differential correctness

The exact YOLO helper definitions have been compared against the Remora
interpreter on 42 CPU fixtures in auto and general modes: all 84 comparisons
pass. See [the validation report](validation/README.md) for scope, numerical
errors, backend selection, GPU results, and reproduction commands. Small helper
agreement does not establish full-network correctness or performance.

## GPU performance

[Measured V100 results](performance/README.md) compare the current compiler,
Futhark CUDA, and direct PyTorch on small YOLO workloads, plus full-network
Futhark/direct baselines. Those measurements describe the original general backend. The new native
optimizer removes the per-cell Python work for recognized tensor functions; see
[the native optimization report](performance/NATIVE.md).

`index2d` follows the reference interpreter: flatten the matrix and select
`n*i+j`, checking the resulting linear address. Several bundled examples rely
on crossing a row boundary. This differs from the Futhark backend’s per-axis
bounds checks for coordinates outside an individual dimension.
