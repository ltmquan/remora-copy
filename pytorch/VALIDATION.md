# Validation record — general language lowering

Validated on September 11, 2026, on macOS arm64 with Python 3.13 and PyTorch
2.14.0. CPU execution was available; CUDA was not.

**Final result: 185 tests run; 183 passed, 2 skipped because CUDA is unavailable.**

## Reproduce

From this directory, using the existing built Remora frontend:

```sh
REMORA=/absolute/path/to/remora \
REMORA_TORCH_TEST_INDUCTOR=1 \
python -m unittest discover -s tests -v
```

The `REMORA_TORCH_TEST_INDUCTOR` flag enables actual CPU Inductor compilation
for the static matrix example's forward/backward and replacement-input checks.
Other static full-graph tests use `aot_eager`. The general runtime is tested in
eager mode: graph capture of arbitrary dynamic/effectful programs is not promised.

## Corpus coverage

General lowering compiles **85 of the repository's 91 source test files**.
These are all the type-correct files, including `dotlib.remora`, a library with
no entry. All 73 programs with a no-argument `main` execute in the inventory
check. The remaining 12 files require arguments or do not expose that entry;
argument fixtures are exercised separately.

The six rejected files are intentional frontend type errors:

- `issue34-flatten0.remora`, `issue34-flatten1.remora`
- `issue34-reverse0.remora`
- `issue35-emptyarray0.remora`, `issue35-emptyarray1.remora`
- `typecheck0.remora`

The inventory is now an automated test, not only a one-off survey. Numerical
fixtures are checked against both their expected values and the actual Remora
interpreter under **auto and forced-general lowering**. Nested box output is
checked structurally. No valid program is excluded for the earlier backend's
existential, dynamic-shape or function-array restrictions.

The initial static backend compiled 74 files. The new path additionally covers
boxes/unboxing, dynamic iota, reify-shape, heterogeneous function arrays and
`mono8`, `mono9`, `mono10`, which the existing defunctionalizer rejects. It
reuses the successful type-checked AST rather than modifying that frontend.

## Targeted checks

- Runtime shapes varying across calls, including zero-sized arrays.
- Nested boxes, boxed entry arguments, witness/shape validation and box gradients.
- Box values crossing generated module boundaries and boxed file expressions.
- Record arguments, field validation and gradients.
- Closures with captures, function arrays, array intrinsics over functions,
  returned functions, curried primitives and polymorphic functions.
- Function entry parameters accepting generated functions or typed Python callbacks.
- Negative integer division, ordered/non-associative folds and reductions,
  prefix broadcasting, indexing bounds and empty frames.
- Global effect ordering, unused global values with effects, and unused type/index aliases.
- `trace`, `trace-file`, binary float32 reads and compilation of expressions read
  by `read-file`; `undefined` raises only when evaluated.
- An intrinsic catalog check against the actual Haskell frontend declarations.
- Standalone generated execution, output preservation on compiler errors,
  gradients and full-graph static compilation.

The dynamic and closure-array README commands were run successfully:

```
dynamic.py 6
15

higher_order.py 10 '[[1,2,3],[4,5,6]]'
[[11.0, 12.0, 13.0], [40.0, 50.0, 60.0]]
```

## Convolution, YOLO and GPU limits

Existing `conv.remora` and `conv_remainder.remora` examples agree with independent
`torch.nn.functional.conv2d` evaluations through the static path, including
full-graph AOT execution. No convolution replacement was inserted into the compiler.

The earlier static implementation compiled the full modified YOLOv4 source to
roughly 585 KiB of Python. That was **compilation only**, using source SHA-256:

```
334346d83578f49a52507fab11d0cfed0da269b10f20ab012ade4f6ca98a6e89
```

It does not establish full-network numerical correctness or performance for
this release. Full YOLO inference, trained-weight agreement, GPU compilation
cost, latency and peak memory remain unmeasured for this general compiler.
The separate earlier YOLO-specific prototype's measurements do not apply.

Both CUDA tests are skipped on this host. General lowering intentionally uses
host control flow for dynamic shapes, function selection and effects; runtime
shape tensors may synchronize to the CPU. Coverage of all current checked AST
variants and the corpus is not a formal proof for every well-typed program.
An empty existential array cannot supply a missing witness for an output shape;
underdetermined results report an error rather than inventing a shape.

`read-file` accepts arbitrary Remora expressions and therefore requires the
compiler package and frontend on the execution host. Ordinary numerical,
higher-order, binary-read and trace programs embed their runtime and require
only PyTorch plus Python's standard library.

All maintained changes are within `remora-copy/pytorch/`. Generated modules and
logs are ignored under `.build/`. The `remora` checkout remains clean.
