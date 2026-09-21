# Teammate YOLOv4 comparison

The teammate source changes all three reductions to `reduce/zero` and keeps separate 5×5, 9×9 and 13×13 pooling windows. The current modified source changes the dot product and reuses three successive 5×5 pools. All source files and prior results are retained.

| Version | Resident ms | Transfer-inclusive ms | GPU footprint MiB |
|---|---:|---:|---:|
| Current Remora (MD) | 481.906 | 482.714 | 30,048 |
| Teammate candidate | 481.929 | 482.852 | 30,050 |

No meaningful performance or memory improvement. Retain the current fork source and the report's MD row. Both paths use the same 608×608 trained fixture and three rounds of five warmups plus thirty timed calls per mode. Memory is measured separately. Startup is excluded from timing; candidate startup took about 99 seconds per process.

The candidate output is bitwise identical to the current modified implementation on all seven distinct inputs. Against Darknet, it passes the original input, noise, gradient, checkerboard and real photo. Black has 18 values outside tolerance, and white has 17, out of 1,933,155 values per image. The tolerance remains `abs(error) ≤ 1e-5 + 1e-4*abs(reference)`.

Confidence filtering at 0.25 and class-wise greedy NMS at IoU 0.45 retain the same candidates and classes as Darknet on all seven inputs. Scores and box coordinates may still differ. This is not labeled-dataset detection accuracy or a general equivalence proof.

Runtime job 10362144 completed three timing rounds, then hit an overly short 90-second correctness-process timeout during startup. Continuation job 10362305 completed only the remaining correctness and memory work. No timing round was repeated or taken from memory-instrumented execution.

[Measurements and checks](results/summary.json) · [Bitwise comparison with previous MD](results/vs-previous-modified.json) · [Source](model.remora) · [Build manifest](build.json)
