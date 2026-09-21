# Why the two full YOLOv4 variants tie

The extra `reduce/zero` changes improve the pooling block, but its absolute cost is too small to move the roughly 482 ms full-network result substantially. The teammate version also gives up cascading-window reuse, so the previous full-network comparison was not a reduction-only ablation.

## Controlled isolated pooling experiment

Actual Remora SPP code, 512×19×19 input, one H200, FP32. Four variants hold either the window strategy or reduction choice constant. Three rounds of ten warmups and 100 synchronized GPU-resident calls, median of round medians. All four outputs are bitwise equal to an independent NumPy max-pool reference. These are isolated block timings, not full-network timings.

| Pooling windows | Reduction | SPP latency ms |
|---|---|---:|
| Separate 5×5, 9×9, 13×13 | `reduce` | 1.421960 |
| Separate 5×5, 9×9, 13×13 | `reduce/zero` | 1.126694 |
| Cascading 5×5 (our MD) | `reduce` | 0.450310 |
| Cascading 5×5 | `reduce/zero` | 0.356384 |

- Holding separate windows fixed, `reduce/zero` reduces latency by 20.8%, saving 0.295 ms.
- Holding cascading windows fixed, it reduces latency by 20.9%, saving 0.094 ms.
- Cascading alone reduces block latency by about 68.3% with either reduction choice.
- Both optimizations together give the best isolated result. The added improvement over our current MD is only about 0.02% of its full 481.906 ms latency. This is a scale comparison, not a measured full-network speedup for the combined variant.

## What the compiler actually does

The source calls max-pooling only three times, inside one SPP block on a 512×19×19 feature map. YOLOv4's 110 convolutions execute elsewhere in the graph. The code is not applying these max-pools after every convolution.

Remora lowers `reduce` to a fold and `reduce/zero` to a SOAC reduction. The isolated optimized GPU IR is different: `separate-fold/model.fut_gpu_mem` retains loops of 4, 8 and 12 iterations containing GPU maps; `separate-zero/model.fut_gpu_mem` includes segmented-reduction alternatives. It is therefore incorrect to conclude that both expressions generate identical code or that Futhark cannot optimize this case.

The executed full-network profile adds a useful distinction: the teammate's actual maximum kernels use a small sequential reduction **inside each GPU thread**, parallelizing across output elements. They do not execute the alternative segmented-reduction path inspected in the isolated IR. For these short reductions, fewer launches can matter more than parallelizing each individual maximum. The profile shows 24 maximum-reduction kernel launches in our cascading/fold version and 6 in the teammate's separate/zero version, per inference.

## Full-network profiling cross-check

The profiles run two instrumented inferences. The figures below use the second invocation; they are diagnostic GPU-event totals, not the warmed latency benchmark. They include allocation/page/cache effects and should not be substituted into the report's timing table.

| Profiled work | Our MD | Teammate |
|---|---:|---:|
| Maximum-reduction kernels | 0.196 ms | 0.092 ms |
| Max-reduction launches | 24 | 6 |
| Pool window indexing, extraction and maxima | 3.761 ms | 5.321 ms |
| All `main` kernels | 487.368 ms | 489.567 ms |

Even in this instrumented sample, the attributed pooling kernels are about 0.77% and 1.09% of all main-kernel time. Most pooling time here is window preparation, not the maximum operation. This explains why a real improvement in the reduction itself is hard to see in full YOLO latency. The uninstrumented microbenchmark demonstrates the local benefit more cleanly.

The profile ranges exclude preceding convolution/activation and subsequent concatenation consumers. Generated CUDA was inspected to identify the window-indexing, extraction and maximum kernels. The profile's microsecond event unit is confirmed by generated `cuEventElapsedTime` reporting `ms*1000`. Initial and second invocation effects differ markedly, so these fractions are descriptive, not precise steady-state attribution.

## Full-network results remain unchanged

Current MD: 481.906 ms resident, 482.714 ms transfer-inclusive, 30,048 MiB. Teammate: 481.929 ms, 482.852 ms, 30,050 MiB. Those differences do not establish a performance ranking. Outputs were bitwise identical on all seven tested inputs. The combined optimization was tested only in the isolated block here; the fork source, report table and presentation were not replaced.

Evidence: [isolated timings](results/summary.json), [profile attribution and executed events](results/profile-attribution.json), [teammate kernel example](results/teammate-pool-example.cu), [our kernel example](results/previous-pool-example.cu). Reproduce with `prepare.py`, `run.sbatch`, `analyze.py`, `inspect_kernels.py`, and `summarize_profile.py`. Explorer job 10363070 completed in 5m18s. No rental GPU was used.
