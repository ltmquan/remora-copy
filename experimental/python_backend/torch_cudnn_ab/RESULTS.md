# PyTorch cuDNN on/off check

One H200 allocation, 608×608 YOLOv4, trained weights, identical FP32/TF32-off policy. Both paths use compiled execution with batch one. The explicit cuDNN flag is recorded and checked after execution; on/off compile caches are separate. Three rounds alternate execution order, with five warmups and thirty timed calls per mode in each process. Compilation and warmup are excluded. Aggregates are medians of the three process medians.

| Model | cuDNN | Resident ms | Transfer-inclusive ms | Resident round medians | Transfer round medians |
|---|---|---:|---:|---|---|
| remora | on | 9.236 | 10.164 | 9.224, 9.236, 9.246 | 10.150, 10.166, 10.164 |
| remora | off | 7.973 | 8.862 | 7.978, 7.973, 7.965 | 8.952, 8.852, 8.862 |
| direct | on | 9.314 | 10.261 | 9.333, 9.314, 9.289 | 10.261, 10.358, 10.176 |
| direct | off | 8.049 | 8.927 | 8.050, 8.049, 8.032 | 8.927, 9.051, 8.917 |

All twelve outputs passed comparison over 1,933,155 values against cuDNN-on Remora/round1 at atol=1e-5, rtol=1e-4. Each process also checked a changed input against eager execution. This is fixture validation, not a general equivalence proof.

The separate Darknet on/off study was a different allocation; comparisons between the studies are contextual, not a matched four-way benchmark.
