# Additional YOLOv4 correctness checks

Six additional 608×608 inputs, trained weights, all nine pipelines. Darknet C++ is the reference. The photo is DARPA’s `evaluation/evalbox.jpg`, oriented using EXIF, converted to RGB, and resized without letterboxing. Other inputs are synthetic. No new performance measurements.

Each cell counts output values exceeding `abs(error) ≤ 1e-5 + 1e-4 × abs(reference)`, out of 1,933,155 values. Zero means the full output passed.

| Pipeline | Black | White | Noise | Gradient | Checkerboard | Photo |
|---|---:|---:|---:|---:|---:|---:|
| Remora (OG) → Futhark | 206,851 | 670,037 | 294,948 | 178,424 | 41,260 | 239,548 |
| Remora (MD) → Futhark | 18 | 17 | 0 | 0 | 0 | 0 |
| Remora (OG) → Python → PyTorch | 4 | 0 | 0 | 0 | 0 | 0 |
| Direct PyTorch | 4 | 0 | 0 | 0 | 0 | 0 |
| Remora (OG) → Python → tinygrad | 896 | 13 | 0 | 0 | 0 | 3 |
| Direct tinygrad | 896 | 13 | 0 | 0 | 0 | 3 |
| Remora (OG) → Python → tinygrad (OP) | 13 | 29 | 0 | 0 | 0 | 0 |
| Direct tinygrad (OP) | 13 | 29 | 0 | 0 | 0 | 0 |

28 of 48 comparisons passed. Original Remora → Futhark failed all six, with maximum absolute errors ranging from about 0.00177 to 1.67. These discrepancies need investigation; the timing table alone does not establish equivalent results on other inputs. Other failures involve fewer values, but have not been established to be harmless rounding.

All 18 generated/direct Python pairs (three backend variants × six inputs) were bitwise identical. Both PyTorch metadata files confirm cuDNN disabled. These checks do not measure detection accuracy against ground-truth labels.

[Full error counts and maxima](results/summary.json) · [Input definitions and hashes](inputs.json)


## Final detections after NMS

We applied the DARPA harness’s postprocessing rules to every saved output: confidence = objectness × class probability, confidence > 0.25, then class-wise greedy IoU NMS with threshold 0.45. This is a CPU implementation of those rules; boxes remain in the resized 608×608 image coordinate system. No additional GPU execution or timing was needed.

| Input | Darknet detections | All other pipelines keep the same candidates and classes? |
|---|---:|---|
| Black | 0 | Yes |
| White | 0 | Yes |
| Noise | 1 | Yes |
| Gradient | 1 | Yes |
| Checkerboard | 0 | Yes |
| Real photo | 1 | Yes |

On the photo, original Remora → Futhark differs from Darknet by 0.0001304 in confidence and at most 0.0397 pixels in a box center/width/height component. The boxes have IoU 0.9999003. Thus the dense-output discrepancies did not change which detections survived here, although numerical outputs still differ. This does not establish equivalence on other images or at other thresholds, or detection accuracy against ground truth.

[Post-NMS outputs and comparisons](results/nms-summary.json) · [Reproducible CPU comparison](compare_nms.py)
