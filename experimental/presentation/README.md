# Remora YOLOv4 presentation

Eight Beamer slides for a ten-minute talk. Each experiment has a pipeline/explanation slide followed by a results/commentary slide. The report remains unchanged.

Build from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error remora_yolov4.tex
pdflatex -interaction=nonstopmode -halt-on-error remora_yolov4.tex
```

The self-contained TeX includes speaker notes. Suggested times for slides 1–8: 45, 65, 55, 80, 55, 95, 95, 110 seconds (10 minutes).

## Structure

1. Executive summary.
2. Original Remora → Futhark pipeline.
3. Original pipeline results and source-optimization question.
4. Our modified Remora source and reduce/zero.
5. Modified-source results and alternative-backend question.
6. Rapid compiler prototype targeting PyTorch/tinygrad.
7. Alternative-backend results and the end-to-end baseline.
8. Next semester: AI-enabled compilation and a general framework.

## Evidence and scope

All numbers are the transfer-inclusive timings from the final supplied report, in milliseconds per image. They include the full network and dense box decoding, image upload and output download, with weights retained. Setup, compilation, warmup and NMS are excluded. Measurement details are in the speaker notes. No resident-mode, memory, optimized tinygrad or multi-GPU columns are shown.

- The modified Remora measurements are our existing remora-copy version, not the teammate variant. The measured source also includes the pooling rewrite, which is omitted from the slide explanation at the user's request. This table does not isolate the reduction change alone.
- The Python compiler maps known named definitions checked against AST fingerprints. It does not discover arbitrary equivalent computations. Python remains the generated intermediate code even though it is omitted from the diagrams.
- Correctness notes preserve the distinction between numerical comparisons and post-NMS agreement. All paths passed the original input's dense check. On six added inputs they kept the same candidate IDs and classes after NMS, with numerical differences in coordinates and scores.
- Timing source: final supplied report, `remora_yolov4_report (1).md`. Underlying results are in `../python_backend/consistent_comparison/results/summary.json`, `../python_backend/torch_cudnn_ab/results/summary.json`, and `../python_backend/cudnn_ab/results/summary.json`.
- Correctness: `../python_backend/multi_input_validation/results/{summary,nms-summary}.json`.
- Compiler: `../python_backend/{compiler.py,contracts.json,runtime.py}`.
- The separate Draper reproduction has different timing boundaries and is not substituted into these results.

The previous `measurements.tex` remains unused. The ZIP contains the current TeX, PDF and this README.
