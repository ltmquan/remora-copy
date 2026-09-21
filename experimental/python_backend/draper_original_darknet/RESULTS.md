# Original Draper Darknet reproduction — September 16, 2026

Job 10383155 completed successfully on an H200. Ran the original benchmark with `./yolov4 -i 10`, the default dog.jpg image, official weights, and cuDNN disabled. See README.md for source commits and necessary build/environment differences.

| Measurement | Result |
|---|---:|
| Mean per full forward pass | 7.97 ms |
| Minimum / maximum | 7.82 / 8.62 ms |
| Total timed loop, 10 forward passes | 79.83 ms |
| Printed inference wall time (rounded) | 0.08 s |
| Full process wall time, including setup and warmup | 5.61 s |
| Network loading | 2545.80 ms |
| Warmup | 2004.07 ms |

Final detections: dog 97.89%, truck 94.19%, bicycle 94.45%. NMS and detection extraction follow the timed loop.

The reported 2.22 s was not reproduced on the H200. If Draper's number is the total ten-inference loop, it corresponds to 222 ms per inference, roughly 27.8 times this run. Their raw timing log is unavailable, so which printed statistic supplied 2.22 s remains unconfirmed. Different hardware and build environments prevent attributing the gap to one cause.

Our earlier 9.364 ms comparison includes CPU dense decoding; this original harness times network_predict alone, so those two measurements have different boundaries.

Raw output: [results/run.log](results/run.log).
