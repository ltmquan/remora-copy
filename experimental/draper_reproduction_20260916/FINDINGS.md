# Draper YOLOv4 reproduction — 16 September 2026

The September 14 report does not identify a newly faster Remora implementation than the teammate version we already tested. Its demo source is identical to that version except for a final newline. The reported 0.48 s is also numerically consistent with our 481.929 ms H200 measurement. This is a source comparison, not evidence that the two measurements used the same hardware or protocol.

## What the new report says

- Darknet: 2.22 s; Remora: 0.48 s; ratio 4.6×.
- Listed machine: Dell Precision 5570, RTX A2000 8GB, i9-12900H, 64 GB RAM.
- Remora now compiles the whole forward pass through Futhark, replacing the earlier C/Remora layer substitutions and MLIR backend.
- Reproduction instructions cover **only the Remora/Futhark measurement**. No current Darknet command, compiler flags, raw log, iteration count, or cuDNN configuration accompanies the 2.22 s figure.

## Source and benchmark scope

Demo: https://github.com/remora-lang/demo-remora-futhark at commit `5d7cf48e53a6b1576932a68d6afee3472579e780` (2026-09-14).

The supplied source uses explicit-zero reductions for the dot product and both max-pooling reductions, and separate 5/9/13 pooling windows. It is the teammate version, not our cascading-pooling version. The benchmark specification is one 608×608 image and 22743×85 dense outputs, without NMS. The Futhark server benchmark times repeated entry-point calls with inputs already loaded; executable startup, compilation, and file input are outside the reported per-call timing.

Our earlier teammate measurement: 481.929 ms resident, 482.852 ms transfer-inclusive. The current report's optimized Remora row is 481.906 ms / 482.714 ms. These are practically tied.

## Potential baseline mismatch — not established

The earlier Draper report already contains the same 2.22 s baseline and explicitly calls it wall time. Its reproduction instructions use the `draper_changes` branch of`remora-lang/darpa-mocha`. At inspected commit `3827a414a68c755a22ca4c7fa4d02630095f4634`, `3_yolov4/run_tests.sh` runs `./yolov4 -i 10`; `yolov4.cc` reports total wall time for that loop separately from per-iteration timing.

This creates an unresolved possibility that 2.22 s is ten inferences while 0.48 s is one. The old raw timing log is needed to establish that. We should **not** label the 4.6× speedup reproduced, or claim a definite factor-ten error. If 2.22 s were the ten-call total, its per-image value would be 222 ms and the comparison would reverse (480ms/222ms≈2.16× slower).

The A2000 and H200 are also different hardware. No A2000 was available for this reproduction. We cannot reproduce the laptop speedup from an H200 measurement alone.

## Fresh reproduction

One bounded H200 job 10380058 was submitted to Explorer. It uses the published Futhark IR directly and release Futhark 0.27.1 (latest stable listed in the linked release index), CUDA 12.8, and the demo's published expected output. The report does not pin its Futhark version.

The published 250 MiB input download was slow. We reconstructed the binary input using the image from the downloaded prefix and the already cached official YOLOv4 weights with the repository's parameter order and binary headers. After the download completed, the entire published input and reconstructed input were byte-identical (`cmp`) and had matching SHA-256: `9e31918d097c0a36a712a83bee1b21e02cbbd9383b773d2e1143f13ca3a74a74`.

Commands and logs are kept alongside this note. The job has a 20-minute Slurm cap and explicit 600 s compile / 480 s benchmark caps. The final recovery completed; see the measurements below.

### The published command failed before inference

Job 10380058 failed after 2m56s. Compilation succeeded, but `futhark bench` failed while restoring input arguments: `./yolov4: Line too long.` The generated server's `process_line` permits only 1000 tokens; this YOLO entry point has 542 input parameters, and its restore command exceeds that token limit. This is a tooling/protocol failure, not a numerical failure or a slow network inference.

One recovery run (job 10380144, eight-minute limit) changes only `int max_num_tokens = 1000` to 4096 in generated C, leaving the generated GPU computation unchanged. It rebuilds the server and requests 30 benchmarked calls without the optional convergence phase. The exact unmodified-command failure is retained separately from this patched-server result.

The first recovery build failed because our manual link command omitted `-lcudart`. This is a reproduction-script mistake, distinct from the upstream server failure. The script was corrected and the same protocol patch was resubmitted once; no GPU computation or inference data was changed.

Final recovery job: **10380877**. No further automatic retries are planned. Remote directory: `/scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916`.

To inspect and collect after completion:

```sh
ssh luu.qu@login.explorer.northeastern.edu 'sacct -j 10380877 --format=JobID,State,Elapsed; tail -35 /scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916/results/bench.log'
ssh luu.qu@login.explorer.northeastern.edu 'cd /scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916 && tar -czf evidence.tar.gz results run.sbatch retry.sbatch stage_input.py run-10380058.log retry-10380144.log retry-10380877.log'
scp -O luu.qu@login.explorer.northeastern.edu:/scratch/luu.qu/remora-copy/experimental/draper-reproduction-20260916/evidence.tar.gz .
```

A successful `bench.json` should contain runtimes in microseconds and a passing output comparison. This check is the demo's expected-output comparison, not the seven-image correctness experiment used in our slides. Record its actual sample count and statistics; do not substitute it into the existing multi-round report table.

## Completed result

Final job **10380877 completed successfully in 3m14s**. Futhark benchmarked 30 calls and passed its comparison with the published expected output. Arithmetic mean: **482.993 ms/image**; median: **482.902 ms/image**; range: **482.318–486.675 ms**. The benchmark process took 1m50.88s including server startup; that is not the per-image runtime.

This reproduces approximately 0.48 s per call on H200, using the published program and byte-identical input with the documented server-protocol patch. It does not reproduce the report's A2000 hardware or its claimed 4.6× advantage over Darknet. The existing teammate result (481.929 ms, a median under our earlier protocol) and this result (482.993 ms, arithmetic mean) are on the same scale; do not interpret their small difference as a source improvement.

The remaining information needed from Draper is the exact Darknet command/build settings and raw timing log establishing whether 2.22 s covers one image or ten iterations. No further GPU jobs are scheduled. Raw evidence is in `evidence.tar.gz` and `results/`; the existing slides/report numbers were not changed.

## Follow-up: README benchmark defaults, exact downloaded files

At the user's request, job **10382183** reruns the benchmark with the published `yolov4_input.bin` itself, the published expected output, and the original benchmark specification restored from demo commit `5d7cf48e53a6b1576932a68d6afee3472579e780`. The earlier reconstructed input was already byte-identical; this rerun additionally uses the original filename/file and restores the benchmark's default sampling/convergence settings instead of the earlier fixed 30 calls. It reuses the published-IR CUDA server with the documented token-limit patch. `--json` only records results. A 540-second external timeout and ten-minute Slurm limit bound resource use. Script: `readme-defaults.sbatch`. Remote results: `readme-defaults/results/`. No new measurement available at submission.

### README-default run completed

Job10382183 completed in1m49s. The default benchmark selected 10 measured calls and reported arithmetic mean **482.7557 ms/image** (median 482.3680 ms). Its published expected-output check passed. It used the actual downloaded input file and original benchmark specification. Benchmark process wall time was1m43.72s including startup. The result remains approximately0.483s, consistent with the earlier fixed30-call result. Evidence: `readme-defaults/results/` and `readme-defaults-evidence.tar.gz`. This experiment is complete; separate teammate/new-compiler job10382330 remains pending.
