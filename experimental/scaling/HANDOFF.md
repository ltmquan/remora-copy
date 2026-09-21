# Eight-GPU handoff

Connected rental: one host with 8× NVIDIA H100 80 GB HBM3, NV18 links between every GPU pair, 208 logical CPUs and about 1.7 TiB host RAM. Driver 570.148.08 and CUDA toolkit 12.8. Everything is in this fork.

| Dimension | Prepared values |
|---|---|
| Image side | 608, 960, 1216, 1536 |
| GPU count | 1, 2, 4, 8 |
| Implementation | Remora/direct PyTorch; Remora/direct tinygrad baseline and im2col; original/modified Remora→Futhark; native Darknet+dense decode |
| Assignment | Uniform contiguous, round-robin, calibrated weighted, dynamic whole-image chunks |
| Timing | GPU resident; transfer inclusive (Darknet only supports the latter) |
| Memory | Separate sampled process-footprint pass; available framework live/reserved counters with their measurement scopes |
| Work | 128 requests per cell by default; strong scaling. Weak scaling is an explicit option. |
| Target | YOLOv4 executable; vector search, language modeling and graph targets reviewed in DARPA_TARGETS.md, adapters deferred |

Local checks verify typechecking for both Remora source variants at every size, generated topology against independent cfg topology, unchanged weights/anchors, partition coverage, matrix dimensions, and size/GPU ratio reporting. CUDA C generation was performed locally; the Linux libraries and pinned Darknet fork have been built on the rental. All nine variants have passed 608² one-/eight-GPU smoke checks under their recorded reference/tolerance settings.

Once SSH is available:

1. Upload the archive and capture hardware/topology/driver details.
2. Configure the Python environment and CUDA builds; fetch the SHA256-checked checkpoint.
3. Produce references and run a small 608² smoke test, progressing through 1, 2, 4, and 8 GPUs, including native Darknet output comparison.
4. Sweep image sizes with uniform assignment. Record out-of-memory/timeouts without treating them as timing values.
5. Test other assignment policies, then collect memory separately and export the CSV/table.

The first sweep measures independent-image throughput. Single-image model/tensor partitioning needs a later executor. GPU memory is not automatically pooled, and we do not tile images in a way that changes the network's boundary behavior.

For the full size sweep, uniform assignment is sufficient initially (272 timing cells). The full policy matrix has 884 timing cells and is planned separately. Use 16 requests for the initial correctness smoke, then 128 fixed requests for strong scaling so the 8-GPU case still has 16 requests per GPU. The report includes GPU scaling efficiency (speedup divided by GPU count).

Select nested GPU subsets after inspecting `nvidia-smi topo -m`; `matrix.py --gpu-order ...` controls the device order and takes prefixes for 1/2/4/8 runs. This matters if the machine has separate NVLink or NUMA groups. CUDA UUID checks still prevent two workers from accidentally using one GPU. CPU thread defaults scale down with available CPUs per worker.

The Darknet build now detects the installed GPUs' CUDA architectures. Python/CUDA version compatibility must still be checked on the host; an eight-GPU allocation does not imply H100 hardware. No work is dispatched to the terminated instance.
