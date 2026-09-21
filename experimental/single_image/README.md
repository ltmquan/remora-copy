# One image across multiple GPUs

Question: can multiple GPUs reduce the time to process **one** YOLOv4 image? Previous whole-image replica throughput measurements do not answer this. A GPU is a device containing many cores; this experiment partitions work across devices.

The prototype partitions each selected convolution by **output rows**. Each device receives the input rows needed for its output rows, including overlap at cuts. Global padding is applied once; there is no independent tile detection or merging of separately detected objects. Weights are cached per device. Results gather onto GPU0 after each split convolution; residuals, concatenation, pooling and decoding keep their existing semantics on GPU0. Input and final output reside on GPU0; every latency includes scatter, overlap, gather and synchronization. No multiple-image batching.

This initial scatter/gather strategy may be communication-bound. It is a concrete first measurement, not the best possible spatial implementation. If useful work is too small relative to communication, the next variant should keep stripes distributed across a block and exchange only boundary rows. Channel splitting would be another subsequent experiment. Neither is implemented here yet.

## Small first rental run

Use one machine with two matching GPUs; record peer access and `nvidia-smi topo -m`. The harness accepts more devices but the initial sweep only uses two. It never starts rentals or shuts down the machine. A 30-minute experiment budget limits trials; it does not stop rental billing.

1. Run a two-GPU eager correctness smoke check at 608×608.
2. Test 608×608 and 1216×1216, one image per invocation.
3. For each size, compare a compiled single-GPU whole model, a single-GPU version of the per-convolution harness, and two-GPU 50/50 and 1/3–2/3 output-row splits. Split only convolutions whose input side is at least half the image side. Smaller layers stay on GPU0.
4. Three warmups, ten synchronized latency samples per trial. Record all samples, median latency, validation, setup duration and logical transfer bytes. These are pilot measurements, not the report's three-round timing study.
5. Stop on numerical mismatch, setup failure, per-trial timeout, or the overall budget. Don't broaden the sweep automatically.

Direct PyTorch comes first. `--implementation remora-torch` runs the generated Remora (OG) graph through the same operations, for a follow-up if the partition strategy is promising. tinygrad and Futhark multi-device implementations are not part of this first pilot. cuDNN is off by default, consistent with the existing report; `--cudnn on` is available for a separately labeled follow-up. FP32, TF32 off throughout.

Validation compares full dense outputs for the existing ramp and zero images against direct single-GPU eager PyTorch using atol=1e-5, rtol=1e-4. This isolates partitioning errors from the previously observed Darknet/backend differences. A failure stops the sweep and requires investigation; the tolerance is not relaxed automatically. More realistic images and NMS checks should follow before claiming general correctness.

## Run on the rental

Package locally with `python3 package.py`; transfer `single-image-experiments.tar.gz` and the official `yolov4.weights` checkpoint. The package contains supporting modules and prepared size-specific generated graphs, but no weights or old results. Extract it into a fresh directory. Use Python with PyTorch 2.8.0 + CUDA 12.8 and NumPy (the previous rental environment), and verify CUDA can see both devices.

```sh
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4
python experimental/single_image/sweep.py --weights /path/to/yolov4.weights --budget-minutes 30
```

Individual follow-up example:

```sh
python experimental/single_image/run.py --weights /path/to/yolov4.weights --size 1216 --implementation remora-torch --devices 0,1 --split 1,1 --min-side 608 --output results/remora-equal.json
```

Report speedup as single-GPU latency / two-GPU latency, separately against the whole-model compiled baseline and the same per-convolution implementation. A win only against the latter is not a practical win over the existing PyTorch pipeline. Record topology alongside results. GPU execution and peer-copy behavior remain to be validated on the rental; locally, 60 NumPy tests check the partition boundaries and convolution halos exactly, including stride two and unequal splits.
