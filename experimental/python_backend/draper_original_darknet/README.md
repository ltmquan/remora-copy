# Original Draper Darknet benchmark reproduction

User requested running their original experiment to investigate2.22s. Source: remora-lang/darpa-mocha draper_changes commit3827a414a68c755a22ca4c7fa4d02630095f4634 (timing code also confirmed at reportsubmissioncommitf5d03142cb310d16858202536c75cee485e13761). Darknet: remora-lang/sei-wohlbier-darknet feature/draper commit58bb1326f8aefe19805b33901101697de5f09336.

Build original Darknet with committed GPU=1,CUDNN=0,CUDNN_HALF=0,OPENCV=1,AVX=1,OPENMP=1 and effectiveARCHcompute70+PTX (laterMakefileassignment overrides precedingcompute86line). Run original `./yolov4 -i 10`, defaultdata/dog.jpg,608cfg,officialweights,letterboxing,onewarmup,NMSaftertiming. Originalbenchmarkdoesnotcallfuse_conv_batchnorm. Unlike our previouscomparisonharness, no customCPUdense decode insideinference timer. Capture completeoutput,totalinferencewalltime,averageperiteration,andprocesswalltime.

Necessary differences: H200 instead of RTX A2000, cluster CUDA12.8/OpenCV4.10, GCC instead of unavailable Clang. Benchmarksourcechangesonlyhardcodedworkingdirectory. OpenCVpkg-configpathfile provided because cluster module omitsone. No application-level tuning or precision overrides. FourallocatedCPUcores (OpenMP observes scheduler affinity); original laptop14cores.12minjobcap mostlybuild;10inferences only. Not an exacthardware/environmentreproduction; cannotpromise2.22s.

All writes here in remora-copy; source archive obtained read-only from existing auditcheckout. Mainreport/slides unchanged.

Submitted job **10383115**. Remote `/scratch/luu.qu/remora-copy/experimental/draper-original-darknet-20260916`. Results pending; capture `results/run.log` and `run-10383115.log`.

Initialjob10383115 failed after24s:Darknetlibrarybuilt; harnesslinkfailed `cannot find -lnvidia-ml`. Retry **10383155** adds CUDA's `lib64/stubs` linkerpath for NVML (runtime still uses installeddriverlibrary). No model/timerchanges.4mincap,reusescompletedbuild. Originallinklogretained asresults/harness-build-10383115.log.
