# Darknet on Draper's published image

Job10382681, oneH200,4mincap. Remote `/scratch/luu.qu/remora-copy/experimental/darknet-draper-input-20260916`.

Reuses our measured cuDNN-off Darknet library, officialweights,608x608FP32,TF32off. Harness copied from cudnn_ab, adds binaryimageargument and uses5warmups3timedcalls. Runs originalramp and exactimageextracted from publishedFutharkbinary in sameallocation. Reports arithmeticmean of network+decode+transfers and network_predict timing separately. NMSexcluded. Saves denseoutputs and compares Draperimage output to publishedFutharkexpected usingatol1e-5,rtol1e-4. This is an input-sensitivity test, not a reproduction of undocumented DraperDarknetbuild/A2000hardware.

Scripts: benchmark.cpp,run.sbatch,analyze.py. Results pending. Preserve mainreport/slides.

Completed job10382681 in12s. Draper input9.305ms/image, originalramp9.562ms/image, eachmeanof3calls. Publishedexpectedoutputcomparison passed. SeeRESULTS.md. No more jobs.
