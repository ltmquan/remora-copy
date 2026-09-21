# Darknet on Draper's exact input

Job10382681 completed in12seconds on oneH200. Reused our cuDNN-off Darknet library, officialweights,608x608FP32,TF32off. Each input:5warmups,3timedcalls, arithmeticmean. No NMS. These short-test means do not replace the report's longer-protocol medians.

| Input | Full network + dense decode and transfers, ms/image | network_predict call alone, ms/image |
|---|---:|---:|
| Our synthetic ramp |9.562248|7.787989|
| Draper's published image |9.304754|7.729930|

Our ramp samples were9.410455,9.403988,9.872301ms; the last raises its mean. Draper-image samples were9.322263,9.300916,9.291084ms. Both inputs remain around9ms; this does not establish a meaningful input-dependent speedup and cannot explain the reported2.22seconds (or hypothetical222msperimage).

The Draper image output passed atol1e-5+rtol1e-4 versus the published Futhark demo expected output:0 failing values, maximum absolute error0.00002372264862060547. Relative tolerance permits an absolute difference above1e-5 for some reference values. This is a dense-output test on this one image, not a new NMS or ground-truth accuracy evaluation.

The network_predict timing already includes whatever transfers Darknet performs internally. It is not a GPU-resident measurement. Full totals additionally include our CPU dense decoding, as in the report.

Interpretation: image choice is not the explanation for the discrepancy. Hardware, Darknet build settings, and total-versus-per-inference reporting remain unresolved. The September2report commit's run_tests.sh uses-i10 and its wall-time field measures the total inference loop, but no raw2.22slog was found to prove that field supplied the report number. We have not reproduced the RTX A2000 run.

Evidence: results/summary.json, results/*.log, results/build-sha256.txt, results/image-sha256.txt and evidence.tar.gz. No report/slides changed; no further GPU runs scheduled.
