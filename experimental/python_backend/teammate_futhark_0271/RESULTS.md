# Teammate YOLOv4 with Futhark 0.27.1

Job10382330 completed successfully in3m45s on oneH200. Same teammate source and Remora-generated SOACS as the earlier run; new Futhark0.27.1 CUDA library. One process, five warmups followed by three timed calls per mode. Arithmetic means requested by the user; startup and build excluded.

| Mode | Three calls, ms | Mean, ms | Earlier0.27.0 prerelease result, ms |
|---|---|---:|---:|
| GPU resident | 482.388499,482.376720,482.386078 |482.383766|481.928952|
| Including transfers |483.188292,483.275685,483.236768|483.233582|482.851956|

No meaningful performance improvement is evident: new means are approximately0.094% and0.079% higher, respectively. Earlier values are medians of three round medians with30calls each; the new values are means of only three calls. Different runs/protocols prevent treating these small differences as a compiler regression.

The timing harness completed its finite-output check. This short test did not repeat the seven-image correctness suite or measure memory. No source, report, or slide numbers were replaced. Raw results and compiler/GPU/hash records are in `results/`; archived in `quick-evidence.tar.gz`.

The originally planned six-process old/new comparison10382237 was cancelled before starting. No further benchmark jobs are needed.
