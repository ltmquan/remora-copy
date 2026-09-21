# Teammate YOLOv4: Futhark version comparison

Requested September 16, 2026. Job **10382237**, queued after exact-input Draper job10382183 to use one GPU at a time. Remote: `/scratch/luu.qu/remora-copy/experimental/teammate-futhark-0271-20260916`.

Compare the prior Futhark0.27.0 prerelease CUDA library with a new Futhark0.27.1 CUDA library. Both use the teammate Remora source (explicit-zero dot product and both nested pooling reductions, separate5/9/13windows). Reuse the identical Remora-generated SOACS to isolate the Futhark compiler; the Remora frontend is not changed. The prior executable is retained from the completed teammate experiment; the new executable uses the same harness and C compiler flags.

One H200 allocation, alternating old/new builds over three rounds. Each process uses five warmups and30timed calls per mode (resident and transfer-inclusive), matching the report protocol. Report median of round medians. Same608x608ramp input and officialweights. Startup excluded. Compare saved dense outputs old versus new for every round; this is not a repeat of the seven-image suite. No new memory measurement requested.

Scripts: `run.sbatch`, `analyze.py`.22minutejobcap. Results pending. Earlier teammate reference:481.929msresident,482.852mstransfer-inclusive; freshly rerun old results will be the primary comparison. Do not replace report/slides until results are reviewed.

## User-requested shorter protocol (supersedes above)

Job10382237 was cancelled while still pending, with no GPU time used. Replacement **10382330**, script `quick.sbatch`, depends on10382183. Only the new compiler is run, in one process: five warmups and three timed calls per resident/transfer mode. Report arithmetic mean of three calls per mode (`analyze-quick.py`). No old-version rerun or full correctness suite. Harness retains its finite-output check. Eight-minute cap including compilation/startup. Compare cautiously with historical three-round median values; protocols differ. This replaces the earlier22minute/sixprocessplan.

Completed: job10382330 finished in3m45s. New mean482.384msresident/483.234mstransfer-inclusive, effectively unchanged. See `RESULTS.md` and `results/summary.json`. No further jobs.
