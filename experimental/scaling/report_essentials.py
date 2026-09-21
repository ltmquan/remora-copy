"""Build the final rental report from preserved JSON, including unmeasured cells."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/full-comparison-20260911'
OUT = ROOT / 'comparison-20260911'
LABELS = {
    'remora-torch': 'Remora → PyTorch', 'direct-torch': 'Direct PyTorch',
    'remora-tinygrad': 'Remora → tinygrad baseline',
    'direct-tinygrad': 'Direct tinygrad baseline',
    'remora-tinygrad-im2col': 'Remora → tinygrad im2col',
    'direct-tinygrad-im2col': 'Direct tinygrad im2col',
    'remora-futhark-original': 'Original Remora → Futhark',
    'remora-futhark-modified': 'Modified remora-copy → Futhark',
    'darknet-dense': 'Darknet + dense decode',
}
DESCRIPTIONS = {'remora-torch': 'Original Remora → Python → PyTorch/CUDA. We wrote a small, specialized compiler prototype for these Remora operations, rather than manually translating the network; it is not yet a general Remora compiler.', 'direct-torch': 'An independently written Python YOLOv4 network using the same PyTorch runtime, weights, and inputs. This is the control for overhead from the Remora frontend.', 'remora-tinygrad': 'Remora-generated Python using baseline tinygrad convolutions and warmed TinyJit execution.', 'direct-tinygrad': 'An independent Python YOLOv4 network using the same baseline tinygrad runtime.', 'remora-tinygrad-im2col': 'Remora-generated Python with optimized tinygrad convolutions: contiguous image patches (im2col) become matrix multiplications, with cached batch-normalization-folded weights. This trades extra storage for speed; timing uses warmed TinyJit without BEAM search.', 'direct-tinygrad-im2col': 'The independent Python network using the same optimized tinygrad runtime.', 'remora-futhark-original': 'Original Remora → Futhark → CUDA, invoked through Python/ctypes. Futhark managed-memory paging can still transfer data in resident mode.', 'remora-futhark-modified': 'The modified YOLOv4 Remora source from remora-copy, compiled through Futhark to CUDA. “Modified” refers to the Remora program, not the Futhark compiler.', 'darknet-dense': 'Native Darknet CUDA, built without cuDNN, invoked through Python/ctypes. Dense decoding matches the other paths’ output before thresholding/NMS; the wrapper includes host copies and exposes only transfer-inclusive timing.'}
NOTES = {'remora-torch': 'Near-linear GPU scaling. The main exception is 1536: transfers reduce single-GPU throughput from 25.66 to 11.89 images/sec; the cause is not yet isolated.', 'direct-torch': 'Closely matches the Remora path at every size; there is no clear frontend performance penalty.', 'remora-tinygrad': 'Good GPU scaling at 608, but slow per-image execution. Larger sizes fail numerical validation; remaining configurations were skipped.', 'direct-tinygrad': 'Matches the Remora baseline in both speed and larger-size validation failures.', 'remora-tinygrad-im2col': 'About 4.1× faster than baseline at 608, and passes all sizes. Memory rises from the baseline’s 918 MiB to 2,810 MiB at 608.', 'direct-tinygrad-im2col': 'Almost identical throughput to the optimized Remora path; also passes all sizes.', 'remora-futhark-original': 'Eight GPUs at 608 are still about 13× slower than one PyTorch GPU. At 960, setup/checks time out near GPU memory capacity; this is not a measured inference duration.', 'remora-futhark-modified': 'Roughly doubles original-source throughput at 608 and successfully runs 960. Still substantially slower than PyTorch; larger tests were cut.', 'darknet-dense': 'Leads the transfer-inclusive comparison at 608, but larger sizes fail the chosen numerical tolerance. This is not a standalone optimized C++/cuDNN baseline.'}

def records(phase):
    return [json.loads(p.read_text()) for p in (DATA / phase).glob('yolov4-*.json')]

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    timing = records('timing-uniform')
    memory = records('memory-essentials') + records('memory-baselines')
    execution = json.loads((DATA / 'timing-uniform/execution.json').read_text())
    plans = json.loads((DATA / 'timing-uniform/plan.json').read_text())['cells']
    status = {x['name']: x for x in execution}
    by_key = {(d['config']['implementation'], d['config']['size'],
               len(d['config']['gpus']), d['config']['mode']): d for d in timing}
    rows = []
    for cell in plans:
        cmd = cell['command']
        impl = cmd[cmd.index('--implementation') + 1]
        size = int(cmd[cmd.index('--size') + 1])
        count = cmd.index('--policy') - cmd.index('--gpus') - 1
        mode = cmd[cmd.index('--mode') + 1]
        d = by_key.get((impl, size, count, mode), {})
        st = status[cell['name']]
        rate = d.get('median_images_per_second')
        one = by_key.get((impl, size, 1, mode), {}).get('median_images_per_second')
        rows.append(dict(implementation=impl, image_side=size, gpus=count, mode=mode,
                         status=d.get('status', st['status']), images_per_second=rate,
                         megapixels_per_second=rate*size*size/1e6 if rate else None,
                         median_seconds_for_128_images=statistics.median(t['seconds'] for t in d['trials']) if rate else None,
                         total_seconds_across_5_trials=sum(t['seconds'] for t in d['trials']) if rate else None,
                         amortized_milliseconds_per_image=1000/rate if rate else None,
                         speedup_vs_1_gpu=rate/one if rate and one else None,
                         requests=128, trials=5, reason=st.get('reason', ''),
                         error=d.get('error', '')))
    write_csv('timing.csv', rows)
    memory_rows = []
    for d in sorted(memory, key=lambda x: (x['config']['implementation'], x['config']['size'], len(x['config']['gpus']))):
        c = d['config']
        footprints = [t['footprint'] for t in d.get('trials', []) if t.get('footprint')]
        per_gpu = [v for f in footprints for v in f['process_peak_bytes_by_gpu'].values()]
        totals = [f['simultaneous_process_peak_bytes'] for f in footprints]
        counters = [w['memory'] for t in d.get('trials', []) for w in t['workers'] if w.get('memory')]
        live = [m['live_peak_bytes'] for m in counters if m.get('live_peak_bytes') is not None]
        reserved = [m['reserved_peak_bytes'] for m in counters if m.get('reserved_peak_bytes') is not None]
        memory_rows.append(dict(implementation=c['implementation'], image_side=c['size'],
            gpus=len(c['gpus']), status=d['status'],
            max_per_gpu_process_MiB=max(per_gpu)/2**20 if per_gpu else None,
            simultaneous_total_process_MiB=max(totals)/2**20 if totals else None,
            max_worker_live_counter_MiB=max(live)/2**20 if live else None,
            max_worker_reserved_counter_MiB=max(reserved)/2**20 if reserved else None,
            samples=sum(f['samples'] for f in footprints)))
    write_csv('memory.csv', memory_rows)
    def rate_text(impl, size, count, mode):
        if impl == 'darknet-dense' and mode == 'resident': return '—'
        d = by_key.get((impl, size, count, mode))
        if d:
            rate = d.get('median_images_per_second')
            return f'{rate:.2f}' if rate else ('Timeout' if d['status'] == 'timeout' else 'Fail')
        name = f'yolov4-{impl}-{size}-{count}gpu-uniform-{mode}'
        reason = status.get(name, {}).get('reason', '')
        return 'Cut' if 'User stopped' in reason else 'Skip'
    counts = Counter(x['status'] for x in execution)
    lines = [
        '# YOLOv4 on the 8× H100 rental', '',
        '## What we ran', '',
        'YOLOv4 with official weights on one **8× H100 80GB** machine. We tested '
        '**608, 960, 1216, and 1536 square images** on **1/2/4/8 GPUs**, distributing '
        'whole images uniformly across batch-one model replicas.', '',
        '**Each timing trial processes 128 images total across all GPUs**, repeated '
        'five times. Tables show median **images/sec**; runtime for 128 images is '
        '**128 ÷ images/sec**. Compilation, setup, and warmup are excluded.', '',
        '**R** = GPU resident; **T** = includes image upload and dense output download. '
        'Inputs alternate between synthetic ramp and zero images. We check numerical '
        'agreement, not detection accuracy. Memory is measured separately after warmup.', '',
        f'**Results:** {counts["completed"]} timing cases passed, {counts["failed"]} failed, '
        f'{counts["skipped"]} skipped; all {len(memory_rows)} memory cases passed. '
        'Further Futhark and partition-strategy runs were cut for cost.', '',
        '**Fail** = numerical mismatch; **Timeout** = 600-second setup/check limit; '
        '**Skip** = unmeasured after a related failure; **Cut** = canceled for cost; '
        '**—** = unavailable.', '',
    ]
    for impl, label in LABELS.items():
        lines += [f'## {label}', '', DESCRIPTIONS[impl], '',
                  '| Image size | 1 GPU R | 1 GPU T | 2 GPUs R | 2 GPUs T | 4 GPUs R | 4 GPUs T | 8 GPUs R | 8 GPUs T |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for size in (608, 960, 1216, 1536):
            values = [rate_text(impl, size, n, mode) for n in (1,2,4,8) for mode in ('resident','transfer')]
            lines.append('| '+ ' | '.join([f'{size}×{size}', *values]) + ' |')
        lines += ['', '**Notes.** ' + NOTES[impl], '']
    implementation_lines = lines
    lines = [
        '## Memory', '',
        'Transfer-inclusive, separately sampled after warmup. MiB below is the **maximum '
        'per-GPU process footprint**; the 8-GPU column is not the total across eight GPUs. '
        'The CSV also contains simultaneous totals and allocator counters. Each worker '
        'replays for at least three seconds. Sampling can miss short peaks and excludes setup.', '',
        '| Implementation | Side | 1 GPU peak MiB | 8 GPUs: max per-GPU MiB |',
        '|---|---:|---:|---:|']
    mem = {(r['implementation'],r['image_side'],r['gpus']):r for r in memory_rows}
    for impl,label in LABELS.items():
        for size in (608,960,1216,1536):
            entries = [mem.get((impl,size,n)) for n in (1,8)]
            if not any(entries): continue
            values = [f'{e["max_per_gpu_process_MiB"]:,.0f}' if e else '—' for e in entries]
            lines.append('| '+' | '.join([label,str(size),*values])+' |')
    lines += ['', '**Notes.** PyTorch’s 1216 memory spike is not a smooth pixel-count '
        'trend; its reserved allocator counters show the same pattern. The cause remains '
        'unresolved. Tinygrad im2col memory grows smoothly with size.', '',
        '**Live allocation** counts buffers the backend considers active; process footprint '
        'also includes reserved pools and other CUDA memory. Lower live allocation can '
        'therefore leave footprint unchanged. It helps compare variants within one backend, '
        'but our counters differ: Futhark includes setup, PyTorch trial counters miss some '
        'CUDA-graph temporary liveness, and tinygrad/Darknet have no equivalent here.', '',
        'No new Futhark memory tests ran after the cutoff. Earlier 608 measurements are '
        'in [RENTAL_CHECKS.md](../RENTAL_CHECKS.md).', '',
        '## Measurement details and files', '',
        'PyTorch 2.8.0+cu128; tinygrad 0.14.0; CUDA 12.8. Darknet fork: '
        '58bb1326f8aefe19805b33901101697de5f09336, without cuDNN. Timed Python uses FP32; '
        'validation uses an independent FP64 reference and the fixed bound '
        'abs(error) ≤ 2e-5 + 2e-4 × abs(reference). Memory trials start with eight total '
        'requests, adding replays to reach three seconds per worker. Sampling can miss '
        'short peaks; no memory-run throughput is reported.', '',
        'The broad partition sweep and additional DARPA adapters were not completed. '
        'Small timing differences between generated and direct Python are not established '
        'speed advantages.', '',
        '[Timing CSV](timing.csv) · [Memory CSV](memory.csv) · '
        '[Cross-implementation CSV](cross-implementation.csv) · '
        '[Raw results](../results/full-comparison-20260911/)', '']
    memory_and_scope_lines = lines
    lines = implementation_lines + [
        '## Across implementations: throughput and image size', '',
        '**One GPU, transfer-inclusive.** Cells show **images/sec (megapixels/sec)**. '
        'Megapixels/sec = images/sec × width × height ÷ 1,000,000. If runtime were '
        'proportional to image area, this value would stay constant across sizes.', '',
        '| Implementation | 608×608 | 960×960 | 1216×1216 | 1536×1536 | 608: seconds / 128 images |',
        '|---|---:|---:|---:|---:|---:|']
    summary_rows = []
    for impl, label in LABELS.items():
        values = []
        for size in (608,960,1216,1536):
            d = by_key.get((impl,size,1,'transfer'), {})
            rate = d.get('median_images_per_second')
            mpixels = rate*size*size/1e6 if rate else None
            values.append(f'{rate:.2f} ({mpixels:.2f})' if rate else rate_text(impl,size,1,'transfer'))
            summary_rows.append(dict(implementation=impl,image_side=size,gpus=1,mode='transfer',
                                    status=d.get('status',rate_text(impl,size,1,'transfer')),
                                    images_per_second=rate,megapixels_per_second=mpixels))
        first = by_key.get((impl,608,1,'transfer'), {})
        duration = statistics.median(t['seconds'] for t in first['trials']) if first.get('median_images_per_second') else None
        lines.append('| '+' | '.join([label,*values,f'{duration:.2f}' if duration else '—'])+' |')
    lines += ['', '**Notes.** Generated and direct Python paths closely match. '
              'PyTorch beats optimized tinygrad; Darknet leads at 608 only. '
              'Megapixels/sec varies with size, so runtime is not simply proportional '
              'to pixel count. Failed and skipped cells are not included in a ranking.', '']
    lines += memory_and_scope_lines
    lines += [
        '## Conclusions: patterns and exceptions', '',
        '- **Image size:** runtime increased in every measured size comparison, but '
        'not proportionally to pixels. From 608→1536, pixels grow 6.38×; single-GPU '
        'resident runtime grows 4.09× for Remora/PyTorch and 4.38× for optimized tinygrad.',
        '- **Size-dependent exceptions:** optimized tinygrad takes only 1.22× longer '
        'from 960→1216, then 1.83× longer from 1216→1536, despite both steps adding '
        'about 1.60× pixels. Kernel-level causes remain untested.',
        '- **Transfers:** PyTorch’s 1216→1536 resident runtime grows 1.57×, but '
        'transfer-inclusive runtime grows 2.70×. Upload/download and host-copy costs '
        'need separate investigation.',
        '- **GPU count:** more GPUs helped every measured case. Resident 1→8 speedups '
        'are 7.88–8.00×; Darknet transfer-inclusive reaches only 6.74×. More throughput '
        'does not imply lower single-image latency or better cost efficiency.',
        '- **Memory:** PyTorch peaks near 78 GB at 1216, then drops near 45 GB at 1536; '
        'its allocator behavior remains unexplained. Replication leaves per-GPU memory '
        'unchanged, so more GPUs do not solve one image’s memory demand.',
        '- **Correctness:** passing 608 did not predict passing larger sizes for baseline '
        'tinygrad or Darknet. These numerical mismatches do not establish worse detection accuracy.',
        '- **Backend choice:** generated Python matches direct Python. Good GPU scaling '
        'does not rescue Futhark’s slow per-image execution in this workload.', '',
        'These conclusions cover uniform, batch-one replicas on this machine. '
        'Model/tensor partitioning and real-image accuracy remain untested.', '']
    write_csv('cross-implementation.csv', summary_rows)
    (OUT/'COMPARISON.md').write_text('\n'.join(lines))
    print(OUT/'COMPARISON.md')

if __name__ == '__main__':
    main()
