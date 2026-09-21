"""Sample per-process GPU footprint with NVIDIA's driver tool in a separate run.

Reported MiB include CUDA context, libraries, allocator caches and autotuning.
Sampling can miss short peaks; this is not framework live-tensor allocation.
"""
import argparse,json,os,subprocess,threading,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();samples=[];phase={'value':'setup_and_warmup'}
    with out.with_suffix('.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        monitor=subprocess.Popen(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits','--loop-ms=20'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        def read_gpu():
            for line in monitor.stdout:
                fields=line.strip().split(',')
                if len(fields)!=2:continue
                try:pid,mib=int(fields[0]),int(fields[1])
                except ValueError:continue
                if pid!=proc.pid:continue
                rss=hwm=None
                try:
                    status=Path(f'/proc/{pid}/status').read_text().splitlines()
                    for entry in status:
                        if entry.startswith('VmRSS:'):rss=int(entry.split()[1])/1024
                        if entry.startswith('VmHWM:'):hwm=int(entry.split()[1])/1024
                except FileNotFoundError:pass
                samples.append({'seconds':time.monotonic()-start,'phase':phase['value'],'gpu_mib':mib,'host_rss_mib':rss,'host_hwm_mib':hwm})
        thread=threading.Thread(target=read_gpu,daemon=True);thread.start()
        for line in proc.stdout:
            log.write(line);log.flush()
            if line.startswith('MEMORY_PHASE '):phase['value']=line.split()[1]
        code=proc.wait();monitor.terminate();thread.join(timeout=5);monitor.wait()
    record={'command':command,'pid':proc.pid,'exit_code':code,'sampling_requested_ms':20,'elapsed_seconds':time.monotonic()-start,'samples':samples,
        'gpu_process_peak_mib':max((s['gpu_mib'] for s in samples),default=None),
        'gpu_setup_inference_peak_mib':max((s['gpu_mib'] for s in samples if s['phase']!='validation'),default=None),
        'host_peak_rss_mib':max((s['host_hwm_mib'] for s in samples if s['host_hwm_mib'] is not None),default=None),
        'phase_gpu_peak_mib':{phase:max(s['gpu_mib'] for s in samples if s['phase']==phase) for phase in {s['phase'] for s in samples}},
        'definition':'Sampled per-process GPU footprint including context, libraries, reserved caches and setup/autotuning; timings from this instrumented run are not the comparison timings.'}
    out.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(out,record['gpu_process_peak_mib'],flush=True)
    if code:raise SystemExit(code)
    if not samples:raise RuntimeError('No GPU process memory samples captured')
if __name__=='__main__':main()
