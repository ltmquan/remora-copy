"""Optional separate memory pass. Sampling perturbs timings; do not mix its rates."""
import subprocess, threading, time

class Sampler:
    def __init__(self,pids):
        self.pids=set(pids); self.stop_event=threading.Event(); self.samples=[]; self.errors=[]
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop_event.is_set():
            try:
                output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.STDOUT,timeout=5)
                devices={}
                for line in output.splitlines():
                    parts=[x.strip() for x in line.split(',')]
                    if len(parts)==3 and int(parts[0]) in self.pids:
                        devices[parts[1]]=devices.get(parts[1],0)+int(parts[2])*1024**2
                if devices: self.samples.append({'time':time.monotonic(),'bytes_by_gpu':devices})
                else: self.errors.append('No tracked GPU process found in sample')
            except Exception as e: self.errors.append(str(e))
            self.stop_event.wait(.05)
    def start(self): self.thread.start()
    def finish(self):
        self.stop_event.set(); self.thread.join(timeout=6)
        peaks={}
        for sample in self.samples:
            for gpu,value in sample['bytes_by_gpu'].items(): peaks[gpu]=max(peaks.get(gpu,0),value)
        return {'process_peak_bytes_by_gpu':peaks,'simultaneous_process_peak_bytes':max((sum(s['bytes_by_gpu'].values()) for s in self.samples),default=None),'samples':len(self.samples),'errors':self.errors,'scope':'warmed inference; excludes setup; sampled nvidia-smi process footprint; short peaks may be missed'}
