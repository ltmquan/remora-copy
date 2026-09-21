"""Read-only rental inventory. Run before CUDA benchmarks."""
import argparse, importlib.metadata, json, os, platform, shutil, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('preflight.json')); a=p.parse_args()
    result={'platform':platform.platform(),'python':platform.python_version(),'cpu_count':os.cpu_count(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'disk_free_bytes':shutil.disk_usage('.').free}
    if Path('/proc/meminfo').exists(): result['host_memory']=Path('/proc/meminfo').read_text()
    for package in ('torch','tinygrad','numpy'):
        try: result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[package]=None
    for name,cmd in [('gpus',['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id','--format=csv']),('topology',['nvidia-smi','topo','-m']),('processes',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv']),('cuda_compiler',['nvcc','--version'])]:
        try:
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            result[name]={'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
        except (OSError,subprocess.TimeoutExpired) as e: result[name]={'error':str(e)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(a.output)
if __name__=='__main__': main()
