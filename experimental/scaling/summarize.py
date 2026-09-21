"""CSV with separate size and GPU scaling columns; no timings from memory passes."""
import argparse,csv,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 records=[]
 for path in sorted(a.directory.glob('*.json')):
  data=json.loads(path.read_text())
  if 'config' in data and 'implementation' in data['config']:records.append(data)
 def key(d,size=None,ngpu=None,policy=None):
  c=d['config'];return (c['implementation'],c['size'] if size is None else size,len(c['gpus']) if ngpu is None else ngpu,c['policy'] if policy is None else policy,c['mode'],c['images'],d.get('reference',{}).get('checkpoint_sha256'),d.get('reference',{}).get('reference'))
 baseline={key(d):d['median_images_per_second'] for d in records if d['status']=='passed' and d.get('measurement_kind')=='timing'}
 rows=[]
 for d in records:
  c=d['config']; rate=d.get('median_images_per_second') if d.get('measurement_kind')=='timing' else None
  one=baseline.get(key(d,ngpu=1,policy='uniform'));small=baseline.get(key(d,size=608))
  footprint=[t['footprint'] for t in d.get('trials',[]) if t.get('footprint')]
  peaks=[f['simultaneous_process_peak_bytes'] for f in footprint if f['simultaneous_process_peak_bytes'] is not None]
  memories=[w['memory'] for t in d.get('trials',[]) for w in t.get('workers',[]) if w.get('memory')]
  live=[m['live_peak_bytes'] for m in memories if m.get('live_peak_bytes') is not None]
  reserved=[m['reserved_peak_bytes'] for m in memories if m.get('reserved_peak_bytes') is not None]
  violations=[v['strict_violations'] for w in d.get('workers',[]) for v in w.get('validation',[])]
  rows.append({'reference':d.get('reference',{}).get('reference'),'validation_atol':c.get('atol',1e-5),'validation_rtol':c.get('rtol',1e-4),'strict_violations_max_per_input':max(violations) if violations else None,'max_worker_live_peak_bytes':max(live) if live else None,'max_worker_reserved_peak_bytes':max(reserved) if reserved else None,'implementation':c['implementation'],'image_side':c['size'],'gpus':len(c['gpus']),'partition':c['policy'],'mode':c['mode'],'requests':c['images'],'kind':d.get('measurement_kind'),'status':d['status'],'images_per_second':rate,'gpu_speedup_vs_1':rate/one if rate and one else None,'gpu_scaling_efficiency':rate/one/len(c['gpus']) if rate and one else None,'throughput_ratio_vs_608':rate/small if rate and small else None,'pixel_count_ratio_vs_608':(c['size']/608)**2,'simultaneous_process_peak_bytes':max(peaks) if peaks else None})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:
  if rows:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(a.output)
if __name__=='__main__':main()
