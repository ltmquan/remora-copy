"""CPU postprocessing of saved outputs; DARPA do_nms_sort thresholds/rules."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parent
variants=['darknet','fresh_futhark','modified_futhark','torch-remora','torch-direct','baseline-remora','baseline-direct','im2col-remora','im2col-direct']
cases=['black','white','noise','gradient','checkerboard','evalbox']
def iou(a,b):
    lo=np.maximum(a[:2]-a[2:]/np.float32(2),b[:2]-b[2:]/np.float32(2))
    hi=np.minimum(a[:2]+a[2:]/np.float32(2),b[:2]+b[2:]/np.float32(2))
    size=np.maximum(hi-lo,np.float32(0));inter=size[0]*size[1]
    union=a[2]*a[3]+b[2]*b[3]-inter
    return float(inter/union) if inter and union else 0.
def nms(a):
    scores=a[:,4,None]*a[:,5:];out=[]
    for cls in range(80):
        ids=np.flatnonzero((a[:,4]>.25)&(scores[:,cls]>.25))
        # No ambiguous equal-score candidates above threshold in these fixtures.
        assert len(np.unique(scores[ids,cls]))==len(ids)
        ids=ids[np.argsort(-scores[ids,cls],kind='stable')].tolist()
        while ids:
            idx=ids.pop(0)
            out.append({'candidate':idx,'class_id':cls,'confidence':float(scores[idx,cls]),'xywh':a[idx,:4].tolist()})
            ids=[j for j in ids if iou(a[idx,:4],a[j,:4])<=.45]
    return out
all_results={};comparisons=[]
for case in cases:
    all_results[case]={}
    for variant in variants:
        a=np.fromfile(root/'results'/f'{variant}-{case}.bin',dtype='<f4').reshape(22743,85)
        assert np.isfinite(a).all()
        all_results[case][variant]=nms(a)
    ref=all_results[case]['darknet'];refkeys={(d['candidate'],d['class_id']):d for d in ref}
    for variant in variants[1:]:
        ds=all_results[case][variant];keys={(d['candidate'],d['class_id']):d for d in ds};common=keys.keys()&refkeys.keys()
        comparisons.append({'input':case,'pipeline':variant,'detections':len(ds),'darknet_detections':len(ref),'same_surviving_candidates_and_classes':keys.keys()==refkeys.keys(),'added':sorted(keys.keys()-refkeys.keys()),'removed':sorted(refkeys.keys()-keys.keys()),'max_confidence_difference':max((abs(keys[k]['confidence']-refkeys[k]['confidence']) for k in common),default=None),'max_xywh_difference_608_pixels':max((float(np.max(np.abs(np.array(keys[k]['xywh'])-refkeys[k]['xywh']))*608) for k in common),default=None),'minimum_matched_iou':min((iou(np.array(keys[k]['xywh'],dtype=np.float32),np.array(refkeys[k]['xywh'],dtype=np.float32)) for k in common),default=None)})
result={'confidence_threshold':.25,'nms_iou_threshold':.45,'method':'Class-wise greedy IoU NMS on objectness*class probability; float32, no clipping, normalized boxes in resized 608x608 image; rules from DARPA do_nms_sort. No ground-truth evaluation.','detections':all_results,'comparisons':comparisons}
(root/'results/nms-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for case in cases:
 print(case,'Darknet detections:',all_results[case]['darknet'])
print('Differences:',*[c for c in comparisons if not c['same_surviving_candidates_and_classes']],sep='\n')
print('Photo comparisons:',*[c for c in comparisons if c['input']=='evalbox'],sep='\n')
