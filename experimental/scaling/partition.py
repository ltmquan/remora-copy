"""Deterministic assignments of whole, independent inference requests."""
import math

def assignments(count, workers, policy, rates=None):
    if count < 1 or workers < 1: raise ValueError('Positive requests and workers required')
    if policy=='round-robin': return [list(range(i,count,workers)) for i in range(workers)]
    if policy not in ('uniform','weighted'): raise ValueError(policy)
    weights = rates if policy=='weighted' else [1]*workers
    if weights is None or len(weights)!=workers or any(not math.isfinite(x) or x<=0 for x in weights):
        raise ValueError('One finite positive calibration rate per GPU required')
    quotas=[count*w/sum(weights) for w in weights]; sizes=[math.floor(q) for q in quotas]
    for i in sorted(range(workers),key=lambda i:(-(quotas[i]-sizes[i]),i))[:count-sum(sizes)]: sizes[i]+=1
    start=0; result=[]
    for size in sizes: result.append(list(range(start,start+size))); start+=size
    return result
