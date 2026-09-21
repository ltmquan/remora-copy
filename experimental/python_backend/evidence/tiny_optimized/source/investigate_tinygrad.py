"""Record BEAM entry/exit events or choose an opt-in lowering for benchmark.py."""
import json,os,time
import benchmark

def main():
    variant=os.getenv('TINYGRAD_VARIANT','baseline')
    if variant!='baseline':
        from tinygrad_optimized import OptimizedOps
        benchmark.Ops=lambda backend,device,fold_bn=False:OptimizedOps(device,variant)
    if os.getenv('TRACE_BEAM'):
        import tinygrad.codegen.opt.search as search
        from tinygrad.helpers import PARALLEL
        original=search.beam_search;events=[]
        def traced(s,*args,**kwargs):
            start=time.monotonic();idx=len(events);events.append(None)
            print('BEAM_EVENT '+json.dumps({'event':'start','index':idx,'shape':str(s.full_shape),'workers':int(PARALLEL.value)}),flush=True)
            result=original(s,*args,**kwargs)
            print('BEAM_EVENT '+json.dumps({'event':'finish','index':idx,'seconds':time.monotonic()-start}),flush=True)
            return result
        search.beam_search=traced
    benchmark.main()
if __name__=='__main__':main()
