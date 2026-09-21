from pathlib import Path
root=Path(__file__).resolve().parent
base=Path('/scratch/luu.qu/remora-python-20260911')
# Keep existing network setup, operators, compilation and capture modes. Replace only the timing/check tail.
s=(base/'benchmark.py').read_text();assert s.count('    warm=[]')==1
s=s.split('    warm=[]')[0]+'''
    cases=json.loads((Path(os.environ['VALIDATION_ROOT'])/'inputs.json').read_text())['images']
    for case in cases:
        image=np.fromfile(Path(os.environ['VALIDATION_ROOT'])/'inputs'/ (case['name']+'.bin'),dtype='<f4').reshape(3,608,608)
        tensor=ops.tensor(image)
        # TinyJit captures on the second call; exercise replay and changed inputs.
        for _ in range(3):
            yy=run(tensor); ops.sync()
        result=ops.numpy(yy).astype('<f4')
        assert result.shape==(22743,85) and np.isfinite(result).all()
        result.tofile(out.parent/(out.name+'-'+case['name']+'.bin'))
        print('validated output '+case['name'],flush=True)
    out.with_suffix('.json').write_text(json.dumps(vars(a)|{'inputs':cases,'cudnn_enabled':False if a.backend=='torch' else None},indent=2)+'\\n')
if __name__=='__main__':main()
'''
s=s.replace("ROOT=Path(__file__).resolve().parent",f"ROOT=Path({str(base)!r})")
s=s.replace('        import torch\n','        import torch\n        torch.backends.cudnn.enabled=False\n')
s=s.replace('from runtime import Ops', "from runtime import Ops\nif os.environ.get('TINYGRAD_VARIANT')=='im2col':\n    from tinygrad_optimized import OptimizedOps\n    Ops=lambda backend,device,fold_bn=False:OptimizedOps(device,'im2col')")
(root/'python_validate.py').write_text(s)
# Preserve decode and model logic, accepting raw CHW input instead of the fixture.
s=(base/'darknet_benchmark.cpp').read_text()
s=s.replace('if(argc!=3 && argc!=4)return 1;','if(argc!=5)return 1;').replace('if(argc==4)load_weights','if(argc>=4)load_weights')
a='std::vector<float> image(3*608*608);for(size_t j=0;j<image.size();j++)image[j]=(float)(j%251)/250.0f;'
assert s.count(a)==1
s=s.replace(a,'std::vector<float> image(3*608*608); FILE *im=fopen(argv[4],"rb");if(!im)return 9;if(fread(image.data(),sizeof(float),image.size(),im)!=image.size())return 10;fclose(im);')
assert s.count('for(int run=-1;run<10;run++){')==1
s=s.replace('for(int run=-1;run<10;run++){','for(int run=0;run<1;run++){')
(root/'darknet.cpp').write_text(s)
for variant in ('fresh_futhark','modified_futhark'):
    s=(base/variant/'harness.c').read_text()
    assert s.count('run=-3;run<10')==2 and s.count('if(run==9)')==1
    s=s.replace('run=-3;run<10','run=0;run<1').replace('if(run==9)','if(run==0)')
    p=root/variant;p.mkdir(exist_ok=True);(p/'harness.c').write_text(s)
