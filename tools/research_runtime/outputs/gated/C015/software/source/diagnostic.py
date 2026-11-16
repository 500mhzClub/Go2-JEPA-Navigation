"""C015 synchronized original/transfer-variant traces; no scientific planning outcome."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from go2nav.eval import balls_planning as planning
from go2nav.eval.slots_inference import load_reviewed_world
from go2nav.data.balls import load as load_data
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
from common import BASE,HELPERS,inputs,need
from costs import VARIANTS,matrices


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-seconds',type=float,required=True);a=p.parse_args();began=time.monotonic()
    _,rh=require_record('C015');sw=json.loads((BASE/'software/result.json').read_text());need(require_review(BASE/'software/review.json')['status']=='passed','Unreviewed software');verify_files(BASE,sw['helper_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256'])
    operation=json.loads((BASE/'operations/manifest.json').read_text());verify_files(Path.cwd(),operation['inputs']);out=BASE/'diagnostic';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    original_distance,original_cem=planning.slot_distance,planning.cem
    funcs={'original':original_distance,**VARIANTS};cases=[];arrays={};qualified={k:True for k in VARIANTS}
    atomic_json(out/'manifest.json',dict(research_id='C015',research_sha256=rh,operation_manifest_sha256=digest(BASE/'operations/manifest.json'),inputs=operation['inputs'],helper_sha256=sw['helper_sha256'],source_commit=operation['source_commit']))
    try:
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for name in ['o0_n3_s0','o1_n3_s0']:
                model=load_reviewed_world(Path('outputs/gated/C001')/name/'model.pt',Path('outputs/gated/C014')/name/'review.json')
                for split in ['train','val']:
                    frames,labels=load_data('outputs/data/balls',3,split)
                    for h in [4,6,8]:
                        t=planning.task_start(3,0,h,len(frames[0]));inp=planning.task_inputs(frames,labels['actions'],0,t,h);seed=[2026111504,3,0,h];traces={};measurements={}
                        for mode,fn in funcs.items():
                            if time.monotonic()-began>a.work_seconds:raise TimeoutError('Diagnostic work allocation exhausted')
                            planning.slot_distance=fn;planning.cem=original_cem
                            torch.cuda.synchronize();start=time.perf_counter();actions=planning.learned_actions(model,inp,h,seed);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                            trace={'proposals':[],'costs':[],'order':[],'matrices':[]}
                            def traced_distance(predicted,goal):
                                _,flat=matrices(predicted,goal);trace['matrices'].append(flat.cpu().numpy().copy());return fn(predicted,goal)
                            def traced_cem(cost,horizon,seed):
                                def observed(proposals):
                                    values=cost(proposals);trace['proposals'].append(proposals.copy());trace['costs'].append(values.copy());trace['order'].append(np.argsort(values,kind='stable'));return values
                                return original_cem(observed,horizon,seed)
                            planning.slot_distance=traced_distance;planning.cem=traced_cem
                            traced_actions=planning.learned_actions(model,inp,h,seed);torch.cuda.synchronize();need(np.array_equal(actions,traced_actions),'Tracing changes selected actions')
                            trace={k:np.stack(v) for k,v in trace.items()};trace['actions']=actions;need(trace['costs'].shape==(20,100) and trace['proposals'].shape==(20,100,h,2),'Wrong CEM trace coverage')
                            file=out/f'{name}_{split}_h{h}_{mode}.npz';np.savez_compressed(file,**trace);arrays[file.name]=digest(file);traces[mode]=trace
                            measurements[mode]=dict(elapsed_seconds=elapsed,score=planning.score(labels['red'][0,t],labels['red'][0,t+h],actions,labels['white'][0,t+1:t+h+1]))
                        ref=traces['original'];comparisons={}
                        for mode in VARIANTS:
                            tr=traces[mode];checks={k:bool(np.array_equal(tr[k],ref[k])) for k in ['proposals','order','actions']};checks['costs']=bool(np.allclose(tr['costs'],ref['costs'],rtol=1e-12,atol=1e-12));checks['physical_score']=measurements[mode]['score']==measurements['original']['score'];checks['finite']=all(np.isfinite(v).all() for v in tr.values());qualified[mode]&=all(checks.values());comparisons[mode]=dict(checks=checks,max_cost_abs_difference=float(np.max(np.abs(tr['costs']-ref['costs']))),speedup=measurements['original']['elapsed_seconds']/measurements[mode]['elapsed_seconds'])
                        row=dict(name=name,split=split,episode=0,horizon=h,start=t,seed=seed,measurements=measurements,comparisons=comparisons);cases.append(row);atomic_json(out/'progress.json',dict(completed_cases=len(cases),latest=row));print(json.dumps(row),flush=True)
                del model
            verify_files(Path.cwd(),operation['inputs']);verify_files(BASE,sw['helper_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256'])
            result=dict(research_id='C015',research_sha256=rh,status='awaiting_literature_review',complete=len(cases)==12,cases=cases,qualified=qualified,arrays_sha256=arrays,inputs_unchanged=True,elapsed_seconds=time.monotonic()-began,manifest_sha256=digest(out/'manifest.json'),limitation='One train and validation episode per selected model, all3 horizons; numerical and runtime diagnosis only. No full scientific planning outcome or general speed guarantee. Separate trace passes add instrumentation overhead; reported times are uninstrumented synchronized passes.')
            atomic_json(out/'result.json',result)
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='C015',status='budget_incomplete' if isinstance(exc,TimeoutError) else 'technical_blocked',complete=False,cases=cases,error=repr(exc),arrays_sha256=arrays,elapsed_seconds=time.monotonic()-began));raise
    finally:planning.slot_distance,planning.cem=original_distance,original_cem

if __name__=='__main__':main()
