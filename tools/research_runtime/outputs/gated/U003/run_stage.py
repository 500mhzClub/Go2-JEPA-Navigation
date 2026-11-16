"""Run unchanged D001 with U001 accounting and reviewed full core evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path.cwd().resolve();sys.path.insert(0,str(ROOT))
from go2nav.research import atomic_json,digest,require_record,require_review,require_training_slot,verify_files
from common import BASE,admit,arguments,core_reviews,parents,progress_paths
stages,supervision,unlimited,source_reconciliation=parents()
consultation,generated,inventory,paths,stage_parents=stages.consultation,stages.generated,stages.inventory,stages.paths,stages.stage_parents
close_intervals,stop_owned=supervision.close_intervals,supervision.stop_owned


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['export','s1','compare'])
    parser.add_argument('--python',required=True);parser.add_argument('--preconsultation',required=True);args=parser.parse_args()
    require_training_slot();_,rh=require_record('D003');_,dh=require_record('D001');_,uh=require_record('U003');_,ah=require_record('U001')
    base=ROOT/BASE;assert require_review(base/'software/review.json')['status']=='passed'
    sw=json.loads((base/'software/result.json').read_text())
    assert sw['research_sha256']==uh and sw['d001_record_sha256']==dh
    verify_files(base,sw['source_sha256']);verify_files(ROOT,sw['repository_source_sha256'])
    verify_files(ROOT,sw['config_sha256']);verify_files(ROOT,sw['frozen_model_sha256'])
    assert digest(ROOT/'outputs/gated/A016/supervision.py')==sw['a016_supervision_sha256']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True)
    p=paths(ROOT,args.stage);reviews=core_reviews(ROOT)+stage_parents(p);files=inventory(p)
    pre=consultation(args.preconsultation,p,files,reviews,rh,dh)
    assert pre['u003_record_sha256']==uh and pre['u001_authorization_sha256']==ah and pre['software_review_sha256']==digest(base/'software/review.json')
    child_out,out=p['science'],p['operations'];assert not child_out.exists() and not out.exists()
    if args.stage=='export':
        assert not any((p['corpus']/f).exists() for f in ['tracks_config.json','tracks_index.json'])
        assert not any(f.is_file() for folder in ['objects','engineered'] for f in (p['corpus']/folder).rglob('*'))
    gpu=subprocess.run(['amd-smi','process','--json'],capture_output=True,text=True,timeout=10,check=True)
    assert not any(isinstance(p['process_info'],dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
    argv=arguments(args.python,p,stages);env=dict(os.environ)
    for key in ['PYTHONPATH','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_ENABLE_SDMA']:env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_TUNABLEOP_ENABLED='0')
    ledger=(ROOT/'outputs/gated/gpu_intervals.json').resolve();env['GO2NAV_GPU_LEDGER']=str(ledger)
    manifest=dict(research_id='D003',integration_research_id='U003',integration_research_sha256=uh,budget_policy='U001',authorization_sha256=ah,stage=args.stage,research_sha256=rh,d001_record_sha256=dh,argv=argv,cwd=str(ROOT),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sw['source_sha256'],software_review_sha256=digest(base/'software/review.json'),
        preconsultation=pre,preconsultation_sha256=digest(args.preconsultation),cleanup_reserve_seconds=45)
    out.mkdir(parents=True,exist_ok=False);(out/'process_before.json').write_text(gpu.stdout)
    began=time.monotonic();outer=admit(ledger,out)
    manifest.update(outer_interval_id=outer['id'],admitted_at=outer['start'],work_seconds=None,stall_seconds=1800,status='running')
    child=process=identity=None;actions=[];errors=[];closed=[];timed_out=False
    try:
        atomic_json(out/'manifest.json',manifest)
        with (out/'stage.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process=psutil.Process(child.pid);identity=process.create_time()
            manifest.update(pid=child.pid,creation_time=identity);atomic_json(out/'manifest.json',manifest)
            timed_out=unlimited.wait_progress(child,process,identity,argv,ROOT,progress_paths(p,out),actions,
                heartbeat=lambda seconds:atomic_json(out/'monitor_latest.json',dict(at=time.time(),pid=child.pid,creation_time=identity,seconds_without_output=seconds,budget_policy='U001')))
    except BaseException as exc:errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or identity is None:raise RuntimeError('Owned child identity unavailable')
                stop_owned(child,process,identity,argv,ROOT,actions)
            except BaseException as exc:errors.append(repr(exc))
        try:closed=close_intervals(ledger,outer['id'],child,child_out,ROOT)
        except BaseException as exc:errors.append(repr(exc))
    terminal=child is None or child.poll() is not None;code=None if child is None else child.returncode
    raw_status=None;artifacts={};inputs_unchanged=False
    if terminal:
        try:
            if (child_out/'result.json').exists():raw_status=json.loads((child_out/'result.json').read_text()).get('status')
            artifacts={str(f.relative_to(ROOT)):digest(f) for f in generated(p)}
            verify_files(ROOT,pre['inputs_sha256']);inputs_unchanged=True
        except BaseException as exc:errors.append(repr(exc))
    status='awaiting_literature_review'
    if errors or code!=0 or raw_status not in ['awaiting_literature_review','budget_incomplete']:status='technical_blocked'
    if timed_out:status='technical_blocked'
    manifest.update(status=status,finished_at=time.time(),elapsed_seconds=time.monotonic()-began)
    atomic_json(out/'manifest.json',manifest)
    result=dict(research_id='D003',integration_research_id='U003',budget_policy='U001',timeout_type='no_output_progress' if timed_out else None,stage=args.stage,status=status,returncode=code,child_terminal=terminal,timed_out=timed_out,
        signals=actions,errors=errors,raw_status=raw_status,closed_interval_ids=closed,inputs_unchanged=inputs_unchanged,
        elapsed_seconds=manifest['elapsed_seconds'],manifest_sha256=digest(out/'manifest.json'),child_artifact_sha256=artifacts,
        next_step='Inspect and literature-review actual D001 science and D003 operations separately; no automatic retry, review or successor.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if status!='awaiting_literature_review':raise SystemExit(1)


if __name__=='__main__':main()
