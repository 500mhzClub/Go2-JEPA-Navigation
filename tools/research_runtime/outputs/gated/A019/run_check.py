"""External deadline for unchanged additional-seed numerical evaluation; no auto-review."""
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
from supervision import admit,close_intervals,remaining,stop_owned,wait_owned
from stages import arguments,consultation,inputs,route


def main():
    parser=argparse.ArgumentParser()
    for name in ['run','python','preconsultation']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=route(ROOT,args.run);require_training_slot()
    _,rh=require_record('A019');_,sh=require_record(p['science_id']);base=ROOT/'outputs/gated/A019'
    assert require_review(base/'software/review.json')['status']=='passed'
    sw=json.loads((base/'software/result.json').read_text());assert sw['research_sha256']==rh
    verify_files(base,sw['source_sha256']);verify_files(ROOT,sw['repository_source_sha256']);verify_files(ROOT,sw['record_sha256'])
    assert not subprocess.check_output(['git','status','--porcelain'],text=True)
    files=inputs(p);pre=consultation(args.preconsultation,p,files,rh,sh)
    out,science=p['operations'],p['science'];assert not out.exists() and not science.exists()
    gpu=subprocess.run(['amd-smi','process','--json'],capture_output=True,text=True,timeout=10,check=True)
    assert not any(isinstance(item['process_info'],dict) for g in json.loads(gpu.stdout) for item in g['process_list'])
    argv=arguments(args.python,p);env=dict(os.environ)
    for key in ['PYTHONPATH','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_ENABLE_SDMA']:env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_TUNABLEOP_ENABLED='0')
    ledger=(ROOT/'outputs/gated/gpu_intervals.json').resolve();env['GO2NAV_GPU_LEDGER']=str(ledger)
    out.mkdir(parents=True,exist_ok=False);(out/'process_before.json').write_text(gpu.stdout)
    began=time.monotonic();outer,left,work=admit(ledger,out,p['science_id'])
    manifest=dict(research_id='A019',scientific_research_id=p['science_id'],research_sha256=rh,scientific_research_sha256=sh,
        stage=p['stage'],name=p['name'],argv=argv,cwd=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        software_review_sha256=digest(base/'software/review.json'),source_sha256=sw['source_sha256'],preconsultation=pre,
        preconsultation_sha256=digest(args.preconsultation),outer_interval_id=outer['id'],admitted_at=outer['start'],
        remaining_at_admission=left,work_seconds=work,per_call_seconds=1800,cleanup_reserve_seconds=45,status='running')
    child=process=created=None;actions=[];errors=[];closed=[];timed=False
    try:
        atomic_json(out/'manifest.json',manifest)
        with (out/'evaluation.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process=psutil.Process(child.pid);created=process.create_time();manifest.update(pid=child.pid,creation_time=created);atomic_json(out/'manifest.json',manifest)
            timed=wait_owned(child,process,created,argv,ROOT,began+work,actions,
                remaining=lambda:remaining(json.loads(ledger.read_text()),time.time(),p['science_id']),
                heartbeat=lambda seconds:atomic_json(out/'monitor_latest.json',dict(at=time.time(),pid=child.pid,creation_time=created,remaining_work_seconds=seconds)))
    except BaseException as exc:errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or created is None:raise RuntimeError('Owned identity unavailable')
                stop_owned(child,process,created,argv,ROOT,actions)
            except BaseException as exc:errors.append(repr(exc))
        try:closed=close_intervals(ledger,outer['id'],child,science,ROOT,p['science_id'])
        except BaseException as exc:errors.append(repr(exc))
    terminal=child is None or child.poll() is not None;code=None if child is None else child.returncode
    raw_status=None;artifacts={};unchanged=False
    if terminal:
        try:
            if (science/'result.json').exists():raw_status=json.loads((science/'result.json').read_text()).get('status')
            artifacts={str(f.relative_to(science)):digest(f) for f in science.rglob('*') if f.is_file()}
            verify_files(ROOT,pre['inputs_sha256']);unchanged=True
        except BaseException as exc:errors.append(repr(exc))
    status='awaiting_literature_review'
    if errors or code!=0 or raw_status not in ['awaiting_literature_review','budget_incomplete']:status='technical_blocked'
    if terminal and not errors and (timed or (code==0 and raw_status=='budget_incomplete')):status='budget_incomplete'
    manifest.update(status=status,finished_at=time.time(),elapsed_seconds=time.monotonic()-began);atomic_json(out/'manifest.json',manifest)
    result=dict(research_id='A019',scientific_research_id=p['science_id'],stage=p['stage'],name=p['name'],status=status,
        returncode=code,child_terminal=terminal,timed_out=timed,actions=actions,errors=errors,closed_interval_ids=closed,
        raw_status=raw_status,inputs_unchanged=unchanged,child_artifact_sha256=artifacts,elapsed_seconds=manifest['elapsed_seconds'],
        manifest_sha256=digest(out/'manifest.json'),next_step='Inspect numerical and derived artifact evidence and separately literature-review science and operations before probes/planning; no automatic successor/retry.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if status!='awaiting_literature_review':raise SystemExit(1)


if __name__=='__main__':main()
