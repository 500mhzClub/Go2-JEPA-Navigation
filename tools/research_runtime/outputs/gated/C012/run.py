"""One C012 diagnostic, charged to C004 reserve and shared allowance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from common import BASE, NAME, admit, close, inherited, inputs
from go2nav.research import atomic_json,digest,require_record,require_review,require_training_slot,verify_files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--python',required=True);parser.add_argument('--preconsultation',required=True)
    args=parser.parse_args();require_training_slot();record,rh=require_record('C012')
    assert require_review(BASE/'software/review.json')['status']=='passed'
    software=json.loads((BASE/'software/result.json').read_text());assert software['research_sha256']==rh
    verify_files(BASE,software['source_sha256']);verify_files(Path.cwd(),software['repository_source_sha256'])
    bound,_=inputs();pre=json.loads(Path(args.preconsultation).read_text())
    assert pre['research_sha256']==rh and pre['inputs_sha256']==bound and pre['software_review_sha256']==digest(BASE/'software/review.json')
    assert pre['consulted_at'] and pre['consequences'] and pre['sources']
    for s in pre['sources']:assert all(s.get(k) for k in ['url','version','sections','consequence'])
    assert not subprocess.check_output(['git','status','--porcelain'],text=True)
    out=BASE/'operations'/NAME;science=BASE/NAME
    assert not out.exists() and not science.exists()
    gpu=subprocess.run(['amd-smi','process','--json'],capture_output=True,text=True,check=True,timeout=10)
    assert not any(isinstance(p['process_info'],dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
    out.mkdir(parents=True,exist_ok=False);(out/'process_before.json').write_text(gpu.stdout)
    ledger=Path('outputs/gated/gpu_intervals.json').resolve();row,remaining,work=admit(ledger,out)
    argv=[args.python,str(BASE/'diagnostic.py'),'--out',str(science),'--work-seconds',str(work)]
    env=dict(os.environ)
    for key in ['PYTHONPATH','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','HSA_ENABLE_SDMA']:env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_TUNABLEOP_ENABLED='0')
    manifest=dict(research_id='C012',research_sha256=rh,argv=argv,cwd=str(Path.cwd()),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=software['source_sha256'],
        software_review_sha256=digest(BASE/'software/review.json'),inputs=bound,preconsultation=pre,
        preconsultation_sha256=digest(args.preconsultation),outer_interval_id=row['id'],remaining=remaining,work_seconds=work)
    child=proc=identity=None;actions=[];errors=[];timed=False;started=time.monotonic();base=inherited()
    try:
        atomic_json(out/'manifest.json',manifest)
        with (out/'diagnostic.log').open('xb') as log:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            proc=psutil.Process(child.pid);identity=proc.create_time();manifest.update(pid=child.pid,creation_time=identity)
            atomic_json(out/'manifest.json',manifest)
            timed=base.wait_owned(child,proc,identity,argv,Path.cwd(),started+work,actions,
                remaining=lambda:base.remaining_seconds(json.loads(ledger.read_text()),time.time()),
                heartbeat=lambda seconds:atomic_json(out/'monitor_latest.json',dict(at=time.time(),pid=child.pid,remaining_work_seconds=seconds)))
    except BaseException as exc:errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:base.stop_owned(child,proc,identity,argv,Path.cwd(),actions)
            except BaseException as exc:errors.append(repr(exc))
        try:close(ledger,row['id'],child)
        except BaseException as exc:errors.append(repr(exc))
    raw=json.loads((science/'result.json').read_text()) if (science/'result.json').exists() else {}
    code=None if child is None else child.returncode
    status='awaiting_literature_review' if code==0 and not errors and raw.get('complete') else 'technical_blocked'
    if not errors and (timed or raw.get('status')=='budget_incomplete'):status='budget_incomplete'
    result=dict(research_id='C012',name=NAME,status=status,returncode=code,child_terminal=child is None or child.poll() is not None,
        timed_out=timed,actions=actions,errors=errors,elapsed_seconds=time.monotonic()-started,manifest_sha256=digest(out/'manifest.json'),
        child_artifact_sha256={str(p.relative_to(science)):digest(p) for p in sorted(science.glob('*')) if p.is_file()},
        limitation='Operation only; diagnostic and any successor require separate manual literature review.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if status!='awaiting_literature_review':raise SystemExit(1)


if __name__=='__main__':main()
