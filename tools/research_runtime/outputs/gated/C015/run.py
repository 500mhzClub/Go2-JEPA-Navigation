"""One C015 diagnostic under inherited owned-child supervision and reserve cap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from common import BASE,admit,close,inherited,inputs,need
from go2nav.research import atomic_json,digest,require_record,require_review,require_training_slot,verify_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--preconsultation',required=True);a=p.parse_args();require_training_slot();_,rh=require_record('C015')
    sw=json.loads((BASE/'software/result.json').read_text());need(require_review(BASE/'software/review.json')['status']=='passed' and sw['research_sha256']==rh,'Software not reviewed');verify_files(BASE,sw['helper_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256'])
    bound=inputs();pre=json.loads(Path(a.preconsultation).read_text());need(pre['research_sha256']==rh and pre['inputs_sha256']==bound and pre['software_review_sha256']==digest(BASE/'software/review.json'),'Changed consultation');need(pre.get('sources') and pre.get('consulted_at') and pre.get('consequences'),'Missing consultation');need(not subprocess.check_output(['git','status','--porcelain'],text=True),'Dirty source')
    out=BASE/'operations';science=BASE/'diagnostic';need(not out.exists() and not science.exists(),'Output already exists');gpu=subprocess.run(['amd-smi','process','--json'],capture_output=True,text=True,check=True,timeout=10);need(not any(isinstance(p['process_info'],dict) for g in json.loads(gpu.stdout) for p in g['process_list']),'GPU occupied')
    out.mkdir(exist_ok=False);(out/'process_before.json').write_text(gpu.stdout);ledger=Path('outputs/gated/gpu_intervals.json').resolve();row,left,work=admit(ledger,out);argv=[a.python,str(BASE/'diagnostic.py'),'--work-seconds',str(work)];env=dict(os.environ)
    for k in ['PYTHONPATH','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_ENABLE_SDMA']:env.pop(k,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_TUNABLEOP_ENABLED='0');started=time.monotonic();base=inherited();child=proc=creation=None;actions=[];errors=[];timed=False
    m=dict(research_id='C015',research_sha256=rh,argv=argv,cwd=str(Path.cwd()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=bound,helper_sha256=sw['helper_sha256'],software_review_sha256=digest(BASE/'software/review.json'),preconsultation=pre,outer_interval_id=row['id'],remaining_at_admission=left,work_seconds=work)
    try:
        atomic_json(out/'manifest.json',m)
        with (out/'diagnostic.log').open('xb') as log:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);proc=psutil.Process(child.pid);creation=proc.create_time();m.update(pid=child.pid,creation_time=creation);atomic_json(out/'manifest.json',m)
            timed=base.wait_owned(child,proc,creation,argv,Path.cwd(),started+work,actions,remaining=lambda:base.remaining_seconds(json.loads(ledger.read_text()),time.time()),heartbeat=lambda seconds:atomic_json(out/'monitor_latest.json',dict(at=time.time(),pid=child.pid,remaining_work_seconds=seconds)))
    except BaseException as exc:errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:base.stop_owned(child,proc,creation,argv,Path.cwd(),actions)
            except BaseException as exc:errors.append(repr(exc))
        try:close(ledger,row['id'],child)
        except BaseException as exc:errors.append(repr(exc))
    raw=json.loads((science/'result.json').read_text()) if (science/'result.json').exists() else {};code=None if child is None else child.returncode
    status='awaiting_literature_review' if code==0 and not errors and raw.get('complete') else 'technical_blocked'
    if not errors and (timed or raw.get('status')=='budget_incomplete'):status='budget_incomplete'
    result=dict(research_id='C015',status=status,returncode=code,child_terminal=child is None or child.poll() is not None,timed_out=timed,actions=actions,errors=errors,elapsed_seconds=time.monotonic()-started,manifest_sha256=digest(out/'manifest.json'),child_artifact_sha256={str(p.relative_to(science)):digest(p) for p in science.glob('*') if p.is_file()},limitation='Operation only; candidate numerical failures remain failures and no successor is launched.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if status!='awaiting_literature_review':raise SystemExit(1)

if __name__=='__main__':main()
