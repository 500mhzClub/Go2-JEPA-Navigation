"""C006 bounded CPU launch; only its owned child may be signalled."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);name=p.parse_args().run
    base=Path('outputs/gated/C006');record,rh=require_record('C006');order=record['protocol']['order'];assert name in order
    assert require_review(base/'software/review.json')['status']=='passed'
    software=json.loads((base/'software/result.json').read_text());assert software['research_sha256']==rh
    verify_files(base,software['source_sha256']);verify_files(Path.cwd(),software['repository_source_sha256'])
    for prev in order[:order.index(name)]:
        assert require_review(Path('outputs/gated/probes')/prev/'review.json')['status'] in ['passed','measured_negative']
        assert require_review(base/'supervision'/prev/'review.json')['status']=='passed'
    pre=json.loads((base/('pre_'+name+'.json')).read_text());assert pre['name']==name and pre['research_sha256']==rh and pre['consequences'] and pre['sources']
    path=Path('outputs/gated/A016/supervision.py');assert digest(path)==software['a016_supervision_sha256']
    spec=importlib.util.spec_from_file_location('owned_cpu_supervision',path);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    out=base/'supervision'/name;assert not (Path('outputs/gated/probes')/name).exists();out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path('outputs/gated/A014/continuation_plan.json').read_text())
    argv=[plan['recovery_commands'][0]['argv'][0],str(base/'analysis.py'),'--run',name]
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
    started=time.monotonic();stop=None;actions=[];peak=0;errors=[]
    with (out/'analysis.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        process=psutil.Process(child.pid);identity=process.create_time()
        execution=dict(research_id='C006',name=name,pid=child.pid,creation_time=identity,argv=argv,cwd=str(Path.cwd()),
            started_at=time.time(),research_sha256=rh,software_review_sha256=digest(base/'software/review.json'),preconsultation_sha256=digest(base/('pre_'+name+'.json')))
        atomic_json(out/'execution.json',execution)
        try:
            while child.poll() is None:
                try:peak=max(peak,process.memory_info().rss)
                except psutil.NoSuchProcess:pass
                elapsed=time.monotonic()-started
                if elapsed>=885:stop='CPU900second_cap_with15second_cleanup_reserve'
                if peak>16*2**30:stop='CPU16GiB_RSS_cap'
                atomic_json(out/'monitor_latest.json',dict(pid=child.pid,at=time.time(),elapsed_seconds=elapsed,peak_rss_gb=peak/2**30,stop=stop))
                if stop:break
                try:child.wait(timeout=min(2,max(.01,885-elapsed)))
                except subprocess.TimeoutExpired:pass
        except BaseException as exc:errors.append(repr(exc))
        finally:
            if child.poll() is None:
                try:owned.stop_owned(child,process,identity,argv,Path.cwd(),actions,10,5)
                except BaseException as exc:errors.append(repr(exc))
    terminal=child.poll() is not None
    status='awaiting_literature_review' if child.returncode==0 and not errors else 'technical_blocked'
    if stop and terminal and not errors:status='budget_incomplete'
    rows=json.loads(ledger.read_text());no_own_rows=not any(r['research_id']=='C006' for r in rows)
    result=dict(research_id='C006',name=name,status=status,returncode=child.returncode,child_terminal=terminal,
        elapsed_seconds=time.monotonic()-started,peak_rss_gb=peak/2**30,stop_reason=stop,signals=actions,errors=errors,
        gpu_ledger_unchanged=digest(ledger)==before,no_C006_gpu_intervals=no_own_rows,execution_sha256=digest(out/'execution.json'),
        note='GPU ledger may legitimately change if an independent GPU job finishes; CPU child has no GPU interval.')
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if child.returncode!=0 or errors:raise SystemExit(1)


if __name__=='__main__':main()
