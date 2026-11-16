"""Serialized bounded CPU probe execution; no automatic review or next case."""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_review,verify_files
from parents import context


@contextmanager
def cpu_lock(root):
    with (Path(root)/'outputs/gated/B003/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def owned_module():
    path=Path('outputs/gated/A016/supervision.py')
    assert require_review(path.parent/'software/review.json')['status']=='passed'
    assert digest(path)=='7cfa40983615bcf263b847c900822c48bc420f2c1d975f4d80ce8034086a7939'
    spec=importlib.util.spec_from_file_location('c011_owned_child',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def supervise(argv,root,out,env,execution,work=885,term=10,kill=5,poll=2):
    owned=owned_module();started=time.monotonic();child=process=identity=None
    actions=[];errors=[];peak=0;reason=None
    try:
        with (out/'analysis.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process=psutil.Process(child.pid);identity=process.create_time()
            execution.update(pid=child.pid,creation_time=identity,argv=argv,cwd=str(root),started_at=time.time())
            atomic_json(out/'execution.json',execution)
            while child.poll() is None:
                try:peak=max(peak,process.memory_info().rss)
                except psutil.NoSuchProcess:pass
                elapsed=time.monotonic()-started
                if elapsed>=work:reason='CPU900second_cap_including15second_cleanup'
                if peak>16*2**30:reason='CPU16GiB_RSS_cap'
                atomic_json(out/'monitor_latest.json',dict(at=time.time(),pid=child.pid,creation_time=identity,
                    elapsed_seconds=elapsed,peak_rss_gb=peak/2**30,stop_reason=reason))
                if reason:break
                try:child.wait(timeout=min(poll,max(.001,work-elapsed)))
                except subprocess.TimeoutExpired:pass
    except BaseException as exc:errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or identity is None:raise RuntimeError('Owned child identity unavailable')
                owned.stop_owned(child,process,identity,argv,root,actions,term,kill)
            except BaseException as exc:errors.append(repr(exc))
    terminal=child is not None and child.poll() is not None;code=None if child is None else child.returncode
    status='awaiting_literature_review' if terminal and code==0 and not errors else 'technical_blocked'
    if reason and terminal and not errors:status='budget_incomplete'
    return dict(status=status,returncode=code,child_terminal=terminal,elapsed_seconds=time.monotonic()-started,
                peak_rss_gb=peak/2**30,stop_reason=reason,signals=actions,errors=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--python',required=True);args=parser.parse_args()
    root=Path.cwd();base=root/'outputs/gated/C011'
    with cpu_lock(root):
        ctx=context(root,args.run)
        assert not subprocess.check_output(['git','status','--porcelain'],text=True)
        science=root/'outputs/gated/probes'/args.run;out=base/'supervision'/args.run
        assert not science.exists() and not out.exists();out.mkdir(parents=True,exist_ok=False)
        argv=[args.python,str(base/'analysis.py'),'--run',args.run]
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        ledger=root/'outputs/gated/gpu_intervals.json';before=digest(ledger)
        execution=dict(research_id='C011',name=args.run,research_sha256=ctx['research_sha256'],
            software_review_sha256=digest(base/'software/review.json'),preconsultation_sha256=digest(ctx['pre']),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        result=supervise(argv,root,out,env,execution)
        artifacts={};unchanged=False;raw_status=None
        if result['child_terminal']:
            try:
                if (science/'result.json').exists():raw_status=json.loads((science/'result.json').read_text())['status']
                artifacts={str(f.relative_to(science)):digest(f) for f in science.rglob('*') if f.is_file()}
                verify_files(root,ctx['consult']['inputs_sha256']);unchanged=True
            except BaseException as exc:result['errors'].append(repr(exc))
        if result['status']=='awaiting_literature_review' and (result['errors'] or raw_status!='awaiting_literature_review'):
            result['status']='technical_blocked'
        result.update(research_id='C011',name=args.run,inputs_unchanged=unchanged,child_artifact_sha256=artifacts,raw_status=raw_status,
            execution_sha256=digest(out/'execution.json') if (out/'execution.json').exists() else None,
            gpu_ledger_unchanged=digest(ledger)==before,no_C011_gpu_intervals=not any(r['research_id']=='C011' for r in json.loads(ledger.read_text())),
            note='Independent GPU completion may change the shared ledger; this CPU case owns no interval. No review/successor is automatic.')
        atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
        if result['status']!='awaiting_literature_review':raise SystemExit(1)


if __name__=='__main__':main()
