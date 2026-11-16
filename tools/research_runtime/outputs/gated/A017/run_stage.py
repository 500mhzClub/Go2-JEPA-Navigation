"""Bound one CPU analysis stage; never signal the GPU trainer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

phase=sys.argv[1];assert phase in ['calibration','probes']
base=Path('outputs/gated/A017');record,rh=require_record('A017')
assert require_review(base/'software/review.json')['status']=='passed'
software=json.loads((base/'software/result.json').read_text());verify_files(base,software['source_sha256'])
if phase=='probes':assert require_review(base/'calibration/review.json')['status']=='passed'
out=base/'supervision'/phase;out.mkdir(parents=True,exist_ok=False)
plan=json.loads(Path('outputs/gated/A014/continuation_plan.json').read_text())
argv=[plan['recovery_commands'][0]['argv'][0],str(base/'analysis.py'),phase]
env=dict(os.environ);env.pop('PYTHONPATH',None)
env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
shared=Path('outputs/gated/gpu_intervals.json');before=digest(shared)
started=time.monotonic();stop=None;actions=[];peak=0
with (out/'analysis.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    p=psutil.Process(child.pid);identity=p.create_time()
    execution=dict(research_id='A017',phase=phase,pid=child.pid,creation_time=identity,argv=argv,
                   started_at=time.time(),research_sha256=rh,software_review_sha256=digest(base/'software/review.json'))
    atomic_json(out/'execution.json',execution)
    while child.poll() is None:
        try:peak=max(peak,p.memory_info().rss)
        except psutil.NoSuchProcess:pass
        elapsed=time.monotonic()-started
        if elapsed>=885:stop='CPU900second_stage_cap_with15second_cleanup_reserve'
        if peak>16*2**30:stop='CPU16GB_RSS_cap'
        atomic_json(out/'monitor_latest.json',dict(pid=child.pid,at=time.time(),elapsed_seconds=elapsed,peak_rss_gb=peak/2**30,stop=stop))
        if stop:
            for label,method,delay in [('SIGTERM',p.terminate,10),('SIGKILL',p.kill,5)]:
                if child.poll() is not None:break
                assert p.is_running() and p.create_time()==identity and p.cmdline()==argv and p.cwd()==str(Path.cwd())
                method();actions.append(label)
                try:child.wait(timeout=delay)
                except subprocess.TimeoutExpired:pass
            break
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
terminal=child.poll() is not None
result=dict(research_id='A017',phase=phase,status='awaiting_literature_review' if child.returncode==0 else 'technical_blocked',
            returncode=child.returncode,child_terminal=terminal,elapsed_seconds=time.monotonic()-started,peak_rss_gb=peak/2**30,
            stop_reason=stop,signals=actions,gpu_ledger_unchanged=digest(shared)==before,execution_sha256=digest(out/'execution.json'))
if stop:result['status']='budget_incomplete' if terminal else 'technical_blocked'
atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
if child.returncode!=0:raise SystemExit(1)
