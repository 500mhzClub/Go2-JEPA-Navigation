"""A016 CPU-only fault checks; temporary ledgers and owned harmless child processes."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
import psutil
from go2nav.research import atomic_json, digest, require_record
from supervision import (admit, close_intervals, identity_matches, remaining_seconds,
                         stop_owned, union_seconds, wait_owned)

require_record('A016')
base = Path('outputs/gated/A016')
out = base/'software'
out.mkdir(exist_ok=False)
shared = Path('outputs/gated/gpu_intervals.json')
before = digest(shared)
checks = {}
started = time.time()


def rejected(name, fn, exception=ValueError):
    try:
        fn()
    except exception as exc:
        checks[name] = dict(rejected=True, error=str(exc))
    else:
        raise AssertionError('Fault accepted: '+name)


try:
    for name in ['supervision.py','run_a003.py','check_supervision.py']:
        ast.parse((base/name).read_text())
    rows = [dict(start=10,end=20,research_id='A003'), dict(start=12,end=19,research_id='A016'),
            dict(start=30,end=35,research_id='C001')]
    assert union_seconds(rows, 40) == 15
    assert remaining_seconds(rows, 40) == dict(global_seconds=48*3600-15,a_seconds=4*3600-10)
    checks['overlapping_intervals_count_once_and_allocations_differ'] = True
    with tempfile.TemporaryDirectory(prefix='a016_test_') as tmp:
        root = Path(tmp);ledger = root/'ledger.json';child_out = root/'child_out';child_out.mkdir()
        atomic_json(ledger, [dict(id='foreign',research_id='C001',start=time.time(),end=None,out='elsewhere')])
        h=digest(ledger)
        rejected('foreign_open_interval_refused', lambda:admit(ledger,root/'out'))
        assert digest(ledger)==h
        for label, hours, research_id in [('A',4,'A001'),('global',48,'C001')]:
            atomic_json(ledger,[dict(id='used',research_id=research_id,start=0,end=hours*3600,out='old')])
            h=digest(ledger)
            rejected(label+'_exhaustion_refused', lambda:admit(ledger,root/'out'))
            assert digest(ledger)==h
        atomic_json(ledger,[])
        outer,remaining,work=admit(ledger,root/'out')
        assert work==4*3600-45
        checks['cleanup_reserve_deducted'] = True
        old=dict(id='closed',research_id='C001',start=1,end=2,out='elsewhere')
        foreign=dict(id='unrelated',research_id='C001',start=outer['start'],end=None,out='elsewhere')
        inner=dict(id='inner',research_id='A003',start=outer['start']+.00001,end=None,out=str(child_out))
        atomic_json(ledger,[old,outer,inner,foreign])
        atomic_json(child_out/'manifest.json',dict(run_id='inner'))
        class Live:
            def poll(self): return None
        h=digest(ledger)
        rejected('live_child_closure_refused',lambda:close_intervals(ledger,outer['id'],Live(),child_out,root),RuntimeError)
        assert digest(ledger)==h
        atomic_json(child_out/'manifest.json',dict(run_id='wrong'))
        rejected('wrong_inner_identity_refused',lambda:close_intervals(ledger,outer['id'],None,child_out,root))
        assert digest(ledger)==h
        atomic_json(child_out/'manifest.json',dict(run_id='inner'))
        changed=close_intervals(ledger,outer['id'],None,child_out,root)
        updated=json.loads(ledger.read_text())
        assert set(changed)=={outer['id'],'inner'} and updated[0]==old and updated[-1]==foreign
        assert union_seconds(updated[1:3],time.time())==updated[1]['end']-outer['start']
        checks['only_owned_outer_and_inner_closed_without_double_counting'] = True
        # Actual subprocesses exercise wait/exit/signal semantics, without importing torch.
        cases=[('normal_success','import time; time.sleep(.05)',1.,0,[]),
               ('nonzero_exit','import time; time.sleep(.05); raise SystemExit(7)',1.,7,[]),
               ('deadline_sigterm','import time; time.sleep(10)',.2,-15,['SIGTERM']),
               ('deadline_sigkill','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)',.4,-9,['SIGTERM','SIGKILL'])]
        for name,code,work,expected_code,expected_signals in cases:
            argv=[sys.executable,'-c',code];actions=[]
            with (out/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                process=psutil.Process(child.pid);identity=process.create_time()
                try:
                    timed_out=wait_owned(child,process,identity,argv,root,time.monotonic()+work,actions,
                                         poll_seconds=.05,term_seconds=.15,kill_seconds=1.)
                    assert child.returncode==expected_code and [a['signal'] for a in actions]==expected_signals
                    assert timed_out==bool(expected_signals)
                    checks[name]=dict(returncode=child.returncode,signals=expected_signals,timed_out=timed_out)
                finally:
                    if child.poll() is None:
                        stop_owned(child,process,identity,argv,root,actions,term_seconds=.1,kill_seconds=1.)
        argv=[sys.executable,'-c','import time; time.sleep(10)'];actions=[]
        with (out/'identity.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process=psutil.Process(child.pid);identity=process.create_time()
            try:
                assert identity_matches(process,identity,argv,root)
                rejected('identity_mismatch_never_signals',lambda:stop_owned(child,process,identity+1,argv,root,actions,.1,1),RuntimeError)
                assert not actions and child.poll() is None
                timed_out=wait_owned(child,process,identity,argv,root,time.monotonic()+10,actions,
                                    remaining=lambda:dict(global_seconds=44,a_seconds=100),term_seconds=.1,kill_seconds=1.)
                assert timed_out and child.poll() is not None
                checks['new_global_budget_exhaustion_stops_before_fixed_deadline']=True
            finally:
                if child.poll() is None:
                    stop_owned(child,process,identity,argv,root,actions,.1,1)
    refusal=subprocess.run([sys.executable,str(base/'run_a003.py'),'--run','l0_n1_s0'],capture_output=True,text=True,timeout=10)
    (out/'live_trainer_refusal.txt').write_text(refusal.stdout+refusal.stderr)
    assert refusal.returncode!=0 and 'Dedicated training slot occupied' in refusal.stderr
    assert not (base/'runs').exists() and digest(shared)==before
    checks['real_live_trainer_refused_before_output_and_shared_ledger_mutation']=True
    assert 'torch' not in sys.modules
    atomic_json(out/'result.json',dict(research_id='A016',status='passed',checks=checks,
        elapsed_seconds=time.time()-started,source_sha256={n:digest(base/n) for n in ['supervision.py','run_a003.py','check_supervision.py']},
        shared_ledger_unchanged=True,torch_imported=False,actual_gpu_cases=0,
        limitation='CPU supervision tests only; successful actual all-six final admission and A003 numerical results remain pending.'))
    print(json.dumps(dict(status='passed',checks=len(checks),elapsed_seconds=time.time()-started)))
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='A016',status='technical_blocked',checks=checks,error=repr(exc),
        shared_ledger_unchanged=digest(shared)==before,
        source_sha256={n:digest(base/n) for n in ['supervision.py','run_a003.py','check_supervision.py']}))
    raise
