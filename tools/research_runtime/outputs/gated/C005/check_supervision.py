"""C005 CPU-only fault checks with temporary ledgers and owned harmless children."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
import psutil
from go2nav.research import atomic_json, digest, require_record, verify_files
from supervision import admit, close_intervals, consultation, remaining_seconds, stop_owned, union_seconds, wait_owned

require_record('C005')
base = Path('outputs/gated/C005')
out = base/'software'
out.mkdir(exist_ok=False)
shared = Path('outputs/gated/gpu_intervals.json')
before = digest(shared)
files = ['supervision.py', 'run_stage.py', 'check_supervision.py']
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
    for name in files:
        ast.parse((base/name).read_text())
    rows = [dict(start=10, end=20, research_id='C005'), dict(start=12, end=19, research_id='C001'),
            dict(start=30, end=35, research_id='D001'), dict(start=40, end=42, research_id='C003')]
    assert union_seconds(rows, 50) == 17
    assert remaining_seconds(rows, 50) == dict(global_seconds=48*3600-17, bc_seconds=28*3600-10)
    checks['overlap_and_BC_D_reserve_separation'] = True
    with tempfile.TemporaryDirectory(prefix='c005_test_') as tmp:
        root = Path(tmp);ledger = root/'ledger.json';child_out = root/'child';child_out.mkdir()
        alias = root/'alias';alias.symlink_to(child_out, target_is_directory=True)
        atomic_json(ledger, [dict(id='foreign', research_id='D001', start=time.time(), end=None, out='foreign')])
        h = digest(ledger)
        rejected('foreign_open_refused', lambda:admit(ledger, root/'out'))
        assert digest(ledger) == h
        for label, hours, rid in [('BC', 28, 'C001'), ('global', 48, 'A007')]:
            atomic_json(ledger, [dict(id='old', research_id=rid, start=0, end=hours*3600, out='old')])
            h = digest(ledger)
            rejected(label+'_exhaustion_refused', lambda:admit(ledger, root/'out'))
            assert digest(ledger) == h
        atomic_json(ledger, [])
        outer, remaining, work = admit(ledger, root/'out')
        assert work == 28*3600-45
        checks['cleanup_reserve_and_initial_cap'] = True
        old = dict(id='old', research_id='A003', start=1, end=2, out='old')
        foreign = dict(id='foreign', research_id='D001', start=outer['start'], end=None, out='foreign')
        inner = dict(id='inner', research_id='C001', start=outer['start']+.00001, end=None, out=str(alias))
        atomic_json(ledger, [old, outer, inner, foreign])
        atomic_json(child_out/'manifest.json', dict(run_id='inner'))
        class Live:
            def poll(self): return None
        h = digest(ledger)
        rejected('live_child_closure_refused', lambda:close_intervals(ledger, outer['id'], Live(), child_out, root), RuntimeError)
        assert digest(ledger) == h
        atomic_json(child_out/'manifest.json', dict(run_id='wrong'))
        rejected('wrong_inner_identity_refused', lambda:close_intervals(ledger, outer['id'], None, child_out, root))
        assert digest(ledger) == h
        atomic_json(child_out/'manifest.json', dict(run_id='inner'))
        atomic_json(ledger, [old, outer, inner, dict(inner, id='duplicate'), foreign])
        hdup = digest(ledger)
        rejected('ambiguous_child_refused', lambda:close_intervals(ledger, outer['id'], None, child_out, root))
        assert digest(ledger) == hdup
        atomic_json(ledger, [old, outer, inner, foreign])
        changed = close_intervals(ledger, outer['id'], None, child_out, root)
        updated = json.loads(ledger.read_text())
        assert set(changed) == {outer['id'], 'inner'} and updated[0] == old and updated[-1] == foreign
        assert union_seconds(updated[1:3], time.time()) == updated[1]['end']-outer['start']
        checks['canonical_symlink_child_only_closed_union_once'] = True
        pre = root/'consult.json'
        atomic_json(pre, dict(name='wrong', c001_record_sha256='hash'))
        rejected('wrong_experiment_consultation_refused', lambda:consultation(pre, 'slots_n3_s0', 'hash'))
        atomic_json(pre, dict(name='slots_n3_s0', c001_record_sha256='hash'))
        rejected('missing_substantive_consultation_refused', lambda:consultation(pre, 'slots_n3_s0', 'hash'))
        source = root/'code.py';source.write_text('first\n');hashes={'code.py':digest(source)};source.write_text('changed\n')
        rejected('changed_source_refused', lambda:verify_files(root, hashes))
        cases = [('normal', 'import time; time.sleep(.05)', 1., 0, []),
                 ('nonzero', 'import time; time.sleep(.05); raise SystemExit(7)', 1., 7, []),
                 ('deadline_TERM', 'import time; time.sleep(10)', .2, -15, ['SIGTERM']),
                 ('deadline_KILL', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)', .4, -9, ['SIGTERM','SIGKILL'])]
        for name, code, work, expected_code, signals in cases:
            argv = [sys.executable, '-c', code];actions=[]
            with (out/(name+'.log')).open('xb') as log:
                child = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                process = psutil.Process(child.pid);identity = process.create_time()
                try:
                    timeout = wait_owned(child, process, identity, argv, root, time.monotonic()+work, actions,
                                         poll_seconds=.05, term_seconds=.15, kill_seconds=1.)
                    assert child.returncode == expected_code and [x['signal'] for x in actions] == signals
                    assert timeout == bool(signals)
                    checks[name] = dict(returncode=child.returncode, signals=signals, timed_out=timeout)
                finally:
                    if child.poll() is None:
                        stop_owned(child, process, identity, argv, root, actions, .1, 1.)
        argv = [sys.executable, '-c', 'import time; time.sleep(10)'];actions=[]
        child = subprocess.Popen(argv, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process = psutil.Process(child.pid);identity = process.create_time()
        try:
            rejected('mismatched_identity_never_signals', lambda:stop_owned(child, process, identity+1, argv, root, actions, .1, 1.), RuntimeError)
            assert child.poll() is None and not actions
            timeout = wait_owned(child, process, identity, argv, root, time.monotonic()+10, actions,
                                 remaining=lambda:dict(global_seconds=100, bc_seconds=44), term_seconds=.1, kill_seconds=1.)
            assert timeout and child.poll() is not None
            checks['BC_exhaustion_preempts_fixed_deadline'] = True
        finally:
            if child.poll() is None:
                stop_owned(child, process, identity, argv, root, actions, .1, 1.)
    assert digest(shared) == before and 'torch' not in sys.modules
    atomic_json(out/'result.json', dict(research_id='C005', status='passed', checks=checks,
        elapsed_seconds=time.time()-started, source_sha256={n:digest(base/n) for n in files},
        reused_a016_supervision_sha256=digest(Path('outputs/gated/A016/supervision.py')),
        shared_ledger_unchanged=True, torch_imported=False, actual_gpu_cases=0,
        limitation='CPU operational checks; no full trainer, GPU convergence or scientific performance claim.'))
    print(json.dumps(dict(status='passed', checks=len(checks), elapsed_seconds=time.time()-started)))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='C005', status='technical_blocked', checks=checks, error=repr(exc),
        source_sha256={n:digest(base/n) for n in files}, shared_ledger_unchanged=digest(shared)==before))
    raise
