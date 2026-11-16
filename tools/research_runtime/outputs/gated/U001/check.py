"""Focused checks of the time-limit removal, original routes and stall handling."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import psutil
import torch
from common import BASE, HELPERS, accounting_remaining, accounting_run, admit, arguments, inherited, inputs, research, route, wait_progress
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

def rejects(fn):
    try: fn()
    except (AssertionError, ValueError, FileNotFoundError): return
    raise AssertionError('Expected rejection')

start = time.monotonic(); out = BASE/'software'; out.mkdir(exist_ok=False)
for name in HELPERS:
    (out/'source').mkdir(exist_ok=True); shutil.copy2(BASE/name, out/'source'/name)
checks = []; ledger = Path('outputs/gated/gpu_intervals.json'); original_ledger = digest(ledger)
original_partial = digest('outputs/gated/planning/o0_n3_s0/cases.jsonl')
try:
    torch.set_num_threads(2)
    assert all(os.environ.get(k) == '' for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
    for name in HELPERS: ast.parse((BASE/name).read_text())
    stages, sup, owned = inherited(); _, rh = require_record('U001')
    checks.append('frozen_scientific_and_original_supervisor_sources_verified')
    with tempfile.TemporaryDirectory(prefix='u001_') as tmp:
        root = Path(tmp); lp = root/'ledger.json'; now = time.time()
        past = dict(id='past', research_id='B001', out='historical', start=now-200000, end=now-1)
        atomic_json(lp, [past]); outer = admit(lp, root/'operations', 'B001')
        assert outer['budget_policy'] == 'U001'
        before = digest(lp); rejects(lambda: admit(lp, root/'another', 'B001')); assert before == digest(lp)
        sup.close_intervals(lp, outer['id'], None, root/'unused_science', Path.cwd(), 'B001')
        assert json.loads(lp.read_text())[0] == past
        checks.append('admission_beyond_old_global_and_stage_caps_open_interval_refusal_and_owned_closure')
        for fail in [False, True]:
            dest = root/('failure' if fail else 'normal')
            with patch.object(research, 'provenance', return_value={'research_id':'B001'}), patch.object(research, 'dedicated_device'), patch.object(torch.cuda, 'get_device_name', return_value='software fixture'):
                try:
                    with accounting_run(dest, 'B001', {}, ledger=lp, limit=0) as m:
                        assert m['budget_hours'] is None and accounting_remaining(m) == float('inf')
                        bad = dict(m, authorization_sha256='wrong'); rejects(lambda: accounting_remaining(bad))
                        if fail: raise ValueError('preserved injected exception')
                except ValueError:
                    assert fail
            m = json.loads((dest/'manifest.json').read_text())
            assert m['status'] == ('technical_blocked' if fail else 'awaiting_literature_review')
            assert all(r['end'] is not None for r in json.loads(lp.read_text()))
        checks.append('accounting_context_normal_exception_closure_no_budget_and_authorization_refusal')
        for mode in ['normal', 'progress', 'stalled']:
            progress = root/(mode+'.progress')
            if mode == 'progress':
                script = 'import pathlib,time; p=pathlib.Path('+repr(str(progress))+');\nfor i in range(12):\n p.write_text(str(i)); time.sleep(.05)\n'
            else: script = 'import time; time.sleep('+('.02' if mode == 'normal' else '5')+')'
            argv = [sys.executable, '-c', script]; actions = []
            child = subprocess.Popen(argv, cwd=Path.cwd(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            proc = psutil.Process(child.pid); created = proc.create_time()
            try:
                stalled = wait_progress(child, proc, created, argv, Path.cwd(), [progress], actions, stall_seconds=.25, poll_seconds=.02, term_seconds=.5, kill_seconds=.5)
                assert (child.returncode, stalled) == ((-15, True) if mode == 'stalled' else (0, False))
                if mode != 'stalled': assert not actions
            finally:
                if child.poll() is None: owned.stop_owned(child, proc, created, argv, Path.cwd(), actions, .5, .5)
        checks.append('normal_completion_progress_beyond_stall_duration_and_exact_owned_stall_cleanup')
    for name in ['o0_n3_s0', 'o1_n3_s0']:
        p = route(name); original = stages.arguments('/fixture/python', p); actual = arguments('/fixture/python', p)
        assert actual[2:] == original[2:] and actual[:2] == ['/fixture/python', str(BASE/'science.py')]
        assert p['science'] == Path.cwd()/'outputs/gated/planning_full'/name
    rejects(lambda: route('l0_n3_s0'))
    checks.append('both_original_scientific_CLI_arguments_unchanged_fresh_output_routes')
    p = route('o0_n3_s0'); bound = inputs(p)
    expected = {f.resolve() for f in stages.inputs(p)}
    probe = Path('outputs/gated/probes/o0_n3_s0/review.json'); pr = require_review(probe)
    expected.update(f.resolve() for f in [probe, probe.parent/pr['result'], Path('docs/research/records/U001.json')])
    assert set(bound) == expected
    rejects(lambda: inputs(route('o1_n3_s0')))
    checks.append('complete_real_O0_parent_binding_and_unreviewed_O1_predecessor_refusal')
    assert not torch.cuda.is_initialized() and digest(ledger) == original_ledger
    assert digest('outputs/gated/planning/o0_n3_s0/cases.jsonl') == original_partial
    checks.append('no_CUDA_shared_ledger_or_preserved_partial_mutation')
    result = dict(research_id='U001', research_sha256=rh, status='awaiting_literature_review', checks=checks,
        elapsed_seconds=time.monotonic()-start, source_sha256={f:digest(BASE/f) for f in HELPERS},
        repository_source_sha256={str(p):digest(p) for p in Path('go2nav').rglob('*.py')}, inputs_count=len(bound),
        limitation='Operational CPU fixtures and real parent admission only; no new scientific experiment, GPU computation or result interpretation.')
    atomic_json(out/'result.json', result); print(json.dumps({k:v for k,v in result.items() if k != 'repository_source_sha256'}), flush=True)
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='U001', status='technical_blocked', checks=checks, error=repr(exc))); raise
