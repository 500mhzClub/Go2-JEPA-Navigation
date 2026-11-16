"""C007 CPU fixtures; no scientific validation or shared-ledger mutation."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review
from supervision import admit, arguments, case_paths, check_parent, close_intervals, consultation, remaining_seconds, union_seconds, wait_owned, stop_owned

base = Path('outputs/gated/C007')
record, rh = require_record('C007')
out = base/'software'
out.mkdir(exist_ok=False)
started = time.monotonic()
checks = {}
ledger_path = Path('outputs/gated/gpu_intervals.json')
ledger_before = digest(ledger_path)
live_sources = {p: digest(p) for p in ['outputs/gated/C005/supervision.py', 'outputs/gated/C005/run_stage.py', 'outputs/gated/C005/v2/supervision.py', 'outputs/gated/C005/v2/run_stage.py']}


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError('Expected refusal')


try:
    for name in ['supervision.py', 'run_check.py', 'check_supervision.py']:
        ast.parse((base/name).read_text())
    verified = json.loads(Path('outputs/gated/C004/verification/manifest.json').read_text())['source_sha256']
    dependencies = ['go2nav/eval/slots_checkpoint.py', 'go2nav/eval/slots_recovery.py',
                    'go2nav/eval/balls_inference.py', 'go2nav/eval/balls_planning.py',
                    'go2nav/training/balls_slots.py', 'go2nav/training/balls.py', 'go2nav/research.py']
    dependencies += [f'go2nav/models/{name}.py' for name in ['balls_slots', 'slots', 'lewm', 'norm']]
    assert all(digest(f) == verified[f] for f in dependencies)
    assert digest('outputs/gated/A016/supervision.py') == '7cfa40983615bcf263b847c900822c48bc420f2c1d975f4d80ce8034086a7939'
    checks['actual_C004_dependencies_and_inherited_A016_unchanged'] = True

    valid = [f'{kind}_n3_s{seed}' for seed in [0, 1, 2] for kind in ['slots', 'o0', 'o1']]
    valid += [f'{kind}_n{n}_s0' for n in [1, 6] for kind in ['slots', 'o1']]
    for name in valid:
        p = case_paths(Path('/fixture'), name)
        argv = arguments('/fixture/python', p, '/fixture')
        expected = ['/fixture/python', '-m', 'go2nav.eval.slots_checkpoint', '--data', '/fixture/outputs/data/balls',
                    '--research', 'C004', '--checkpoint', str(p['checkpoint']), '--parent-review', str(p['parent']), '--out', str(p['science'])]
        if p['kind'] != 'slots':
            expected += ['--encoder-review', str(p['encoder_review'])]
        assert argv == expected
    for name in ['../slots_n3_s0', 'slots_n2_s0', 'slots_n1_s1', 'o0_n6_s0', 'o1_n6_s2', 'l0_n3_s0']:
        rejects(lambda: case_paths('/fixture', name))
    checks['thirteen_registered_identities_and_exact_unchanged_C004_argv'] = True
    p = case_paths('/fixture', 'o1_n3_s0')
    parent = dict(research_id='C001', kind='o1', n=3, seed=0)
    check_parent(parent, p)
    for key, value in [('research_id', 'C004'), ('kind', 'o0'), ('n', 6), ('seed', 1)]:
        wrong = dict(parent); wrong[key] = value
        rejects(lambda: check_parent(wrong, p))
    checks['parent_research_arm_count_seed_mismatches_rejected'] = True

    rows = [dict(research_id=rid, start=start, end=end) for rid, start, end in
            [('A003', 0, 10), ('C003', 20, 30), ('C007', 40, 60), ('C004', 45, 55), ('C001', 70, 100)]]
    remaining = remaining_seconds(rows, 100)
    assert remaining == dict(global_seconds=48*3600-70, reserve_seconds=8*3600-30)
    spec = importlib.util.spec_from_file_location('c007_check_existing_bc', 'outputs/gated/C005/v2/supervision.py')
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    assert bc.remaining_seconds(rows, 100) == dict(global_seconds=48*3600-70, bc_seconds=28*3600-30)
    checks['nested_union_reserve_and_BC_separation'] = True

    with tempfile.TemporaryDirectory(prefix='c007_checks_') as tmp:
        root = Path(tmp)
        ledger = root/'ledger.json'
        atomic_json(ledger, [])
        outer, _, work = admit(ledger, root/'op')
        assert work == 1755 and outer['research_id'] == 'C007'
        before = digest(ledger)
        rejects(lambda: admit(ledger, root/'another'))
        assert digest(ledger) == before
        checks['half_hour_including_cleanup_and_open_interval_refusal'] = True
        for rid, seconds in [('C003', 8*3600), ('A003', 48*3600)]:
            now = time.time()
            atomic_json(ledger, [dict(id='old', research_id=rid, out='fixture', start=now-seconds, end=now)])
            before = digest(ledger)
            rejects(lambda: admit(ledger, root/'exhausted'))
            assert digest(ledger) == before
        atomic_json(ledger, [])
        before = digest(ledger)
        rejects(lambda: admit(ledger, root/'no_cleanup_space', local_seconds=40))
        rejects(lambda: admit(ledger, root/'too_long', local_seconds=1801))
        assert digest(ledger) == before
        checks['reserve_global_local_exhaustion_no_mutation'] = True
        for rid, total in [('C003', 8*3600), ('A003', 48*3600)]:
            now = time.time()
            atomic_json(ledger, [dict(id='old', research_id=rid, out='fixture', start=now-(total-60), end=now)])
            _, _, work = admit(ledger, root/'small_remaining')
            assert abs(work-15) < 1e-6
        checks['smaller_shared_or_reserve_remainder_caps_admission'] = True

        real = root/'real'; real.mkdir()
        (root/'alias').symlink_to(real, target_is_directory=True)
        child_out = real/'check'; child_out.mkdir()
        now = time.time()
        fixture = [dict(id='outer', research_id='C007', out='op', start=now-10, end=None),
                   dict(id='inner', research_id='C004', out='alias/check', start=now-9, end=None),
                   dict(id='other', research_id='C001', out='unrelated', start=now-11, end=None)]
        atomic_json(child_out/'manifest.json', dict(run_id='inner'))
        atomic_json(ledger, fixture)
        class Terminal:
            def poll(self): return 0
        class Live:
            def poll(self): return None
        before = digest(ledger)
        rejects(lambda: close_intervals(ledger, 'outer', Live(), child_out, root), RuntimeError)
        assert digest(ledger) == before
        changes = close_intervals(ledger, 'outer', Terminal(), child_out, root)
        result = json.loads(ledger.read_text())
        assert set(changes) == {'outer', 'inner'} and result[2] == fixture[2]
        assert result[0]['end'] == result[1]['end']
        assert abs(union_seconds(result[:2], result[0]['end'])-(result[0]['end']-fixture[0]['start'])) < 1e-6
        checks['canonical_owned_only_terminal_closure_and_union_once'] = True
        invalid = []
        wrong = copy.deepcopy(fixture); wrong[1]['start'] = wrong[0]['start']-1; invalid.append(wrong)
        wrong = copy.deepcopy(fixture); wrong.append(dict(wrong[1], id='duplicate')); invalid.append(wrong)
        wrong = copy.deepcopy(fixture); wrong[1]['id'] = 'not_manifest'; invalid.append(wrong)
        wrong = copy.deepcopy(fixture); wrong[0]['research_id'] = 'C005'; invalid.append(wrong)
        for rows in invalid:
            atomic_json(ledger, rows); before = digest(ledger)
            rejects(lambda: close_intervals(ledger, 'outer', Terminal(), child_out, root))
            assert digest(ledger) == before
        checks['historical_ambiguous_wrong_manifest_and_outer_refused'] = True

        paths = case_paths(root, 'slots_n3_s0')
        paths['checkpoint'].parent.mkdir(parents=True)
        paths['checkpoint'].write_bytes(b'synthetic checkpoint')
        paths['parent'].write_text('synthetic parent')
        files = {f'outputs/data/balls/n3/{split}/{name}' for split in ['train', 'val'] for name in ['frames.npy', 'labels.npz', 'meta.json']}
        for name in files:
            f = root/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b'synthetic data')
        value = dict(name='slots_n3_s0', c004_record_sha256='science', c007_record_sha256='operations',
                     checkpoint_sha256=digest(paths['checkpoint']), parent_review_sha256=digest(paths['parent']),
                     consulted_at='fixture', consequences='synthetic guard check, not scientific approval',
                     sources=[dict(url='fixture', version='fixture', sections='fixture', consequence='fixture')],
                     data_sha256={f: digest(root/f) for f in files})
        pre = root/'pre.json'; atomic_json(pre, value)
        consultation(pre, paths, root, 'science', 'operations')
        for field in ['name', 'c004_record_sha256', 'c007_record_sha256', 'checkpoint_sha256', 'parent_review_sha256', 'sources']:
            wrong = copy.deepcopy(value); wrong[field] = None; atomic_json(pre, wrong)
            rejects(lambda: consultation(pre, paths, root, 'science', 'operations'))
        wrong = copy.deepcopy(value); wrong['data_sha256'].pop(next(iter(files))); atomic_json(pre, wrong)
        rejects(lambda: consultation(pre, paths, root, 'science', 'operations'))
        atomic_json(pre, value)
        (root/next(iter(files))).write_bytes(b'tampered data')
        rejects(lambda: consultation(pre, paths, root, 'science', 'operations'))
        checks['consultation_parent_record_source_and_all_data_bindings'] = True

        for mode, duration, deadline in [('normal', .02, 2), ('deadline', 10, .10)]:
            argv = [sys.executable, '-c', f'import time; time.sleep({duration})']
            signals = []
            with (root/f'{mode}.log').open('wb') as log:
                child = subprocess.Popen(argv, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                process = psutil.Process(child.pid); identity = process.create_time()
                try:
                    timed = wait_owned(child, process, identity, argv, Path.cwd(), time.monotonic()+deadline, signals,
                        poll_seconds=.02, reserve=0, term_seconds=.5, kill_seconds=.5)
                    assert child.poll() is not None
                    if mode == 'normal':
                        assert child.returncode == 0 and not timed and not signals
                    else:
                        assert timed and child.returncode == -15 and [s['signal'] for s in signals] == ['SIGTERM']
                    checks[f'owned_CPU_{mode}'] = dict(returncode=child.returncode, timed_out=timed, signals=signals)
                finally:
                    if child.poll() is None:
                        stop_owned(child, process, identity, argv, Path.cwd(), signals, .5, .5)

    assert 'torch' not in sys.modules
    assert digest(ledger_path) == ledger_before
    assert all(digest(p) == h for p, h in live_sources.items())
    checks['no_torch_shared_ledger_or_existing_supervisor_mutation'] = True
    result = dict(research_id='C007', research_sha256=rh, c004_record_sha256=digest('docs/research/records/C004.json'),
                  status='awaiting_literature_review', checks=checks,
                  elapsed_seconds=time.monotonic()-started, scientific_GPU_cases=0, torch_imported=False,
                  shared_ledger_unchanged=True, existing_supervisor_sources_unchanged=True,
                  source_sha256={name: digest(base/name) for name in ['supervision.py', 'run_check.py', 'check_supervision.py']},
                  repository_source_sha256={str(p): digest(p) for p in sorted(Path('go2nav').rglob('*.py'))},
                  a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'),
                  c004_verified_dependencies={f: digest(f) for f in dependencies},
                  inherited_A016_review_sha256=digest('outputs/gated/A016/software/review.json'),
                  limitation='CPU fixtures and source checks only. No selected checkpoint, actual GPU timeout or final numerical result executed. Inherited A016 process handling is unchanged.')
    atomic_json(out/'result.json', result)
    print(json.dumps(dict(checks=len(checks), observed=checks, elapsed_seconds=result['elapsed_seconds'])))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='C007', research_sha256=rh, status='technical_blocked', checks=checks, error=repr(exc)))
    raise
