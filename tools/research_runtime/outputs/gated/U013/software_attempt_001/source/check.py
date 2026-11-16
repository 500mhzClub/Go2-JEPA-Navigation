"""Exercise the B004 entry point with existing probes; no scientific experiment."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path.cwd()))
import integration
from integration import BASE, ORIGINAL, HELPERS, module
from go2nav.research import atomic_json, digest, require_record, verify_files


def refuses(call):
    with pytest.raises((AssertionError, ValueError, KeyError, FileNotFoundError, SystemExit)):
        call()


def functions(path, names, scope):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def rebind(folder, **changes):
    path = folder/'result.json'
    raw = json.loads(path.read_text())
    raw.update(changes)
    atomic_json(path, raw)
    rp = folder/'review.json'
    if rp.exists():
        review = json.loads(rp.read_text())
        review['result_sha256'] = digest(path)
        atomic_json(rp, review)
    return raw


def modify(path, **changes):
    raw = json.loads(path.read_text())
    raw.update(changes)
    atomic_json(path, raw)
    return raw


started = time.monotonic()
root = Path.cwd()
out = BASE/'software'
out.mkdir(exist_ok=False)
(out/'source').mkdir()
for name in HELPERS:
    shutil.copy2(BASE/name, out/'source'/name)
checks = []
ledger = root/'outputs/gated/gpu_intervals.json'
ledger_before = digest(ledger)
frozen = {str(p): digest(p) for label in ('B004', 'C011', 'U001', 'U007')
          for p in (root/'outputs/gated'/label).glob('*.py')}

try:
    record, rh = require_record('U013')
    parent = integration.inherited()
    inherited_parent = parent.inherited()
    source = (ORIGINAL/'run_one.py').read_text()
    for before, after in json.loads((BASE/'controlled_changes.json').read_text()):
        assert source.count(before) == 1
        source = source.replace(before, after)
    assert source == (BASE/'run_one.py').read_text()
    oldtree = ast.parse((ORIGINAL/'run_one.py').read_text())
    newtree = ast.parse((BASE/'run_one.py').read_text())
    for name in ('cpu_lock', 'owned_module', 'supervise'):
        old = next(n for n in oldtree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in newtree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old) == ast.dump(new)
        assert ast.get_source_segment((ORIGINAL/'run_one.py').read_text(), old) == ast.get_source_segment(source, new)
    checks.append('Four exact substitutions; supervision/lock/ownership byte-identical; all original science, inspector, parent, runtime and record inventories verify')

    # Import the original entry point without importing scientific model modules.
    sys.path.insert(0, str((root/ORIGINAL).resolve()))
    oldrun = module('u013_old_launcher', root/ORIGINAL/'run_one.py')
    sys.path.pop(0)
    newrun = module('u013_new_launcher', root/BASE/'run_one.py')
    owned = oldrun.owned_module()
    qualification = dict(research_id='U013', research_sha256=rh,
                         software_review_sha256='synthetic fixture only', source_sha256={})

    def launcher_fixture(folder, name, failure=False):
        probe = folder/'outputs/gated/probes'/name
        probe.mkdir(parents=True)
        atomic_json(probe/'result.json', dict(research_id='C011', status='awaiting_literature_review', fixture=True))
        (probe/'cases.npz').write_bytes(b'preserved probe fixture')
        (folder/'outputs/gated/B003').mkdir(parents=True)
        base = folder/'outputs/gated/B004'
        (base/'software').mkdir(parents=True)
        (base/'software/review.json').write_text('synthetic software digest only')
        script = '''import json, sys
from pathlib import Path
name=sys.argv[sys.argv.index('--run')+1]
out=Path('outputs/gated/planning')/name
out.mkdir(parents=True,exist_ok=False)
(out/'result.json').write_text(json.dumps(dict(research_id='B004',status=STATUS,fixture=True)))
(out/'cases.jsonl').write_text('synthetic planning cases\\n')
raise SystemExit(CODE)
'''.replace('STATUS', repr('technical_blocked' if failure else 'awaiting_literature_review')).replace('CODE', '7' if failure else '0')
        (base/'planning.py').write_text(script)
        atomic_json(folder/'outputs/gated/gpu_intervals.json', [])
        pre = base/f'pre_{name}.json'
        consult = dict(inputs_sha256={str(p): digest(p) for p in probe.iterdir()},
                       u013_record_sha256=rh, u013_software_review_sha256='synthetic fixture only')
        atomic_json(pre, consult)
        ctx = dict(research_sha256=require_record('B004')[1], pre=pre, consult=consult, integration=qualification)
        return ctx, probe, {p.name: p.read_bytes() for p in probe.iterdir()}

    def call_main(launcher, folder, name, ctx):
        # Only research context and git metadata are synthetic. main, supervision,
        # subprocess creation/wait and artifact collection execute normally.
        def git_output(argv, **kwargs):
            assert argv[:2] in (['git', 'status'], ['git', 'rev-parse'])
            return '' if argv[1] == 'status' else 'synthetic-fixture-commit\n'
        with pytest.MonkeyPatch.context() as patch, redirect_stdout(io.StringIO()):
            patch.chdir(folder)
            patch.setattr(sys, 'argv', ['fixture', '--run', name, '--python', sys.executable])
            patch.setattr(launcher, 'context', lambda r, n: ctx)
            patch.setattr(launcher, 'owned_module', lambda: owned)
            patch.setattr(launcher.subprocess, 'check_output', git_output)
            launcher.main()

    legacy = None
    with tempfile.TemporaryDirectory(prefix='u013_launcher_') as td:
        temporary = Path(td)
        for name in parent.ORDER:
            folder = temporary/name
            ctx, probe, before = launcher_fixture(folder, name)
            with pytest.raises(AssertionError):
                call_main(oldrun, folder, name, ctx)
            assert not (folder/'outputs/gated/B004/operations'/name).exists()
            legacy = dict(original_source_sha256=digest(root/ORIGINAL/'run_one.py'),
                          observed='AssertionError before operation-directory creation with required probe directory present',
                          fixture_only=True, names=parent.ORDER)
            call_main(newrun, folder, name, ctx)
            op = folder/'outputs/gated/B004/operations'/name
            science = folder/'outputs/gated/planning'/name
            result = json.loads((op/'result.json').read_text())
            execution = json.loads((op/'execution.json').read_text())
            assert result['status'] == 'awaiting_literature_review' and result['returncode'] == 0
            assert result['child_terminal'] and result['inputs_unchanged'] and not result['errors'] and not result['signals']
            assert result['integration'] == execution['integration'] == qualification
            assert result['integration_research_id'] == execution['integration_research_id'] == 'U013'
            assert execution['argv'] == [sys.executable, str(folder/'outputs/gated/B004/planning.py'), '--run', name]
            assert result['child_artifact_sha256'] == {p.name: digest(p) for p in science.iterdir()}
            assert result['no_B004_gpu_intervals'] and result['gpu_ledger_unchanged']
            assert {p.name: p.read_bytes() for p in probe.iterdir()} == before
        checks.append('Original main refuses all four required-probe fixtures before output; corrected main launches real CPU children, collects only planning artifacts and leaves probes unchanged')
        for which in ('planning', 'operations'):
            folder = temporary/f'existing_{which}'
            name = parent.ORDER[0]
            ctx, probe, before = launcher_fixture(folder, name)
            target = folder/('outputs/gated/planning' if which == 'planning' else 'outputs/gated/B004/operations')/name
            target.mkdir(parents=True)
            (target/'sentinel').write_bytes(b'existing evidence')
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(newrun, 'supervise', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('must not launch')))
                with pytest.raises(AssertionError):
                    call_main(newrun, folder, name, ctx)
            assert (target/'sentinel').read_bytes() == b'existing evidence'
            assert {p.name: p.read_bytes() for p in probe.iterdir()} == before
        folder = temporary/'failure'
        name = parent.ORDER[0]
        ctx, probe, before = launcher_fixture(folder, name, failure=True)
        with pytest.raises(SystemExit):
            call_main(newrun, folder, name, ctx)
        failure = json.loads((folder/'outputs/gated/B004/operations'/name/'result.json').read_text())
        assert failure['status'] == 'technical_blocked' and failure['returncode'] == 7 and failure['child_terminal']
        assert failure['raw_status'] == 'technical_blocked' and failure['child_artifact_sha256']
        assert {p.name: p.read_bytes() for p in probe.iterdir()} == before
        checks.append('Existing planning/operation directories refuse before subprocess creation; failed child remains technical_blocked with collected failure artifacts')

    fixtures = functions(root/'outputs/gated/C011/check.py',
                         ['fake_review', 'training_fixture', 'numerical_fixture', 'data_fixture'],
                         dict(Path=Path, atomic_json=atomic_json, digest=digest))
    functions(root/ORIGINAL/'check.py', ['probe_fixture'], fixtures)
    u007 = json.loads((root/'outputs/gated/U007/software/result.json').read_text())
    u006 = json.loads((root/'outputs/gated/U006/software/result.json').read_text())
    with tempfile.TemporaryDirectory(prefix='u013_parents_') as td:
        temporary = Path(td)
        fixtures['data_fixture'](temporary)
        for name in parent.ORDER:
            fixtures['numerical_fixture'](temporary, name)
            sf = temporary/'outputs/gated/C001'/name
            audit = temporary/'outputs/gated/A018/runs'/name
            training = temporary/'outputs/gated/C010/runs'/name
            modify(audit/'manifest.json', integration_research_id='U006', integration=dict(research_sha256=u006['research_sha256']))
            rebind(audit, integration_research_id='U006', manifest_sha256=digest(audit/'manifest.json'))
            modify(sf/'review.json', final_audit_review_sha256=digest(audit/'review.json'))
            modify(training/'manifest.json', integration_research_id='U005', budget_policy='U001', work_seconds=None)
            rebind(training, integration_research_id='U005', manifest_sha256=digest(training/'manifest.json'))
            science = temporary/'outputs/gated/A019/inference'/name
            op = temporary/'outputs/gated/A019/runs'/name
            verification = temporary/'outputs/gated/A019/verification'/name
            modify(science/'manifest.json', budget_policy='U001', budget_hours=None)
            rebind(science, budget_remaining_hours=None)
            modify(op/'manifest.json', integration_research_id='U007', budget_policy='U001', work_seconds=None,
                   integration_research_sha256=u007['research_sha256'])
            rebind(op, integration_research_id='U007', budget_policy='U001', manifest_sha256=digest(op/'manifest.json'),
                   child_artifact_sha256={p.name: digest(p) for p in science.iterdir() if p.name != 'review.json'})
            rebind(verification, integration_research_id='U007', integration=dict(research_sha256=u007['research_sha256']),
                   scientific_result_sha256=digest(science/'result.json'), operational_result_sha256=digest(op/'result.json'))
            modify(science/'review.json', verification_review_sha256=digest(verification/'review.json'))
            ctx = inherited_parent.numerical_parent(temporary, name)
            fixtures['probe_fixture'](temporary, name)
            assert parent.probe_parent(temporary, name, ctx)
        # The first complete route needs no previous-seed fixture. Both chains
        # exercise the real original parent functions, not a replacement gate.
        ctx = parent.inputs(temporary, parent.ORDER[0])
        assert ctx['checkpoint'].name == 'calibrated.pt' and ctx['files']
        science = temporary/'outputs/gated/A019/inference'/parent.ORDER[0]
        review = science/'review.json'
        saved = review.read_bytes()
        modify(review, run_identity=dict(kind='l0', n=3, seed=2))
        refuses(lambda: inherited_parent.numerical_parent(temporary, parent.ORDER[0]))
        review.write_bytes(saved)
        artifact = science/'calibrated.pt'
        saved = artifact.read_bytes()
        artifact.write_bytes(b'changed fixture')
        refuses(lambda: inherited_parent.numerical_parent(temporary, parent.ORDER[0]))
        artifact.write_bytes(saved)
        checks.append('Four original numerical/probe parent fixtures accept additive U005/U006/U007 and null-budget metadata; first full input route passes; wrong seed and changed artifact refuse')

    pre = dict(u013_record_sha256=rh, u013_software_review_sha256=qualification['software_review_sha256'])
    integration.consultation(pre, qualification)
    for key in pre:
        wrong = dict(pre)
        wrong[key] = 'stale'
        refuses(lambda: integration.consultation(wrong, qualification))
        del wrong[key]
        refuses(lambda: integration.consultation(wrong, qualification))
    checks.append('Fresh consultation binds U013 record and software; two stale and two absent bindings refuse')
    try:
        parent.inputs(root, parent.ORDER[0])
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve() == (root/'outputs/gated/C001/l0_n3_s1/review.json').resolve()
    else:
        raise AssertionError('Missing actual training admitted')
    assert not (root/'outputs/gated/planning/l0_n3_s1').exists()
    assert not (root/'outputs/gated/B004/operations/l0_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger) == ledger_before
    verify_files(root, frozen)
    checks.append('Actual absent training refuses; no actual scientific output, Torch import, GPU interval or live/historical helper change')
    result = dict(research_id='U013', research_sha256=rh, status='awaiting_literature_review',
                  check_groups=len(checks), checks=checks, legacy_refusal=legacy,
                  source_sha256={name: digest(BASE/name) for name in HELPERS},
                  controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
                  record_sha256={f'docs/research/records/{name}.json': require_record(name)[1]
                                 for name in ['U013', 'B004', 'C011', 'A019']},
                  runtime_base_commit=record['protocol']['runtime_base_commit'],
                  elapsed_seconds=time.monotonic()-started, torch_imported=False,
                  actual_scientific_runs=0, fixture_children=5,
                  limitation='Complete launcher fixtures use synthetic context and Git metadata. Original scientific parent gates are tested separately with annotated fixtures; actual reviewed training/numerics/probes remain absent. No model or scientific computation rerun.')
    atomic_json(out/'result.json', result)
    print(json.dumps(result), flush=True)
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='U013', status='technical_blocked', checks=checks, error=repr(exc)))
    raise
