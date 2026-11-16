"""CPU metadata/execution qualification; original scientific suites are inherited."""
import ast
import copy
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import pytest
import torch
import common
from common import BASE, ORIGINAL, HELPERS
from go2nav.research import atomic_json, digest, require_record


def refuses(call):
    with pytest.raises((ValueError, FileNotFoundError, KeyError, RuntimeError)):
        call()


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def fixture_review(path, raw, status='passed', **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = path.parent/('inspection.json' if path.name == 'inspection_review.json' else 'result.json')
    atomic_json(result, raw)
    atomic_json(path, dict(research_id=raw['research_id'], result=result.name, result_sha256=digest(result),
        reviewed_at='synthetic fixture', status=status,
        sources=[dict(url='synthetic', version='synthetic', sections='synthetic', consequence='fixture only')],
        measurements='synthetic only', uncertainty='not actual evidence', explanations=['fixture'], next_step='fixture only', **extra))


def measure_fixture(root, stage, name, expected, partial=False, wrong_integration=False):
    p = common.route(root, stage, name)
    sf, op = p['science'], p['operations']
    sf.mkdir(parents=True, exist_ok=True); op.mkdir(parents=True, exist_ok=True)
    p['checkpoint'].parent.mkdir(parents=True, exist_ok=True); p['checkpoint'].write_bytes(b'fixture checkpoint')
    cases = sf/('cases.npz' if stage == 'probes' else 'cases.jsonl'); cases.write_bytes(b'fixture cases')
    raw = dict(research_id=p['science_id'], status='awaiting_literature_review', cases_sha256=digest(cases))
    if stage == 'planning':
        raw['arms'] = {p['kind']: {str(h): dict(count=999 if partial else 1000) for h in [4, 6, 8]}}
    atomic_json(sf/'result.json', raw)
    manifest = dict(execution_integration=expected, preconsultation=dict(inputs_sha256={}))
    atomic_json(op/'manifest.json', manifest)
    operation = dict(research_id='C009', name=name, stage=stage, returncode=0, child_terminal=True,
                     inputs_unchanged=True, timed_out=False, actions=[], errors=[],
                     manifest_sha256=digest(op/'manifest.json'),
                     child_artifact_sha256={'result.json': digest(sf/'result.json'), cases.name: digest(cases)})
    fixture_review(op/'review.json', operation)
    iv = dict(research_id='C014', execution_integration={} if wrong_integration else expected,
              result_sha256=digest(sf/'result.json'), operational_result_sha256=digest(op/'result.json'),
              cases=4999 if partial else 5000, episodes=1000, windows=4999 if partial else 5000)
    fixture_review(sf/'inspection_review.json', iv)
    identity = dict(n=p['n'], kind=p['kind'], seed=p['seed'], checkpoint_sha256=digest(p['checkpoint']))
    fixture_review(sf/'review.json', raw, status='measured_negative', run_identity=identity,
                   inspection_review_sha256=digest(sf/'inspection_review.json'))
    return p


def policy_fixture(root, stage, name, expected, hashes):
    rid = common.SCIENCE[stage]
    o = dict(research_id='C009', scientific_research_id=rid, stage=stage, name=name,
             execution_research_id='U012', budget_policy='U001', timeout_type=None)
    pre = dict(u012_record_sha256=expected['research_sha256'],
               u012_software_review_sha256=expected['software_review_sha256'],
               u001_authorization_sha256=expected['authorization_sha256'])
    m = dict(research_id='C009', scientific_research_id=rid, stage=stage, name=name,
             execution_integration=copy.deepcopy(expected), execution_source_sha256=hashes,
             budget_policy='U001', work_seconds=None,
             remaining_at_admission=dict(global_seconds=None, bc_seconds=None, science_seconds=None),
             stall_seconds=1800, cleanup_reserve_seconds=45, scientific_research_sha256=require_record(rid)[1],
             integration_research_id='C014', integration_research_sha256=require_record('C014')[1],
             preconsultation=pre, outer_interval_id='outer', cwd=str(root))
    sm = dict(research_id=rid, research_sha256=require_record(rid)[1], budget_policy='U001',
              authorization_sha256=expected['authorization_sha256'], budget_hours=None, run_id='inner')
    rows = [dict(id='outer', research_id=rid, budget_policy='U001', supervisor_research_id='C009',
                 execution_research_id='U012', start=1, end=3, out=str(root/ORIGINAL/'runs'/stage/name)),
            dict(id='inner', research_id=rid, budget_policy='U001', start=2, end=3,
                 out=str(root/'outputs/gated'/stage/name))]
    return o, m, sm, rows


began = time.monotonic()
torch.set_num_threads(2)
out = BASE/'software'; out.mkdir(exist_ok=False); (out/'source').mkdir()
for name in HELPERS:
    shutil.copy2(BASE/name, out/'source'/name)
checks = []
ledger = Path('outputs/gated/gpu_intervals.json'); before = digest(ledger)
frozen = {str(p): digest(p) for label in ['C009/v2', 'U001', 'U011']
          for p in Path('outputs/gated', label).glob('*.py')}
try:
    _, rh = require_record('U012')
    original, supervision, unlimited, numerical = common.parents()
    expected = common.qualified(check_self=False); expected['software_review_sha256'] = 'synthetic-only'
    hashes = common.helper_hashes()
    changes = json.loads((BASE/'controlled_changes.json').read_text())
    identical = {}
    for filename, spec in changes.items():
        old = (ORIGINAL/filename).read_text(); transformed = old
        assert digest(ORIGINAL/filename) == spec['original_sha256']
        for change in spec['changes']:
            assert transformed.count(change['before']) == 1
            transformed = transformed.replace(change['before'], change['after'])
        assert (BASE/filename).read_text() == transformed
        if filename.startswith('inspect_'):
            a, b = functions(old), functions(transformed)
            identical[filename] = sorted(set(a)-{'actual'})
            for name in identical[filename]:
                assert ast.dump(a[name]) == ast.dump(b[name])
                assert ast.get_source_segment(old, a[name]) == ast.get_source_segment(transformed, b[name])
    for name in HELPERS:
        ast.parse((BASE/name).read_text())
    for stage in common.SCIENCE:
        tree = ast.parse(Path(f'go2nav/eval/balls_{stage}.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'budget_remaining']
        assert len(calls) == 1
        comparisons = [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and n.left is calls[0]]
        assert len(comparisons) == 1 and isinstance(comparisons[0].ops[0], ast.LtE) and ast.literal_eval(comparisons[0].comparators[0]) == 0
    checks.append('Exact eight-edit supervisor and five-edit inspector copies; every original non-actual inspector function byte/AST-identical; all three scientific budget accesses are comparison-only and original software/source qualifications verify')

    assert len(common.CASES) == 16
    for stage, name in common.CASES:
        p = common.route(Path.cwd(), stage, name)
        assert p == original.route(Path.cwd(), stage, name)
        old = original.arguments('/python', p); new = common.arguments('/python', p)
        assert new == [old[0], str(BASE/'science.py'), *old[2:]]
    for stage, name in [('binding', 'slots_n3_s0'), ('probes', 'o0_n3_s0'), ('planning', 'o1_n3_s0'),
                         ('planning', 'o0_n1_s0'), ('probes', 'o1_n3_s3'), ('other', 'o1_n3_s1'), ('binding', '../slots_n3_s1')]:
        refuses(lambda: common.route(Path.cwd(), stage, name))
    p = common.route(Path.cwd(), 'probes', 'o1_n3_s1')
    assert common.preceding(p) == [('binding', 'slots_n3_s1'), ('probes', 'o0_n3_s1')]
    p = common.route(Path.cwd(), 'planning', 'o1_n3_s2')
    assert common.preceding(p) == [('probes', 'o1_n3_s2'), ('planning', 'o0_n3_s2')]
    assert common.preceding(common.route(Path.cwd(), 'binding', 'slots_n6_s0')) == []
    checks.append('All16 original routes and command tails retained; seven completed/unapproved cases refuse; binding-before-probes and probe/O0-before-planning dependencies explicit')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, 'qualified', lambda: expected)
        patch.setattr(common, 'parents', lambda: (original, supervision, unlimited, numerical))
        patch.setattr(common.research, 'gpu_run', common.research.gpu_run)
        patch.setattr(common.research, 'budget_remaining', common.research.budget_remaining)
        seen = []
        def execute(name, **kwargs):
            assert common.research.gpu_run is unlimited.accounting_run
            assert common.research.budget_remaining is unlimited.accounting_remaining
            seen.append((name, kwargs, list(sys.argv)))
        patch.setattr(runpy, 'run_module', execute)
        for stage in common.SCIENCE:
            name = f'go2nav.eval.balls_{stage}'
            patch.setattr(sys, 'argv', ['science.py', name, '--original-argument'])
            exec(compile((BASE/'science.py').read_text(), str(BASE/'science.py'), 'exec'), {'__name__': '__main__'})
            assert seen[-1] == (name, dict(run_name='__main__', alter_sys=True), ['science.py', '--original-argument'])
        patch.setattr(sys, 'argv', ['science.py', 'go2nav.training.balls'])
        with pytest.raises(AssertionError):
            exec(compile((BASE/'science.py').read_text(), str(BASE/'science.py'), 'exec'), {'__name__': '__main__'})
    checks.append('All three scientific wrappers install exact U001 accounting functions before run_module, retain arguments and refuse another module; no evaluator executed')

    with tempfile.TemporaryDirectory(prefix='u012_') as temporary:
        root = Path(temporary)
        for rid in common.SCIENCE.values():
            lf = root/f'{rid}.json'; sf = root/rid/'science'; sf.mkdir(parents=True)
            prior = dict(id='old', research_id=rid, start=time.time()-60*3600-1, end=time.time()-1)
            atomic_json(lf, [prior]); row, left, work = common.admit(lf, root/rid/'operation', rid)
            assert left == dict(global_seconds=None, bc_seconds=None, science_seconds=None) and work is None
            refuses(lambda: common.admit(lf, root/'other', rid))
            refuses(lambda: common.close_intervals(lf, row['id'], SimpleNamespace(poll=lambda: None), sf, root, rid))
            inner = dict(id='inner', research_id=rid, budget_policy='U001', out=str(sf), start=row['start'], end=None)
            foreign = dict(id='foreign', research_id='foreign', out=str(root/'foreign'), start=row['start'], end=None)
            atomic_json(lf, [prior, row, inner, foreign]); atomic_json(sf/'manifest.json', dict(run_id='inner'))
            refuses(lambda: common.close_intervals(lf, 'wrong', None, sf, root, rid))
            closed = common.close_intervals(lf, row['id'], SimpleNamespace(poll=lambda: 0), sf, root, rid)
            rows = json.loads(lf.read_text()); assert set(closed) == {row['id'], 'inner'} and rows[0] == prior and rows[-1] == foreign
            assert all(v['end'] is not None for v in rows[1:3])
            refuses(lambda: common.close_intervals(lf, row['id'], None, sf, root, rid))
        checks.append('All three science identities admit synthetic60hour histories with null limits; open/live/wrong/double closure refuses; original nested closure preserves foreign intervals')

        for stage, name in [('binding', 'slots_n3_s1'), ('probes', 'o0_n3_s1'), ('planning', 'o1_n3_s1')]:
            good = policy_fixture(root, stage, name, expected, hashes)
            common.check_execution(stage, name, *good, expected, hashes)
        # Corrupt the planning fixture above; original terminal checks remain inherited.
        faults = []
        for index, key, value in [(0, 'research_id', 'wrong'), (0, 'scientific_research_id', 'wrong'),
            (0, 'stage', 'probes'), (0, 'name', 'o0_n3_s1'), (1, 'research_id', 'wrong'),
            (1, 'scientific_research_id', 'wrong'), (1, 'stage', 'probes'), (1, 'name', 'o0_n3_s1'),
            (0, 'execution_research_id', 'wrong'), (1, 'execution_source_sha256', {}),
            (0, 'budget_policy', 'wrong'), (1, 'budget_policy', 'wrong'), (2, 'budget_policy', 'wrong'),
            (2, 'authorization_sha256', 'wrong'), (2, 'budget_hours', 1), (1, 'work_seconds', 1),
            (1, 'remaining_at_admission', {}), (1, 'stall_seconds', 1), (1, 'cleanup_reserve_seconds', 1),
            (0, 'timeout_type', 'no_output_progress'), (2, 'research_id', 'wrong'), (2, 'research_sha256', 'wrong'),
            (1, 'integration_research_id', 'wrong'), (1, 'integration_research_sha256', 'wrong')]:
            bad = copy.deepcopy(good); bad[index][key] = value; faults.append(bad)
        for key in expected:
            bad = copy.deepcopy(good); bad[1]['execution_integration'][key] = 'wrong'; faults.append(bad)
        for key in good[1]['preconsultation']:
            bad = copy.deepcopy(good); bad[1]['preconsultation'][key] = 'wrong'; faults.append(bad)
        for index, key, value in [(0, 'research_id', 'wrong'), (1, 'research_id', 'wrong'),
            (0, 'budget_policy', 'wrong'), (1, 'budget_policy', 'wrong'), (0, 'end', None), (1, 'end', None),
            (0, 'end', 0), (1, 'out', 'foreign'), (0, 'supervisor_research_id', 'wrong'),
            (0, 'execution_research_id', 'wrong'), (1, 'supervisor_research_id', 'wrong'), (1, 'start', 0)]:
            bad = copy.deepcopy(good); bad[3][index][key] = value; faults.append(bad)
        bad = copy.deepcopy(good); bad[3].append(copy.deepcopy(good[3][0])); faults.append(bad)
        for bad in faults:
            refuses(lambda: common.check_execution(stage, name, *bad, expected, hashes))
        checks.append(f'Valid binding/probe/planning execution metadata accepted; {len(faults)} identity, authorization, software, cap, consultation and interval corruptions refuse')

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common, 'qualified', lambda: expected)
            for stage, name in common.CASES:
                measure_fixture(root, stage, name, expected)
                assert common.prior_measure(root, stage, name)
            for stage, name in [('binding', 'slots_n3_s1'), ('probes', 'o0_n3_s1'), ('planning', 'o0_n3_s1')]:
                p = measure_fixture(root, stage, name, expected, partial=True)
                refuses(lambda: common.prior_measure(root, stage, name))
                measure_fixture(root, stage, name, expected, wrong_integration=True)
                refuses(lambda: common.prior_measure(root, stage, name))
                p = measure_fixture(root, stage, name, expected)
                f = p['science']/'review.json'; saved = f.read_bytes(); f.unlink()
                refuses(lambda: common.prior_measure(root, stage, name)); f.write_bytes(saved)
                f = p['science']/('cases.npz' if stage == 'probes' else 'cases.jsonl'); f.write_bytes(b'changed')
                refuses(lambda: common.prior_measure(root, stage, name))
            p = common.route(root, 'binding', 'slots_n3_s1'); data = root/'consulted'; data.write_bytes(b'fixture')
            pre = dict(stage=p['stage'], name=p['name'], research_sha256=require_record('C014')[1],
                       scientific_research_sha256=require_record('C002')[1], inputs_sha256={str(data.resolve()): digest(data)},
                       u012_record_sha256=rh, u012_software_review_sha256=expected['software_review_sha256'],
                       u001_authorization_sha256=expected['authorization_sha256'], consulted_at='fixture',
                       sources=[dict(url='fixture', version='fixture', sections='fixture', consequence='not actual evidence')], consequences='fixture')
            prefile = root/'pre.json'; atomic_json(prefile, pre)
            common.consultation(prefile, p, [data], require_record('C014')[1], require_record('C002')[1])
            for key in ['u012_record_sha256', 'u012_software_review_sha256', 'u001_authorization_sha256', 'inputs_sha256']:
                bad = dict(pre); bad.pop(key); atomic_json(prefile, bad)
                refuses(lambda: common.consultation(prefile, p, [data], require_record('C014')[1], require_record('C002')[1]))
            atomic_json(prefile, pre); data.write_bytes(b'changed')
            refuses(lambda: common.consultation(prefile, p, [data], require_record('C014')[1], require_record('C002')[1]))
        checks.append('All16 synthetic prior-measure negatives admit after passing inspections/operations; partial coverage, wrong integration, missing reviews and changed cases refuse; exact new/original consultation fields and changed bytes refuse')

        # The new numerical-parent adapter consumes U011 metadata; C014's original
        # full checkpoint/array admission is retained unchanged ahead of it.
        p = common.route(root, 'binding', 'slots_n3_s1'); sf = root/'outputs/gated/C014'/p['name']; op = root/'outputs/gated/C014/operations'/p['name']
        sf.mkdir(parents=True); op.mkdir(parents=True)
        nq = numerical.qualified(); nh = numerical.integration_hashes()
        nr = dict(research_id='C014', name=p['name'], research_sha256=require_record('C014')[1], integration=nq, budget_policy='U001', work_seconds=None)
        no = dict(research_id='C014', integration_research_id='U011', budget_policy='U001', timeout_type=None, status='awaiting_literature_review')
        nm = dict(research_id='C014', integration=nq, budget_policy='U001', work_seconds=None, integration_helper_sha256=nh)
        np = dict(name=p['name'], research_sha256=require_record('C014')[1], inputs_sha256={}, software_review_sha256='original',
                  u011_record_sha256=nq['research_sha256'], u011_software_review_sha256=nq['software_review_sha256'], u001_authorization_sha256=nq['authorization_sha256'])
        nom = dict(research_id='C014', name=p['name'], integration=nq, budget_policy='U001', work_seconds=None,
                   remaining=dict(global_seconds=None, reserve_seconds=None), stall_seconds=1800, cleanup_reserve_seconds=45,
                   integration_helper_sha256=nh, preconsultation=np, inputs={}, software_review_sha256='original', outer_interval_id='numeric', cwd=str(root))
        ni = dict(integration_research_id='U011', integration=nq)
        for f, value in [(sf/'result.json', nr), (op/'result.json', no), (sf/'manifest.json', nm), (op/'manifest.json', nom), (sf/'inspection.json', ni)]:
            atomic_json(f, value)
        atomic_json(root/'outputs/gated/gpu_intervals.json', [dict(id='numeric', budget_policy='U001', integration_research_id='U011', start=1, end=3, out=str(op))])
        bound = set(); common.numerical_parent(p, bound); assert len(bound) == len(nh)
        for f, key, value in [(sf/'inspection.json', 'integration', {}), (op/'result.json', 'integration_research_id', 'wrong'),
                              (sf/'manifest.json', 'integration_helper_sha256', {}), (op/'manifest.json', 'work_seconds', 1)]:
            saved = f.read_bytes(); bad = json.loads(saved); bad[key] = value; atomic_json(f, bad)
            refuses(lambda: common.numerical_parent(p, set())); f.write_bytes(saved)
        checks.append('Exact U011 numerical qualification/helpers/inspection accepted; changed inspection, operation integration, helper inventory or time cap refused')
    try:
        common.inputs(common.route(Path.cwd(), 'binding', 'slots_n3_s1'))
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve() == Path('outputs/gated/C014/slots_n3_s1/review.json').resolve()
    else:
        raise AssertionError('Actual absent numerical result must refuse')
    assert not Path('outputs/gated/binding/slots_n3_s1').exists()
    assert not torch.cuda.is_initialized() and digest(ledger) == before and all(digest(f) == h for f, h in frozen.items())
    checks.append('Actual missing later numerical review refuses before output/GPU; no tensor computation, shared ledger or live/historical helper changes')
    sw = json.loads((ORIGINAL/'software/result.json').read_text())
    records = {f'docs/research/records/{r}.json': require_record(r)[1] for r in ['B001', 'C001', 'C002', 'C009', 'C014', 'U001', 'U011', 'U012']}
    result = dict(research_id='U012', research_sha256=rh, status='awaiting_literature_review', check_groups=len(checks), checks=checks,
                  metadata_corruptions=len(faults), unchanged_inspector_functions=identical, elapsed_seconds=time.monotonic()-began,
                  source_sha256=common.helper_hashes(), repository_source_sha256=sw['repository_source_sha256'], record_sha256=records,
                  controlled_changes_sha256=digest(BASE/'controlled_changes.json'), cuda_initialized=False, actual_evaluations=0,
                  limitation='CPU provenance/execution fixtures only; original scientific, inspector and owned-process suites inherited by source identity. No actual later binding/probe/planning outcome.')
    atomic_json(out/'result.json', result)
    print(json.dumps({k: result[k] for k in ['check_groups', 'metadata_corruptions', 'elapsed_seconds', 'checks']}, indent=2))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='U012', status='technical_blocked', checks=checks, error=repr(exc), elapsed_seconds=time.monotonic()-began))
    raise
