"""CPU-only qualification of time/provenance integration, without new numerics."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import pytest
import common
from common import BASE, ORIGINAL, HELPERS, need
from go2nav.research import atomic_json, digest, require_record, verify_files


def refuses(call):
    with pytest.raises((ValueError, FileNotFoundError, KeyError, RuntimeError)):
        call()


def nodes(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


started = time.monotonic()
out = BASE/'software'
out.mkdir(exist_ok=False)
(out/'source').mkdir()
for name in HELPERS:
    shutil.copy2(BASE/name, out/'source'/name)
checks = []
ledger = Path('outputs/gated/gpu_intervals.json')
ledger_before = digest(ledger)
frozen = {str(p): digest(p) for label in ['C014', 'U001', 'U010']
          for p in Path('outputs/gated', label).glob('*.py')}
try:
    record, rh = require_record('U011')
    original, unlimited, audit = common.parents()
    common.qualified(check_self=False)
    changes = json.loads((BASE/'controlled_changes.json').read_text())
    same = dict(validation=['gradient_names', 'summarize'],
                inspect_result=['recompute', 'assignment_costs', 'inspect_payload'])
    for filename, specification in changes.items():
        text = (ORIGINAL/filename).read_text()
        assert digest(ORIGINAL/filename) == specification['original_sha256']
        expected = text
        for edit in specification['changes']:
            assert expected.count(edit['before']) == 1
            expected = expected.replace(edit['before'], edit['after'])
        assert (BASE/filename).read_text() == expected
        old_nodes, new_nodes = nodes(text), nodes(expected)
        for name in same.get(Path(filename).stem, []):
            assert ast.dump(old_nodes[name]) == ast.dump(new_nodes[name])
            assert ast.get_source_segment(text, old_nodes[name]) == ast.get_source_segment(expected, new_nodes[name])
        if filename == 'validation.py':
            old_loop = next(n for n in ast.walk(old_nodes['run']) if isinstance(n, ast.For) and ast.unparse(n.target) == 'split')
            new_loop = next(n for n in ast.walk(new_nodes['run']) if isinstance(n, ast.For) and ast.unparse(n.target) == 'split')
            assert ast.dump(old_loop) == ast.dump(new_loop)
            assert ast.get_source_segment(text, old_loop) == ast.get_source_segment(expected, new_loop)
            assert '--work-seconds' not in expected and 'numerical deadline' not in expected
            assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in ['work', 'work_seconds']
                           for n in ast.walk(ast.parse(expected)))
            callback = next(n for n in ast.walk(new_nodes['run']) if isinstance(n, ast.FunctionDef) and n.name == 'check')
            assert ast.unparse(callback) == 'def check():\n    progress(out, name)'
            assert "dedicated_device(.5)" in expected and "set_per_process_memory_fraction(.5)" in expected
    for name in HELPERS:
        ast.parse((BASE/name).read_text())
    verify_files(ORIGINAL, record['protocol']['inherited_helpers'])
    checks.append('Exact controlled copies (11 validation, 9 supervisor, 3 inspector edits); all five original summary/inspection functions and entire numerical split loop byte/AST identical; original math/loader/source/parent qualifications retained')

    assert len(common.CASES) == 10
    for name in common.CASES:
        assert common.case(name) == original.case(name)
        assert common.arguments('/python', name) == ['/python', str(BASE/'validation.py'), '--run', name]
    for name in ['slots_n3_s0', 'o0_n3_s0', 'o1_n3_s0', 'o0_n1_s0', 'slots_n2_s0', 'o1_n3_s3', '../o1_n3_s1']:
        refuses(lambda: common.case(name))
        refuses(lambda: common.arguments('/python', name))
    checks.append('All ten remaining case identities and original model arguments retained; only obsolete work-seconds CLI removed; seven completed/unapproved/path cases refused')

    # Execute the unchanged original summary and its existing structural fixture,
    # not model computation, tensors or a new GPU numerical validation.
    validation_nodes = nodes((BASE/'validation.py').read_text())
    namespace = dict(json=json, Path=Path, need=need, copy=copy)
    for name in ['gradient_names', 'summarize']:
        exec(compile(ast.Module(body=[validation_nodes[name]], type_ignores=[]), str(BASE/'validation.py'), 'exec'), namespace)
    fixture = nodes((ORIGINAL/'check_numerics.py').read_text())['fixture']
    exec(compile(ast.Module(body=[fixture], type_ignores=[]), str(ORIGINAL/'check_numerics.py'), 'exec'), namespace)
    for kind, count in [('slots', 1104), ('o0', 2772), ('o1', 4164)]:
        good = namespace['fixture'](kind)
        assert namespace['summarize'](good) == (count, True)
        bad = copy.deepcopy(good); bad['windows']['train'].pop()
        refuses(lambda: namespace['summarize'](bad))
        bad = copy.deepcopy(good); bad['windows']['val'][0]['encoder']['gradients'].pop(next(iter(namespace['gradient_names']()['encoder'])))
        refuses(lambda: namespace['summarize'](bad))
    bad = namespace['fixture']('o1'); bad['windows']['val'][0].pop('masked')
    refuses(lambda: namespace['summarize'](bad))
    negative = namespace['fixture']('o0')
    negative['planning']['train']['checks']['cpu']['rollout']['passed'] = False
    negative['planning']['train']['passed'] = False
    assert namespace['summarize'](negative) == (2772, False)
    checks.append('Original 1104/2772/4164 coverage summaries pass; omitted pairs/gradients/O1 masking refuse and complete numerical failure remains negative')

    with tempfile.TemporaryDirectory(prefix='u011_') as temporary:
        root = Path(temporary)
        lf = root/'ledger.json'
        prior = dict(id='old', research_id='C004', start=time.time()-60*3600-1, end=time.time()-1)
        atomic_json(lf, [prior])
        row, remaining, work = common.admit(lf, root/'operation')
        assert remaining == dict(global_seconds=None, reserve_seconds=None) and work is None
        assert row['research_id'] == 'C004' and row['supervisor'] == 'C014'
        assert row['integration_research_id'] == 'U011' and row['budget_policy'] == 'U001'
        refuses(lambda: common.admit(lf, root/'other'))
        live = SimpleNamespace(poll=lambda: None)
        refuses(lambda: common.close(lf, row['id'], live))
        before = lf.read_bytes()
        refuses(lambda: common.close(lf, 'foreign', None))
        assert lf.read_bytes() == before
        common.close(lf, row['id'], SimpleNamespace(poll=lambda: 0))
        closed = json.loads(lf.read_text())
        assert closed[0] == prior and closed[1]['end'] >= closed[1]['start']
        refuses(lambda: common.close(lf, row['id'], None))
        assert 'Infinity' not in lf.read_text()
        checks.append('Synthetic 60-hour ledger admits with null work/global/reserve values; open interval, live child, foreign and duplicate closure refuse; original closed row preserved')

        sf = root/'science'; sf.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common, 'time', SimpleNamespace(time=lambda: 90*3600, monotonic=lambda: 90*3600))
            common.progress(sf, 'slots_n3_s1')
        progress = json.loads((sf/'progress.json').read_text())
        assert progress['monotonic'] == 90*3600 and progress['budget_policy'] == 'U001'
        paths = common.progress_paths(sf, root/'operation')
        assert paths == [root/'operation/validation.log']+[sf/f for f in ['progress.json', 'result.json', 'train.npz', 'val.npz']]
        with pytest.MonkeyPatch.context() as patch:
            seen = []
            def monitor(*args, **kwargs):
                seen.append((args, kwargs)); return False
            patch.setattr(common, 'parents', lambda: (original, SimpleNamespace(wait_progress=monitor), audit))
            assert common.wait_progress('owned', paths=paths) is False
            assert seen == [(('owned',), dict(paths=paths))]
        checks.append('Existing numerical callback reports progress even at synthetic 90 hours; exact result/log/array paths and unchanged U001 monitor delegation verified')

        expected = common.qualified(check_self=False)
        expected['software_review_sha256'] = 'synthetic-only'
        hashes = common.integration_hashes()
        pre = dict(name='slots_n3_s1', research_sha256=require_record('C014')[1], inputs_sha256={'fixture': 'hash'},
                   software_review_sha256='original-review', u011_record_sha256=expected['research_sha256'],
                   u011_software_review_sha256=expected['software_review_sha256'],
                   u001_authorization_sha256=expected['authorization_sha256'])
        r = dict(research_id='C014', name='slots_n3_s1', research_sha256=require_record('C014')[1],
                 integration=copy.deepcopy(expected), budget_policy='U001', work_seconds=None)
        operation = dict(research_id='C014', integration_research_id='U011', budget_policy='U001',
                         timeout_type=None, status='awaiting_literature_review')
        manifest = dict(research_id='C014', integration=copy.deepcopy(expected), budget_policy='U001',
                        work_seconds=None, integration_helper_sha256=hashes)
        om = dict(research_id='C014', name=r['name'], integration=copy.deepcopy(expected), budget_policy='U001',
                  work_seconds=None, remaining=remaining, stall_seconds=1800, cleanup_reserve_seconds=45,
                  integration_helper_sha256=hashes, preconsultation=pre, inputs=pre['inputs_sha256'],
                  software_review_sha256=pre['software_review_sha256'], outer_interval_id='outer', cwd=str(root))
        rows = [dict(id='outer', research_id='C004', supervisor='C014', integration_research_id='U011',
                     budget_policy='U001', start=1, end=3, out=str(root/ORIGINAL/'operations'/r['name']))]
        good = (r, operation, manifest, om, rows)
        common.check_policy(*good, expected, hashes)
        faults = []
        for index in range(4):
            for key, value in [('research_id', 'wrong'), ('budget_policy', 'wrong')]:
                bad = copy.deepcopy(good); bad[index][key] = value; faults.append(bad)
        for index in [0, 2, 3]:
            for key in expected:
                bad = copy.deepcopy(good); bad[index]['integration'][key] = 'wrong'; faults.append(bad)
            bad = copy.deepcopy(good); bad[index]['work_seconds'] = 1; faults.append(bad)
        for index, key, value in [(1, 'integration_research_id', 'wrong'),
                                 (1, 'timeout_type', 'no_output_progress'), (1, 'status', 'technical_blocked'),
                                 (2, 'integration_helper_sha256', {}), (3, 'integration_helper_sha256', {}),
                                 (3, 'remaining', dict(global_seconds=1, reserve_seconds=None)),
                                 (3, 'remaining', dict(global_seconds=None, reserve_seconds=1)),
                                 (3, 'stall_seconds', 1), (3, 'cleanup_reserve_seconds', 1),
                                 (3, 'name', 'o0_n3_s1')]:
            bad = copy.deepcopy(good); bad[index][key] = value; faults.append(bad)
        for key in pre:
            bad = copy.deepcopy(good); bad[3]['preconsultation'][key] = 'wrong'; faults.append(bad)
        for key, value in [('budget_policy', 'wrong'), ('integration_research_id', 'wrong'),
                           ('end', None), ('end', 0), ('out', 'foreign')]:
            bad = copy.deepcopy(good); bad[4][0][key] = value; faults.append(bad)
        bad = copy.deepcopy(good); bad[4].append(copy.deepcopy(rows[0])); faults.append(bad)
        for bad in faults:
            refuses(lambda: common.check_policy(*bad, expected, hashes))
        checks.append(f'Valid U011/U001 science/operation/manifest metadata accepted; all {len(faults)} identity, record, software, authorization, cap, consultation and interval corruptions refused')

        audit_expected = audit.qualified()
        av = dict(integration_research_id='U010', integration=copy.deepcopy(audit_expected))
        training = dict(integration_research_id='U009', budget_policy='U001')
        common.parent_integration(av, training, audit_expected)
        for key in audit_expected:
            bad = copy.deepcopy(av); bad['integration'][key] = 'wrong'
            refuses(lambda: common.parent_integration(bad, training, audit_expected))
        for key in training:
            bad = dict(training); bad[key] = 'wrong'
            refuses(lambda: common.parent_integration(av, bad, audit_expected))
        checks.append('Qualified U010 audit and U009/U001 trainer lineage accepted; altered parent qualification/training policy refuses on top of original C014 prerequisites')
    try:
        common.inputs('slots_n3_s1')
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve() == Path('outputs/gated/C001/slots_n3_s1/review.json').resolve()
    else:
        raise AssertionError('Actual missing later training must refuse')
    assert not (ORIGINAL/'slots_n3_s1').exists() and not (ORIGINAL/'operations/slots_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger) == ledger_before
    assert all(digest(f) == h for f, h in frozen.items())
    checks.append('Actual absent later training refuses before Torch/GPU/output; shared ledger and active/historical helper bytes unchanged')
    inherited_sw = json.loads((ORIGINAL/'software_numerics/result.json').read_text())
    records = {f'docs/research/records/{r}.json': require_record(r)[1] for r in ['C014', 'U001', 'U009', 'U010', 'U011']}
    result = dict(research_id='U011', research_sha256=rh, status='awaiting_literature_review',
                  checks=checks, check_groups=len(checks), metadata_corruptions=len(faults),
                  elapsed_seconds=time.monotonic()-started, source_sha256=common.integration_hashes(),
                  repository_source_sha256=inherited_sw['repository_source_sha256'], record_sha256=records,
                  controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
                  inherited_software_review_sha256=digest(ORIGINAL/'software_numerics/review.json'),
                  actual_numerical_cases=0, torch_imported=False,
                  limitation='CPU provenance/time-policy integration only; original C014 tensor/array and U001 owned-process evidence inherited unchanged. No new selected-checkpoint numerical result.')
    atomic_json(out/'result.json', result)
    print(json.dumps({k: result[k] for k in ['check_groups', 'metadata_corruptions', 'elapsed_seconds', 'checks']}, indent=2))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='U011', status='technical_blocked', checks=checks,
                                     error=repr(exc), elapsed_seconds=time.monotonic()-started))
    raise
