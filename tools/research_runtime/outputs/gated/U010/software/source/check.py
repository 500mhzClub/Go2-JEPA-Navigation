"""Metadata integration checks; original C008 tensor checks are inherited unchanged."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import psutil
import pytest
import metadata
from metadata import BASE, ORIGINAL, TRAINING, HELPERS, need
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files


def refuses(call):
    with pytest.raises((ValueError, FileNotFoundError, KeyError, RuntimeError)):
        call()


def function_nodes(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


began = time.monotonic()
out = BASE/'software'
out.mkdir(exist_ok=False)
(out/'source').mkdir()
for name in HELPERS:
    shutil.copy2(BASE/name, out/'source'/name)
checks = []
ledger = Path('outputs/gated/gpu_intervals.json')
ledger_before = digest(ledger)
frozen = {str(p): digest(p) for label in ['C008', 'U001', 'U006', 'U009']
          for p in Path('outputs/gated', label).glob('*.py')}
try:
    record, rh = require_record('U010')
    changes = json.loads((BASE/'controlled_changes.json').read_text())
    original = (ORIGINAL/'audit.py').read_text()
    assert digest(ORIGINAL/'audit.py') == changes['original_sha256'] == record['protocol']['inherited_audit_sha256']
    expected = original
    for change in changes['changes']:
        assert expected.count(change['before']) == 1
        expected = expected.replace(change['before'], change['after'])
    assert (BASE/'audit.py').read_text() == expected
    old_nodes, new_nodes = function_nodes(original), function_nodes(expected)
    unchanged = ['need', 'equal', 'finite', 'make_model', 'counts', 'state_checks',
                 'terminal_checks', 'verify_source_inventory']
    for name in unchanged:
        assert ast.dump(old_nodes[name]) == ast.dump(new_nodes[name])
        assert ast.get_source_segment(original, old_nodes[name]) == ast.get_source_segment(expected, new_nodes[name])
    for name in HELPERS:
        ast.parse((BASE/name).read_text())
    metadata.qualified(check_self=False)
    sw = metadata.training_software()
    policy = metadata.expected_policy(sw)
    checks.append('Exact ten-change controlled C008 copy; all eight original state/terminal/source functions byte and AST identical; actual C008/U009/U006 qualifications verified')

    names = {f'{kind}_n{n}_s{seed}' for n, seed in [(3, 1), (3, 2), (1, 0), (6, 0)]
             for kind in (['slots', 'o0', 'o1'] if n == 3 else ['slots', 'o1'])}
    assert len(names) == 10 and names == metadata.CASES
    for name in names:
        metadata.case_name(name)
    for name in ['slots_n3_s0', 'o0_n3_s0', 'o1_n3_s0', 'extract_n3_s1',
                 'o0_n1_s0', 'o1_n3_s3', 'slots_n2_s0', '../o0_n3_s1']:
        refuses(lambda: metadata.case_name(name))
    checks.append('All ten remaining training cases accepted; eight completed, extraction, other-count/seed and path cases refused')

    # Only these unmodified original functions execute; no tensor import occurs.
    namespace = dict(need=need, psutil=psutil, Path=Path, verify_files=verify_files)
    for name in ['terminal_checks', 'verify_source_inventory']:
        exec(compile(ast.Module(body=[new_nodes[name]], type_ignores=[]), str(BASE/'audit.py'), 'exec'), namespace)
    terminal = namespace['terminal_checks']
    sources = namespace['verify_source_inventory']
    with tempfile.TemporaryDirectory(prefix='u010_') as temporary:
        root = Path(temporary)
        science, operation = root/'science', root/'operation'
        op = dict(research_id='C005', integration_research_id='U009', budget_policy='U001',
                  name='slots_n3_s1', timeout_type=None, status='awaiting_literature_review',
                  child_terminal=True, returncode=0, timed_out=False, actions=[], errors=[],
                  raw_status='awaiting_literature_review', closed_interval_ids=['outer'])
        pre = {key: policy[key] for key in ['u009_record_sha256', 'software_review_sha256',
                                         'c001_record_sha256', 'c005_record_sha256']}
        pre['u001_authorization_sha256'] = policy['authorization_sha256']
        om = dict(research_id='C005', research_sha256=policy['c005_record_sha256'],
                  integration_research_id='U009', integration_research_sha256=policy['u009_record_sha256'],
                  name=op['name'], budget_policy='U001', authorization_sha256=policy['authorization_sha256'],
                  software_review_sha256=policy['software_review_sha256'], source_sha256=policy['source_sha256'],
                  work_seconds=None, remaining_at_admission=dict(global_seconds=None, bc_seconds=None),
                  cleanup_reserve_seconds=45, preconsultation=pre, outer_interval_id='outer',
                  pid=os.getpid(), creation_time=0., cwd=str(root))
        sm = dict(research_id='C001', research_sha256=policy['c001_record_sha256'],
                  budget_policy='U001', authorization_sha256=policy['authorization_sha256'],
                  budget_hours=None, run_id='inner')
        rows = [dict(id='outer', research_id='C005', integration_research_id='U009',
                     budget_policy='U001', start=1, end=3, out=str(operation)),
                dict(id='inner', research_id='C001', budget_policy='U001', start=2, end=3, out=str(science))]
        good = (op, om, sm, rows)
        metadata.check_policy(*good, policy)
        terminal(*good, science, operation)
        faults = []
        for index, key, value in [
            (0, 'name', 'o0_n3_s1'), (0, 'research_id', 'wrong'), (1, 'research_id', 'wrong'),
            (2, 'research_id', 'wrong'), (0, 'integration_research_id', 'wrong'),
            (1, 'integration_research_id', 'wrong'), (0, 'budget_policy', 'wrong'),
            (1, 'budget_policy', 'wrong'), (2, 'budget_policy', 'wrong'),
            (1, 'authorization_sha256', 'wrong'), (2, 'authorization_sha256', 'wrong'),
            (1, 'integration_research_sha256', 'wrong'), (1, 'research_sha256', 'wrong'),
            (2, 'research_sha256', 'wrong'), (1, 'software_review_sha256', 'wrong'),
            (1, 'source_sha256', {}), (1, 'work_seconds', 1),
            (1, 'remaining_at_admission', dict(global_seconds=1, bc_seconds=None)),
            (1, 'remaining_at_admission', dict(global_seconds=None, bc_seconds=1)),
            (2, 'budget_hours', 1), (1, 'cleanup_reserve_seconds', 1),
            (0, 'timeout_type', 'no_output_progress'), (0, 'status', 'budget_incomplete')]:
            bad = copy.deepcopy(good); bad[index][key] = value; faults.append(bad)
        for key in pre:
            bad = copy.deepcopy(good); bad[1]['preconsultation'][key] = 'wrong'; faults.append(bad)
        for index, key, value in [(0, 'budget_policy', 'wrong'), (1, 'budget_policy', 'wrong'),
                                  (0, 'integration_research_id', 'wrong')]:
            bad = copy.deepcopy(good); bad[3][index][key] = value; faults.append(bad)
        for bad in faults:
            refuses(lambda: metadata.check_policy(*bad, policy))
        checks.append(f'Valid original identities plus U009/U001 null allowance accepted; {len(faults)} metadata/software/authorization/consultation/interval corruptions refused')

        terminal_faults = []
        for key, value in [('child_terminal', False), ('returncode', 1), ('timed_out', True),
                           ('actions', ['TERM']), ('errors', ['error']), ('raw_status', 'budget_incomplete'),
                           ('closed_interval_ids', []), ('closed_interval_ids', ['outer', 'foreign'])]:
            bad = copy.deepcopy(good); bad[0][key] = value; terminal_faults.append(bad)
        bad = copy.deepcopy(good); bad[1]['creation_time'] = psutil.Process(os.getpid()).create_time(); terminal_faults.append(bad)
        bad = copy.deepcopy(good); bad[2]['run_id'] = 'outer'; terminal_faults.append(bad)
        for key, value in [('end', None), ('end', 0), ('research_id', 'wrong'), ('out', 'foreign')]:
            bad = copy.deepcopy(good); bad[3][0][key] = value; terminal_faults.append(bad)
        bad = copy.deepcopy(good); bad[3].append(copy.deepcopy(rows[0])); terminal_faults.append(bad)
        for bad in terminal_faults:
            refuses(lambda: terminal(*bad, science, operation))
        checks.append(f'All {len(terminal_faults)} original incomplete/live-process/owned-interval terminal faults refused')

        runtime, audit_root = root/'runtime', root/'audit'
        for folder in [runtime, audit_root]:
            (folder/'go2nav').mkdir(parents=True)
            (folder/'go2nav/code.py').write_text('same computation')
            (folder/'docs').mkdir()
            (folder/'docs/note.md').write_text('old' if folder == runtime else 'new')
        bound = {str(f.relative_to(runtime)): digest(f) for f in runtime.rglob('*') if f.is_file()}
        sources(audit_root, runtime, bound)
        (runtime/'docs/note.md').write_text('changed')
        refuses(lambda: sources(audit_root, runtime, bound))
        (runtime/'docs/note.md').write_text('old')
        (audit_root/'go2nav/code.py').write_text('changed')
        refuses(lambda: sources(audit_root, runtime, bound))
        old_sw = json.loads((ORIGINAL/'software/result.json').read_text())
        reconciled = metadata.reconcile_sources(Path.cwd(), old_sw['repository_source_sha256'])
        assert len(reconciled) == 3
        bad = dict(old_sw['repository_source_sha256'])
        extra = next(name for name in bad if name.startswith('go2nav/') and name not in reconciled)
        bad[extra] = 'wrong'
        refuses(lambda: metadata.reconcile_sources(Path.cwd(), bad))
        checks.append('Original runtime/current-source distinction retained; exact three historical C014 files reconcile and a fourth change refuses')

        data, parent = root/'data', root/'parent_review'
        data.write_bytes(b'original data'); parent.write_bytes(b'original review')
        inputs = [data, parent]
        training_pre = dict(inputs_sha256={str(f): digest(f) for f in inputs})
        metadata.check_prerequisites(training_pre, inputs)
        for f in inputs:
            saved = f.read_bytes(); f.write_bytes(b'changed')
            refuses(lambda: metadata.check_prerequisites(training_pre, inputs))
            f.write_bytes(saved)
        refuses(lambda: metadata.check_prerequisites(dict(inputs_sha256={str(data): digest(data)}), inputs))
        science.mkdir(); operation.mkdir()
        (science/'result.json').write_text('synthetic science')
        (operation/'result.json').write_text('synthetic operation')
        fake = root/'integration'
        (fake/'software').mkdir(parents=True)
        (fake/'software/review.json').write_text('synthetic review only')
        expected_pre = dict(name='slots_n3_s1', research_sha256=require_record('C008')[1],
                            c001_record_sha256=require_record('C001')[1], u010_record_sha256=rh,
                            u010_software_review_sha256=digest(fake/'software/review.json'),
                            u009_software_review_sha256=digest(TRAINING/'software/review.json'),
                            u001_authorization_sha256=require_record('U001')[1],
                            scientific_result_sha256=digest(science/'result.json'),
                            operational_result_sha256=digest(operation/'result.json'),
                            inputs_sha256={str(f.resolve()): digest(f) for f in inputs})
        value = dict(expected_pre, consulted_at='synthetic',
                     sources=[dict(url='synthetic', version='synthetic', sections='synthetic', consequence='fixture only')],
                     consequences='Fixture only; not actual literature consultation')
        prefile = root/'pre.json'
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(metadata, 'BASE', fake)
            atomic_json(prefile, value)
            metadata.consultation(prefile, 'slots_n3_s1', inputs, science, operation)
            for key in list(expected_pre) + ['consulted_at', 'sources', 'consequences']:
                bad = dict(value); bad.pop(key); atomic_json(prefile, bad)
                refuses(lambda: metadata.consultation(prefile, 'slots_n3_s1', inputs, science, operation))
            atomic_json(prefile, value)
            for f in inputs + [science/'result.json', operation/'result.json']:
                saved = f.read_bytes(); f.write_bytes(b'changed')
                refuses(lambda: metadata.consultation(prefile, 'slots_n3_s1', inputs, science, operation))
                f.write_bytes(saved)
        checks.append('Exact training prerequisite inventory and fresh audit consultation bind all input/result/record/review bytes; missing fields, changed data/reviews/results and missing prerequisite refuse')

    # Exercise the real run's admission through the absent operation, without
    # importing Torch or pretending this as-yet unreviewed adapter is qualified.
    namespace.update(time=time, json=json, BASE=ORIGINAL, require_record=require_record,
                     require_review=require_review, case_name=metadata.case_name,
                     qualified=lambda: metadata.qualified(check_self=False),
                     reconcile_sources=metadata.reconcile_sources)
    exec(compile(ast.Module(body=[new_nodes['run']], type_ignores=[]), str(BASE/'audit.py'), 'exec'), namespace)
    with pytest.raises(ValueError, match='Actual terminal operation missing/ambiguous'):
        namespace['run']('slots_n3_s1', 'nonexistent-consultation')
    assert not (ORIGINAL/'runs/slots_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger) == ledger_before
    assert all(digest(f) == h for f, h in frozen.items())
    checks.append('Actual absent later training refuses before tensors or output; no Torch/GPU execution and all shared ledger/active/historical helpers unchanged')

    records = {f'docs/research/records/{r}.json': require_record(r)[1]
               for r in ['C001', 'C005', 'C008', 'U001', 'U006', 'U009', 'U010']}
    result = dict(research_id='U010', research_sha256=rh, status='awaiting_literature_review',
                  checks=checks, check_groups=len(checks), metadata_corruptions=len(faults),
                  terminal_corruptions=len(terminal_faults), unchanged_functions=unchanged,
                  elapsed_seconds=time.monotonic()-began,
                  source_sha256={name: digest(BASE/name) for name in HELPERS},
                  repository_source_sha256=sw['repository_source_sha256'], record_sha256=records,
                  controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
                  inherited_software_review_sha256=digest(ORIGINAL/'software/review.json'),
                  actual_checkpoint_audits=0, cuda_initialized=False,
                  limitation='CPU metadata integration only; original twelve-group C008 tensor evidence is inherited by exact source identity. Each actual terminal checkpoint still requires complete state checks and separate result review.')
    atomic_json(out/'result.json', result)
    print(json.dumps({k: result[k] for k in ['check_groups', 'metadata_corruptions', 'terminal_corruptions', 'elapsed_seconds', 'checks']}, indent=2))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='U010', status='technical_blocked', checks=checks,
                                     error=repr(exc), elapsed_seconds=time.monotonic()-began))
    raise
