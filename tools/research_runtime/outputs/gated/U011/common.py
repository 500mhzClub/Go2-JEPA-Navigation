"""C014 accounting-only integration; original numerical computation is retained."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U011')
ORIGINAL = Path('outputs/gated/C014')
HELPERS = ('common.py', 'validation.py', 'run.py', 'inspect_result.py', 'check.py')
ORIGINAL_HELPERS = ('common.py', 'math_checks.py', 'validation.py', 'run.py', 'inspect_result.py', 'check_numerics.py')
CASES = {f'{kind}_n{n}_s{seed}' for n, seed in [(3, 1), (3, 2), (1, 0), (6, 0)]
         for kind in (['slots', 'o0', 'o1'] if n == 3 else ['slots', 'o1'])}


def need(value, message):
    if not value:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def parents():
    record, _ = require_record('U011')
    protocol = record['protocol']
    need(require_review(ORIGINAL/'software_numerics/review.json')['status'] == 'passed', 'C014 software unreviewed')
    need(digest(ORIGINAL/'software_numerics/review.json') == protocol['c014_software_review_sha256'], 'C014 software review changed')
    sw = json.loads((ORIGINAL/'software_numerics/result.json').read_text())
    need(sw['research_sha256'] == require_record('C014')[1], 'C014 protocol changed')
    verify_files(ORIGINAL, sw['helper_sha256'])
    verify_files(ORIGINAL, protocol['inherited_helpers'])
    verify_files(Path.cwd(), sw['repository_source_sha256'])
    for label, key in [('U001', 'u001_software_review_sha256'), ('U010', 'u010_software_review_sha256')]:
        base = Path('outputs/gated')/label
        need(require_review(base/'software/review.json')['status'] == 'passed', label+' software unreviewed')
        need(digest(base/'software/review.json') == protocol[key], label+' software review changed')
        value = json.loads((base/'software/result.json').read_text())
        need(value['research_sha256'] == require_record(label)[1], label+' protocol changed')
        verify_files(base, value['source_sha256'])
    audit = module('u011_audit_metadata', Path('outputs/gated/U010/metadata.py'))
    audit.qualified()
    return module('u011_original_common', ORIGINAL/'common.py'), module('u011_unlimited', Path('outputs/gated/U001/common.py')), audit


def qualified(check_self=True):
    parents()
    _, rh = require_record('U011')
    if check_self:
        need(require_review(BASE/'software/review.json')['status'] == 'passed', 'U011 software unreviewed')
        sw = json.loads((BASE/'software/result.json').read_text())
        need(sw['research_sha256'] == rh, 'U011 protocol changed')
        verify_files(BASE, sw['source_sha256'])
        verify_files(Path.cwd(), sw['repository_source_sha256'])
        verify_files(Path.cwd(), sw['record_sha256'])
    return dict(research_id='U011', research_sha256=rh,
                software_review_sha256=digest(BASE/'software/review.json') if check_self else None,
                authorization_sha256=require_record('U001')[1])


def case(name):
    need(name in CASES, 'Unapproved or completed numerical case')
    return parents()[0].case(name)


def code_hashes():
    return parents()[0].code_hashes()


def inherited():
    return parents()[0].inherited()


def reviewed(directory, research, allowed=('passed',)):
    return parents()[0].reviewed(directory, research, allowed)


def parent_integration(audit, operation, expected):
    need(audit['integration_research_id'] == 'U010' and audit['integration'] == expected, 'Wrong checkpoint audit integration')
    need(operation['integration_research_id'] == 'U009' and operation['budget_policy'] == 'U001', 'Wrong training integration')


def inputs(name):
    case(name)
    original, _, audit = parents()
    identity, bound = original.inputs(name)
    av = json.loads((Path('outputs/gated/C008/runs')/name/'result.json').read_text())
    om = json.loads((Path('outputs/gated/C005/v2/runs')/name/'manifest.json').read_text())
    parent_integration(av, om, audit.qualified())
    for label in ['U001', 'U009', 'U010', 'U011']:
        base = Path('outputs/gated')/label
        sw = json.loads((base/'software/result.json').read_text())
        original.bind(bound, [base/'software/result.json', base/'software/review.json',
                             Path(f'docs/research/records/{label}.json'),
                             *[base/f for f in sw['source_sha256']]])
    return identity, bound


def integration_hashes():
    return {name: digest(BASE/name) for name in HELPERS}


def extra_consultation(pre, expected=None):
    expected = qualified() if expected is None else expected
    fields = dict(u011_record_sha256=expected['research_sha256'],
                  u011_software_review_sha256=expected['software_review_sha256'],
                  u001_authorization_sha256=expected['authorization_sha256'])
    need(all(pre.get(k) == v for k, v in fields.items()), 'Wrong integration consultation')


def arguments(python, name):
    need(name in CASES, 'Unapproved or completed numerical case')
    return [str(python), str(BASE/'validation.py'), '--run', name]


def admit(ledger, out):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        need(all(row['end'] is not None for row in rows), 'Open GPU interval')
        row = dict(id=uuid.uuid4().hex, research_id='C004', supervisor='C014',
                   integration_research_id='U011', budget_policy='U001',
                   out=str(out.resolve()), start=time.time(), end=None)
        rows.append(row)
        atomic_json(ledger, rows)
    return row, dict(global_seconds=None, reserve_seconds=None), None


def close(ledger, row_id, child):
    return parents()[0].close(ledger, row_id, child)


def progress(out, name):
    atomic_json(out/'progress.json', dict(name=name, at=time.time(), monotonic=time.monotonic(),
                                         budget_policy='U001', event='numerical_check_boundary'))


def progress_paths(science, operation):
    return [operation/'validation.log']+[science/f for f in ['progress.json', 'result.json', 'train.npz', 'val.npz']]


def wait_progress(*args, **kwargs):
    return parents()[1].wait_progress(*args, **kwargs)


def check_policy(result, operation, manifest, operational, rows, expected, hashes):
    need(result['name'] == operational['name'] in CASES, 'Wrong numerical case')
    need(all(v['research_id'] == 'C014' for v in [result, operation, manifest, operational]), 'Wrong original numerical identity')
    need(operation['integration_research_id'] == 'U011', 'Wrong operation integration')
    need(all(v['integration'] == expected for v in [result, manifest, operational]), 'Wrong integration qualification')
    need(all(v['budget_policy'] == 'U001' for v in [result, operation, manifest, operational]), 'Wrong numerical time policy')
    need(all(v['work_seconds'] is None for v in [result, manifest, operational]), 'Superseded numerical time cap')
    need(operational['remaining'] == dict(global_seconds=None, reserve_seconds=None), 'Superseded numerical allocation')
    need(operational['stall_seconds'] == 1800 and operational['cleanup_reserve_seconds'] == 45, 'Wrong numerical stall policy')
    need(operation['timeout_type'] is None and operation['status'] == 'awaiting_literature_review', 'Nonterminal or stalled operation')
    need(manifest['integration_helper_sha256'] == operational['integration_helper_sha256'] == hashes, 'Wrong integration helper inventory')
    extra_consultation(operational['preconsultation'], expected)
    pre = operational['preconsultation']
    need(pre['name'] == result['name'] and pre['research_sha256'] == result['research_sha256']
         and pre['inputs_sha256'] == operational['inputs']
         and pre['software_review_sha256'] == operational['software_review_sha256'], 'Wrong original numerical consultation')
    owned = [row for row in rows if row['id'] == operational['outer_interval_id']]
    need(len(owned) == 1, 'Wrong numerical interval count')
    row = owned[0]
    need(row['budget_policy'] == 'U001' and row['integration_research_id'] == 'U011', 'Wrong numerical interval policy')
    need(row['end'] is not None and row['end'] >= row['start'], 'Numerical interval open or invalid')
    need(Path(row['out']).resolve() == (Path(operational['cwd'])/ORIGINAL/'operations'/result['name']).resolve(), 'Wrong numerical interval path')


def operation_metadata(result, operation, manifest, operational, rows):
    expected = qualified()
    check_policy(result, operation, manifest, operational, rows, expected, integration_hashes())
    runtime = Path(operational['cwd'])
    need(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip()
         == operational['source_commit'] == manifest['code_commit'], 'Pinned numerical runtime changed')
    verify_files(runtime, manifest['source_sha256'])
    verify_files(BASE, manifest['integration_helper_sha256'])
