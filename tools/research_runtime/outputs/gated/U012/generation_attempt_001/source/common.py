"""Remaining C009 evaluations with U001 accounting and explicit U012 lineage."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path.cwd()))
from go2nav import research
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U012')
ORIGINAL = Path('outputs/gated/C009/v2')
HELPERS = ('common.py', 'science.py', 'run_eval.py', 'inspect_binding.py', 'inspect_probes.py', 'inspect_planning.py', 'check.py')
SCIENCE = dict(binding='C002', probes='C001', planning='B001')
CASES = {(stage, f'{kind}_n{n}_s{seed}')
         for n, seed in [(3, 1), (3, 2), (1, 0), (6, 0)]
         for stage in SCIENCE
         for kind in (['slots'] if stage == 'binding' else ['o0', 'o1'] if n == 3 else ['o1'])}
INSPECTIONS = dict(binding='binding_inspection_software', probes='probe_inspection_software', planning='planning_inspection_software')


def need(value, message):
    if not value:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def parents():
    record, _ = require_record('U012')
    protocol = record['protocol']
    for base, key, rid in [(ORIGINAL, 'c009_review', 'C014'), (Path('outputs/gated/U001'), 'u001_review', 'U001'),
                            (Path('outputs/gated/U011'), 'u011_review', 'U011')]:
        need(require_review(base/'software/review.json')['status'] == 'passed', 'Parent software unreviewed')
        need(digest(base/'software/review.json') == protocol['parents'][key], 'Parent software review changed')
        sw = json.loads((base/'software/result.json').read_text())
        need(sw['research_sha256'] == require_record(rid)[1], 'Parent protocol changed')
        verify_files(base, sw['source_sha256'])
        verify_files(Path.cwd(), sw['repository_source_sha256'])
        verify_files(Path.cwd(), sw.get('record_sha256', {}))
    for stage, folder in INSPECTIONS.items():
        base = ORIGINAL/folder
        need(require_review(base/'review.json')['status'] == 'passed', 'Original inspector unreviewed')
        expected = protocol['inspectors'][stage]
        need(digest(base/'review.json') == expected['review_sha256'], 'Inspector review changed')
        sw = json.loads((base/'result.json').read_text())
        need(sw['source_sha256'] == digest(ORIGINAL/f'inspect_{stage}.py') == expected['source_sha256'], 'Inspector source changed')
        verify_files(Path.cwd(), sw['repository_source_sha256'])
    numerical = module('u012_numerical', Path('outputs/gated/U011/common.py'))
    numerical.qualified()
    return (module('u012_original_stages', ORIGINAL/'stages.py'),
            module('u012_original_supervision', ORIGINAL/'supervision.py'),
            module('u012_unlimited', Path('outputs/gated/U001/common.py')), numerical)


def qualified(check_self=True):
    parents()
    _, rh = require_record('U012')
    if check_self:
        need(require_review(BASE/'software/review.json')['status'] == 'passed', 'U012 software unreviewed')
        sw = json.loads((BASE/'software/result.json').read_text())
        need(sw['research_sha256'] == rh, 'U012 protocol changed')
        verify_files(BASE, sw['source_sha256'])
        verify_files(Path.cwd(), sw['repository_source_sha256'])
        verify_files(Path.cwd(), sw['record_sha256'])
    return dict(research_id='U012', research_sha256=rh,
                software_review_sha256=digest(BASE/'software/review.json') if check_self else None,
                authorization_sha256=require_record('U001')[1])


def helper_hashes():
    return {name: digest(BASE/name) for name in HELPERS}


def route(root, stage, name):
    need((stage, name) in CASES, 'Unapproved or completed evaluation')
    return parents()[0].route(root, stage, name)


def arguments(python, p):
    need((p['stage'], p['name']) in CASES, 'Unapproved or completed evaluation')
    argv = parents()[0].arguments(python, p)
    return [argv[0], str(BASE/'science.py'), *argv[2:]]


def numerical_parent(p, files):
    numerical = parents()[3]
    sf = p['root']/'outputs/gated/C014'/p['name']
    op = p['root']/'outputs/gated/C014/operations'/p['name']
    r, o, m, om = [json.loads(f.read_text()) for f in [sf/'result.json', op/'result.json', sf/'manifest.json', op/'manifest.json']]
    expected = numerical.qualified()
    numerical.check_policy(r, o, m, om, json.loads((p['root']/'outputs/gated/gpu_intervals.json').read_text()), expected, numerical.integration_hashes())
    iv = json.loads((sf/'inspection.json').read_text())
    need(iv['integration_research_id'] == 'U011' and iv['integration'] == expected, 'Wrong numerical inspector integration')
    verify_files(Path('outputs/gated/U011'), m['integration_helper_sha256'])
    files.update(Path('outputs/gated/U011')/f for f in m['integration_helper_sha256'])


def preceding(p):
    result = []
    if p['stage'] == 'probes':
        result.append(('binding', f"slots_n{p['n']}_s{p['seed']}"))
        if p['n'] == 3 and p['kind'] == 'o1':
            result.append(('probes', f"o0_n3_s{p['seed']}"))
    if p['stage'] == 'planning':
        result.append(('probes', p['name']))
        if p['n'] == 3 and p['kind'] == 'o1':
            result.append(('planning', f"o0_n3_s{p['seed']}"))
    return result


def prior_measure(root, stage, name):
    p = route(root, stage, name)
    sf, op = p['science'], p['operations']
    review = require_review(sf/'review.json')
    raw = json.loads((sf/review['result']).read_text())
    identity = dict(n=p['n'], kind=p['kind'], seed=p['seed'], checkpoint_sha256=digest(p['checkpoint']))
    need(review['status'] in ['passed', 'measured_negative'] and review['research_id'] == raw['research_id'] == p['science_id'], 'Previous science not reviewed or wrong identity')
    need(raw['status'] == 'awaiting_literature_review' and review['run_identity'] == identity, 'Previous science incomplete or wrong checkpoint')
    cases = sf/('cases.npz' if stage == 'probes' else 'cases.jsonl')
    need(raw['cases_sha256'] == digest(cases), 'Previous cases changed')
    ir = require_review(sf/'inspection_review.json')
    inspection = sf/ir['result']; iv = json.loads(inspection.read_text())
    need(ir['status'] == 'passed' and review['inspection_review_sha256'] == digest(sf/'inspection_review.json'), 'Previous inspection unreviewed or unbound')
    need(iv['execution_integration'] == qualified() and iv['result_sha256'] == digest(sf/'result.json'), 'Previous inspection changed')
    opr = require_review(op/'review.json'); operation = json.loads((op/opr['result']).read_text())
    manifest = json.loads((op/'manifest.json').read_text())
    need(opr['status'] == 'passed' and operation['research_id'] == 'C009' and operation['name'] == name and operation['stage'] == stage, 'Previous operation unreviewed or wrong identity')
    need(operation['returncode'] == 0 and operation['child_terminal'] and operation['inputs_unchanged']
         and not any(operation[k] for k in ['timed_out', 'actions', 'errors']), 'Previous operation incomplete')
    need(operation['manifest_sha256'] == digest(op/'manifest.json') and iv['operational_result_sha256'] == digest(op/'result.json'), 'Previous operation or inspection changed')
    need(manifest['execution_integration'] == qualified(), 'Previous execution qualification changed')
    verify_files(sf, operation['child_artifact_sha256']); verify_files(root, manifest['preconsultation']['inputs_sha256'])
    if stage == 'planning':
        need(all(raw['arms'][p['kind']][str(h)]['count'] == 1000 for h in [4, 6, 8]), 'Previous planning partial')
    elif stage == 'probes':
        need(iv['cases'] == 5000 and iv['episodes'] == 1000, 'Previous probes partial')
    else:
        need(iv['episodes'] == 1000 and iv['windows'] == 5000, 'Previous binding partial')
    return {sf/'review.json', sf/'result.json', sf/'inspection_review.json', inspection, cases,
            op/'review.json', op/'result.json', op/'manifest.json'} | {sf/f for f in operation['child_artifact_sha256']} | {root/f for f in manifest['preconsultation']['inputs_sha256']}


def inputs(p):
    files = set(parents()[0].inputs(p))
    numerical_parent(p, files)
    for stage, name in preceding(p):
        files.update(prior_measure(p['root'], stage, name))
    for label in ['U001', 'U011', 'U012']:
        base = Path('outputs/gated')/label
        sw = json.loads((base/'software/result.json').read_text())
        files.update([base/'software/result.json', base/'software/review.json', Path(f'docs/research/records/{label}.json')])
        files.update(base/f for f in sw['source_sha256'])
    return sorted({f.resolve() for f in files})


def extra_consultation(pre, expected):
    fields = dict(u012_record_sha256=expected['research_sha256'],
                  u012_software_review_sha256=expected['software_review_sha256'],
                  u001_authorization_sha256=expected['authorization_sha256'])
    need(all(pre.get(k) == v for k, v in fields.items()), 'Wrong execution consultation')


def consultation(path, p, files, rh, sh):
    pre = parents()[0].consultation(path, p, files, rh, sh)
    extra_consultation(pre, qualified())
    return pre


def admit(ledger, out, science):
    need(science in SCIENCE.values(), 'Unapproved scientific allocation')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        need(all(row['end'] is not None for row in rows), 'Open GPU interval')
        outer = dict(id=uuid.uuid4().hex, research_id=science, supervisor_research_id='C009',
                     execution_research_id='U012', budget_policy='U001', out=str(out), start=time.time(), end=None)
        rows.append(outer); atomic_json(ledger, rows)
    return outer, dict(global_seconds=None, bc_seconds=None, science_seconds=None), None


def close_intervals(*args, **kwargs):
    return parents()[1].close_intervals(*args, **kwargs)


def stop_owned(*args, **kwargs):
    return parents()[1].stop_owned(*args, **kwargs)


def wait_progress(*args, **kwargs):
    return parents()[2].wait_progress(*args, **kwargs)


def progress_paths(science, operation):
    return [operation/'evaluation.log']+[science/f for f in ['cases.jsonl', 'cases.npz', 'probe.pt', 'result.json']]


def check_execution(stage, name, o, m, sm, rows, expected, hashes):
    need((stage, name) in CASES, 'Unapproved or completed evaluation')
    rid = SCIENCE[stage]
    need((o['research_id'], o['scientific_research_id'], o['stage'], o['name']) == ('C009', rid, stage, name), 'Wrong operation identity')
    need((m['research_id'], m['scientific_research_id'], m['stage'], m['name']) == ('C009', rid, stage, name), 'Wrong operation manifest identity')
    need(o['execution_research_id'] == 'U012' and m['execution_integration'] == expected, 'Wrong execution qualification')
    need(m['execution_source_sha256'] == hashes, 'Wrong execution helper inventory')
    need(o['budget_policy'] == m['budget_policy'] == sm['budget_policy'] == 'U001', 'Wrong time policy')
    need(sm['authorization_sha256'] == expected['authorization_sha256'], 'Wrong time authorization')
    need(sm['budget_hours'] is None and m['work_seconds'] is None
         and m['remaining_at_admission'] == dict(global_seconds=None, bc_seconds=None, science_seconds=None), 'Superseded GPU time cap')
    need(m['stall_seconds'] == 1800 and m['cleanup_reserve_seconds'] == 45 and o['timeout_type'] is None, 'Wrong stall policy')
    need(sm['research_id'] == rid and sm['research_sha256'] == m['scientific_research_sha256'], 'Wrong scientific protocol')
    need(m['integration_research_id'] == 'C014' and m['integration_research_sha256'] == require_record('C014')[1], 'Wrong numerical admission protocol')
    extra_consultation(m['preconsultation'], expected)
    selected = []
    for identifier, path in [(m['outer_interval_id'], Path(m['cwd'])/ORIGINAL/'runs'/stage/name),
                             (sm['run_id'], Path(m['cwd'])/'outputs/gated'/stage/name)]:
        found = [row for row in rows if row['id'] == identifier]
        need(len(found) == 1, 'Missing or ambiguous accounting interval')
        row = found[0]
        need(row['research_id'] == rid and row['budget_policy'] == 'U001' and row['end'] is not None and row['end'] >= row['start'], 'Wrong or unclosed interval')
        need((Path(m['cwd'])/row['out']).resolve() == path.resolve(), 'Wrong interval output')
        selected.append(row)
    outer, inner = selected
    need(outer['id'] != inner['id'] and outer['supervisor_research_id'] == 'C009'
         and outer['execution_research_id'] == 'U012' and not inner.get('supervisor_research_id')
         and inner['start'] >= outer['start'], 'Wrong interval ownership')


def operation_metadata(stage, name, o, m, sm):
    expected = qualified()
    check_execution(stage, name, o, m, sm, json.loads(Path('outputs/gated/gpu_intervals.json').read_text()), expected, helper_hashes())
    runtime = Path(m['cwd'])
    need(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip()
         == m['source_commit'] == sm['code_commit'], 'Pinned runtime changed')
    verify_files(runtime, sm['source_sha256']); verify_files(runtime, sm['inputs'])
    verify_files(BASE, m['execution_source_sha256'])
    prefile = Path(m['preconsultation_path'])
    need(digest(prefile) == m['preconsultation_sha256'] and json.loads(prefile.read_text()) == m['preconsultation'], 'Actual consultation changed')
    p = route(runtime, stage, name)
    files = inputs(p)
    need(m['preconsultation']['inputs_sha256'] == {str(f): digest(f) for f in files}, 'Canonical actual inputs changed or incomplete')
    consultation(prefile, p, files, require_record('C014')[1], require_record(SCIENCE[stage])[1])
