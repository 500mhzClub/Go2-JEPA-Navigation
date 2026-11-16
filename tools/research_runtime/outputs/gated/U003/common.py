"""P26 operational integration with unchanged science and U001 time policy."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U003')
ORIGINAL = Path('outputs/gated/D003')
UNLIMITED = Path('outputs/gated/U001')
HELPERS = ('common.py', 'science.py', 'run_stage.py', 'check.py')
BALLS_CHANGES = {'go2nav/eval/balls_binding.py', 'go2nav/eval/balls_planning.py', 'go2nav/eval/balls_probes.py'}

def need(condition, message):
    if not condition: raise ValueError(message)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

def reconcile(root, original_hashes, admitted_hashes):
    changed = set()
    for name, expected in original_hashes.items():
        actual = digest(Path(root)/name)
        if actual != expected:
            need(name in BALLS_CHANGES and admitted_hashes.get(name) == actual, 'Unqualified source change: '+name)
            changed.add(name)
    need(changed == BALLS_CHANGES, 'Unexpected source reconciliation inventory')
    return sorted(changed)

def parents():
    need(require_review(ORIGINAL/'software/review.json')['status'] == 'passed', 'D003 software unreviewed')
    sw = json.loads((ORIGINAL/'software/result.json').read_text())
    need(sw['research_sha256'] == require_record('D003')[1] and sw['d001_record_sha256'] == require_record('D001')[1], 'D003 protocol changed')
    verify_files(ORIGINAL, sw['source_sha256']); verify_files(Path.cwd(), sw['config_sha256']); verify_files(Path.cwd(), sw['frozen_model_sha256'])
    numerical = Path('outputs/gated/C014/o0_n3_s0')
    need(require_review(numerical/'review.json')['status'] == 'passed', 'C014 source admission missing')
    evidence = json.loads((numerical/'result.json').read_text())
    need(evidence['research_sha256'] == require_record('C014')[1], 'C014 protocol changed')
    changed = reconcile(Path.cwd(), sw['repository_source_sha256'], evidence['implementation_sha256'])
    need(require_review(UNLIMITED/'software/review.json')['status'] == 'passed', 'U001 software unreviewed')
    unlimited = json.loads((UNLIMITED/'software/result.json').read_text())
    need(unlimited['research_sha256'] == require_record('U001')[1], 'U001 authorization changed')
    verify_files(UNLIMITED, unlimited['source_sha256']); verify_files(Path.cwd(), unlimited['repository_source_sha256'])
    return (module('u003_original_stages', ORIGINAL/'stages.py'), module('u003_original_supervision', ORIGINAL/'supervision.py'),
            module('u003_unlimited', UNLIMITED/'common.py'), changed)

def core_reviews(root):
    root = Path(root); reviews = []; object_planning = {}
    for kind in ['l0','l1','o0','o1']:
        for phase in ['planning','probes']:
            folder = 'planning_full' if phase == 'planning' and kind in ('o0','o1') else phase
            path = root/'outputs/gated'/folder/f'{kind}_n3_s0/review.json'
            review = require_review(path); data = json.loads((path.parent/review['result']).read_text()); identity = review.get('run_identity', {})
            actual = (data.get('n',identity.get('n')),data.get('kind',identity.get('kind')),data.get('seed',identity.get('seed')))
            need(review['status'] in ('passed','measured_negative') and actual == (3,kind,0), 'Core review/identity missing')
            cases = path.parent/('cases.jsonl' if phase == 'planning' else 'cases.npz')
            need(digest(cases) == data['cases_sha256'], 'Core raw cases changed'); reviews.append(path)
            if folder == 'planning_full': object_planning[kind] = (path, review, data)
    path = root/'outputs/gated/binding/slots_n3_s0/review.json'; review = require_review(path); data = json.loads((path.parent/review['result']).read_text())
    need(review['status'] in ('passed','measured_negative') and (data['n'],data['kind'],data['seed']) == (3,'slots',0), 'Binding core absent'); reviews.append(path)
    for phase in ['D001/adapter_check','D002/legacy']:
        path = root/'outputs/gated'/phase/'review.json'; need(require_review(path)['status'] == 'passed', 'Existing D evidence failed'); reviews.append(path)
    for kind, (science_review, science, raw) in object_planning.items():
        check_path = science_review.parent/'inspection_review.json'; check_review = require_review(check_path)
        checked = json.loads((check_path.parent/check_review['result']).read_text())
        operation_path = root/'outputs/gated/U001/operations'/f'{kind}_n3_s0/review.json'; operation_review = require_review(operation_path)
        operation_result = operation_path.parent/operation_review['result']; operation = json.loads(operation_result.read_text())
        need(check_review['status'] == 'passed' and checked['research_id'] == 'U002' and checked['checks']['cases'] == 3000 and checked['all_task_scoring_summary_checks_passed'], 'Full independent object planning absent')
        need(checked['result_sha256'] == digest(science_review.parent/science['result']) and checked['operational_result_sha256'] == digest(operation_result), 'Planning inspection linkage differs')
        need(operation_review['status'] == 'passed' and operation['research_id'] == 'C009' and operation['budget_policy'] == 'U001' and operation['name'] == f'{kind}_n3_s0', 'Wrong planning operation')
        need(operation['returncode'] == 0 and operation['child_terminal'] and operation['inputs_unchanged'] and not any(operation[k] for k in ['timed_out','actions','errors']), 'Planning operation incomplete')
        need(operation['child_artifact_sha256']['result.json'] == checked['result_sha256'], 'Wrong operation artifact')
        reviews.extend([check_path,operation_path])
    return reviews

def arguments(python, p, stages):
    original = stages.arguments(python, p)
    return [python, str(BASE/'science.py'), original[2], *original[3:]]

def admit(ledger, out):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); rows = json.loads(ledger.read_text())
        need(all(r['end'] is not None for r in rows), 'Open GPU interval')
        row = dict(id=uuid.uuid4().hex,research_id='D003',budget_policy='U001',integration_research_id='U003',out=str(out),start=time.time(),end=None)
        rows.append(row); atomic_json(ledger,rows)
    return row

def progress_paths(p, out):
    paths = [out/'stage.log']
    if p['stage'] == 'export':
        paths += [p['corpus']/folder/split for folder in ['objects','engineered'] for split in ['train','val']]
    elif p['stage'] == 's1': paths += [p['science']/f for f in ['train.jsonl','validation.jsonl','checkpoint.pt','model.pt']]
    return paths
