"""Admission and elapsed-only supervision for the frozen F004 training protocol."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = ROOT/'outputs/gated/F004'
ARMS = ('factual', 'branch', 'paired')
RUNNER_SOURCES = ('tools/research/f004_runtime.py', 'tools/research/f004_train.py',
                  'tools/research/f004_run.py', 'tools/research/f004_execution_checks.py')


def need(value, message):
    if not value:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def supervision():
    path = ROOT/'outputs/gated/U001'
    review = require_review(path/'software/review.json')
    software = json.loads((path/'software/result.json').read_text())
    need(review['status'] == 'passed', 'Unqualified inherited supervision')
    verify_files(path, software['source_sha256'])
    verify_files(ROOT, software['repository_source_sha256'])
    return module('f004_qualified_u001', path/'common.py')


def sources():
    primitive = json.loads((BASE/'software/result.json').read_text())['source_sha256']
    verify_files(ROOT, primitive)
    return dict(primitive, **{p:digest(ROOT/p) for p in RUNNER_SOURCES})


def reviewed(path, statuses=('passed',)):
    review = require_review(path)
    need(review['status'] in statuses, f'Unqualified review: {path}')
    return review, Path(path).parent/review['result']


def inventory(arm):
    """Exact evidence required before a training launch; no old O1 prerequisite is forged."""
    need(arm in ARMS, 'Unplanned arm')
    record, rh = require_record('F004')
    require_record('U001')
    paths = [ROOT/'docs/research/records/F004.json', ROOT/'docs/research/records/U001.json',
             BASE/'execution_preimplementation.json', BASE/'additional_inputs.json']
    for phase in ('software', 'data', 'execution_software'):
        review_path = BASE/phase/'review.json'
        review, result_path = reviewed(review_path)
        result = json.loads(result_path.read_text())
        need(result['record_sha256'] == rh, f'Stale {phase} protocol')
        verify_files(ROOT, result['source_sha256'])
        paths.extend([review_path, result_path])
        if phase == 'data':
            verify_files(BASE/'data', result['files'])
            paths.append(BASE/'data/verification.json')
            need(digest(paths[-1]) == review['measurements']['verification_sha256'], 'Changed data verification')
    additional = json.loads((BASE/'additional_inputs.json').read_text())
    need(digest(BASE/'additional_inputs.json') == record['protocol']['inputs']['additional_manifest_sha256'], 'Changed raster inventory')
    verify_files(ROOT/'outputs/data/ring_corpus_v1', additional['rasters'])
    for name in ('review.json','inspection_review.json'):
        p = ROOT/'outputs/gated/planning_full/o0_n3_s0'/name
        review, result_path = reviewed(p, ('passed','measured_negative'))
        need(review['run_identity']['n'] == 3 and review['run_identity']['kind'] == 'o0'
             and review['run_identity']['seed'] == 0, 'Wrong O0 identity')
        paths.extend([p, result_path])
    p = ROOT/'outputs/gated/U001/operations/o0_n3_s0/review.json'
    _, result_path = reviewed(p)
    paths.extend([p, result_path])
    for prior in ARMS[:ARMS.index(arm)]:
        for name in ('review.json','operations/review.json'):
            p = BASE/'runs'/f'{prior}_s0'/name
            review, result_path = reviewed(p)
            need(review['run_identity'] == dict(arm=prior, seed=0), 'Wrong prior training arm')
            paths.extend([p,result_path])
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def validate_pre(pre, arm, bound, commit, source_hashes):
    _, rh = require_record('F004')
    need(pre['arm'] == arm and pre['seed'] == 0 and pre['record_sha256'] == rh, 'Wrong preconsultation identity')
    need(pre['inputs_sha256'] == bound and pre['source_sha256'] == source_hashes, 'Stale/incomplete preconsultation inputs/source')
    need(pre['source_commit'] == commit, 'Wrong pinned source commit')
    need(pre.get('consulted_at') and pre.get('consequences') and pre.get('sources'), 'Missing substantive consultation')
    for s in pre['sources']:
        need(all(s.get(k) for k in ('url','version','sections','consequence')), 'Incomplete consultation source')


def gpu_idle(report):
    need(isinstance(report,list) and report, 'Missing AMD device report')
    for device in report:
        need('process_list' in device, 'Missing process list')
        for process in device['process_list']:
            need(not isinstance(process['process_info'],dict), 'GPU compute process remains active')


def admit(ledger, out, arm):
    need(arm in ARMS, 'Unplanned arm')
    ledger = Path(ledger)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        need(all(r['end'] is not None for r in rows), 'Open GPU interval')
        row = dict(id=uuid.uuid4().hex, research_id='F004', budget_policy='U001',
                   arm=arm, seed=0, out=str(out), start=time.time(), end=None)
        rows.append(row)
        atomic_json(ledger,rows)
    return row


def close_interval(ledger, interval, child):
    need(child is None or child.poll() is not None, 'Owned child remains live; interval stays open')
    ledger = Path(ledger)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        matches = [r for r in rows if r['id'] == interval['id']]
        need(len(matches) == 1 and matches[0] == interval, 'Interval identity/state changed')
        need(time.time() >= interval['start'], 'Clock reversal')
        matches[0].update(end=time.time(), closure='F004 supervisor: owned child terminal confirmed')
        atomic_json(ledger, rows)


def fresh_paths(out):
    need(not Path(out).exists(), 'Fresh output already exists')


def gpu_device():
    import torch
    from go2nav.research import dedicated_device
    need(torch.cuda.is_available(), 'GPU training required; CPU fallback disabled')
    device = dedicated_device(.85)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    return device
