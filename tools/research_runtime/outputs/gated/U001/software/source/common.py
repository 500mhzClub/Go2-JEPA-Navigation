"""User-authorized accounting-only GPU execution; scientific sources stay frozen."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path.cwd()))
from go2nav import research as research
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U001')
HELPERS = ('common.py', 'science.py', 'run.py', 'check.py')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj

def inherited():
    base = Path('outputs/gated/C009/v2')
    assert require_review(base/'software/review.json')['status'] == 'passed'
    sw = json.loads((base/'software/result.json').read_text())
    verify_files(base, sw['source_sha256'])
    verify_files(Path.cwd(), sw['repository_source_sha256'])
    verify_files(Path.cwd(), sw['record_sha256'])
    stages = module('u001_original_stages', base/'stages.py')
    supervision = module('u001_original_supervision', base/'supervision.py')
    owned = module('u001_original_owned', Path('outputs/gated/A016/supervision.py'))
    return stages, supervision, owned

def route(name):
    assert name in ('o0_n3_s0', 'o1_n3_s0'), 'Initial U001 route is paired n3 seed0 planning'
    stages, _, _ = inherited()
    p = stages.route(Path.cwd(), 'planning', name)
    p['science'] = Path.cwd()/'outputs/gated/planning_full'/name
    p['operations'] = Path.cwd()/BASE/'operations'/name
    return p

def inputs(p):
    stages, _, _ = inherited()
    files = set(stages.inputs(p))
    probe = Path('outputs/gated/probes')/p['name']/'review.json'
    pr = require_review(probe)
    assert pr['status'] in ('passed', 'measured_negative')
    assert pr['run_identity'] == dict(n=3, kind=p['kind'], seed=0, checkpoint_sha256=digest(p['checkpoint']))
    files.update([probe, probe.parent/pr['result'], Path('docs/research/records/U001.json')])
    if p['name'] == 'o1_n3_s0':
        for path in [Path('outputs/gated/planning_full/o0_n3_s0/review.json'), Path('outputs/gated/planning_full/o0_n3_s0/inspection_review.json'), BASE/'operations/o0_n3_s0/review.json']:
            review = require_review(path)
            assert review['status'] in ('passed', 'measured_negative')
            files.update([path, path.parent/review['result']])
    return sorted({f.resolve() for f in files})

def arguments(python, p):
    stages, _, _ = inherited()
    original = stages.arguments(python, p)
    return [python, str(BASE/'science.py'), original[2], *original[3:]]

def admit(ledger, out, science):
    assert science == 'B001'
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows): raise ValueError('Open GPU interval')
        row = dict(id=uuid.uuid4().hex, research_id=science, supervisor_research_id='C009', budget_policy='U001', out=str(out), start=time.time(), end=None)
        rows.append(row); atomic_json(ledger, rows)
    return row

@contextmanager
def accounting_run(out, research_id, config, inputs=(), *, ledger=None, limit=None):
    """Original provenance/device checks and interval lifecycle, without time caps."""
    _, authorization = require_record('U001')
    manifest = research.provenance(research_id, config, inputs)
    research.dedicated_device()
    import torch
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ledger = Path(ledger or os.environ.get('GO2NAV_GPU_LEDGER', research.REPO/'outputs/gated/gpu_intervals.json'))
    ledger.parent.mkdir(parents=True, exist_ok=True); run_id = uuid.uuid4().hex
    def update(end=None):
        with ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = json.loads(ledger.read_text()) if ledger.exists() else []
            if end is None:
                rows.append(dict(id=run_id, research_id=research_id, budget_policy='U001', out=str(out), start=time.time(), end=None))
            else:
                found = [r for r in rows if r['id'] == run_id]
                assert len(found) == 1 and found[0]['end'] is None
                found[0]['end'] = end
            atomic_json(ledger, rows)
    update()
    manifest.update(run_id=run_id, gpu_ledger=str(ledger), budget_hours=None, budget_policy='U001', authorization_sha256=authorization,
                    status='running', device=torch.cuda.get_device_name(0), hip=torch.version.hip, process_memory_ceiling=.85)
    try:
        atomic_json(out/'manifest.json', manifest)
        atomic_json(out/'manifests'/f'{run_id}.json', manifest)
        yield manifest
        manifest['status'] = 'awaiting_literature_review'
    except BaseException as exc:
        manifest.update(status='technical_blocked', error=repr(exc))
        if not (out/'result.json').exists():
            atomic_json(out/'result.json', dict(research_id=research_id, status='technical_blocked', error=repr(exc)))
        raise
    finally:
        update(time.time())
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out/'manifest.json', manifest)
        atomic_json(out/'manifests'/f'{run_id}.json', manifest)

def accounting_remaining(manifest):
    assert manifest['budget_policy'] == 'U001'
    assert manifest['authorization_sha256'] == require_record('U001')[1]
    return float('inf')

def wait_progress(child, process, created, argv, cwd, paths, actions, heartbeat=None,
                  stall_seconds=1800, poll_seconds=10, term_seconds=30, kill_seconds=10):
    _, _, owned = inherited()
    last = time.monotonic(); previous = None
    while child.poll() is None:
        if not owned.identity_matches(process, created, argv, cwd):
            if child.poll() is not None: break
            raise RuntimeError('Owned child identity mismatch')
        state = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths)
        if state != previous: previous, last = state, time.monotonic()
        quiet = time.monotonic()-last
        if heartbeat: heartbeat(quiet)
        if quiet >= stall_seconds:
            owned.stop_owned(child, process, created, argv, cwd, actions, term_seconds, kill_seconds)
            return True
        try: child.wait(timeout=min(poll_seconds, stall_seconds-quiet))
        except subprocess.TimeoutExpired: pass
    return False
