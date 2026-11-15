"""Literature preregistration, result-review gates and run provenance for P25/P26.

Records are scientific attestations, not automatic literature reviews. A successor
must name the reviewed result and its exact digest; missing/stale reviews fail closed.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO/'docs/research/records'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with tmp.open('w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def require_review(path, result=None):
    path = Path(path)
    review = json.loads(path.read_text())
    for key in ('research_id', 'result', 'result_sha256', 'reviewed_at', 'sources',
                'measurements', 'uncertainty', 'explanations', 'next_step', 'status'):
        if not review.get(key):
            raise ValueError(f'{path}: missing result-review field {key}')
    result = Path(result) if result else path.parent/review['result']
    if digest(result) != review['result_sha256']:
        raise ValueError(f'{path}: stale result review')
    for source in review['sources']:
        if not all(source.get(k) for k in ('url', 'version', 'sections', 'consequence')):
            raise ValueError(f'{path}: incomplete post-result literature consultation')
    return review


def require_record(research_id, *, records=RECORDS):
    if Path(research_id).name != research_id:
        raise ValueError('research ID must be a record name')
    path = Path(records)/f'{research_id}.json'
    record = json.loads(path.read_text())
    for key in ('id', 'reviewed_at', 'question', 'hypothesis', 'sources', 'differences',
                'contrary_evidence', 'protocol', 'success', 'stop', 'resources'):
        if not record.get(key):
            raise ValueError(f'{research_id}: missing preregistration field {key}')
    if record['id'] != research_id or record.get('stage') != 'preregistered':
        raise ValueError(f'{research_id}: not preregistered')
    for source in record['sources']:
        if not all(source.get(k) for k in ('url', 'version', 'sections', 'consequence')):
            raise ValueError(f'{research_id}: incomplete literature consultation')
    for predecessor in record.get('predecessors', []):
        review = require_review(path.parent/predecessor['review'])
        if review['status'] not in predecessor.get('allow', ['passed']):
            raise ValueError(f'{research_id}: predecessor does not pass')
    return record, digest(path)


def provenance(research_id, config, inputs=()):
    record, research_hash = require_record(research_id)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    # Include new and modified source files, not just the HEAD/dirty flag.
    sources = [p for base in ('go2nav', 'configs', 'docs/research')
               for p in (REPO/base).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.md')]
    return dict(research_id=research_id, research_sha256=research_hash, research_record=record,
                created_at=datetime.now(timezone.utc).isoformat(), code_commit=commit,
                python=sys.version, packages={p: metadata.version(p) for p in ('torch', 'numpy', 'scipy')},
                source_sha256={str(p.relative_to(REPO)): digest(p) for p in sorted(sources)},
                config=config, inputs={str(p): digest(p) for p in inputs})


def reviewed_artifact(review_path, artifact, field='checkpoint_sha256'):
    review_path = Path(review_path)
    review = require_review(review_path)
    result = json.loads((review_path.parent/review['result']).read_text())
    if review['status'] != 'passed' or result.get(field) != digest(artifact):
        raise ValueError(f'{artifact}: no passing literature review bound to this artifact')
    return result


def verify_files(root, hashes):
    """Verify the data bytes named by a reviewed index, not only the index itself."""
    for name, expected in hashes.items():
        if digest(Path(root)/name) != expected:
            raise ValueError(f'{name}: artifact changed since its index was recorded')


def dedicated_device(memory_limit=.85):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A dedicated GPU is required; CPU fallback is disabled')
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 16*2**30:
        raise RuntimeError(f'cuda:0 is not the dedicated training GPU: {props.name}')
    free, total = torch.cuda.mem_get_info(0)
    if 1-free/total >= memory_limit:
        raise RuntimeError('GPU already at or above the 85% memory ceiling')
    return torch.device('cuda:0')


def require_training_slot():
    """New training is sequential and waits for the six preserved runs to finish."""
    import psutil
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['pid'] == os.getpid():
            continue
        args = process.info['cmdline'] or []
        if any(arg.startswith('go2nav.training.') for arg in args):
            active.append(process.info['pid'])
    if active:
        raise RuntimeError(f'Dedicated training slot occupied by {active}; leave existing jobs running and retry later')


def device_hours(intervals):
    """Union elapsed time, so overlapping diagnostics are not double-counted."""
    merged, hours = None, 0.
    for start, end in sorted((r['start'], r.get('end') or time.time()) for r in intervals):
        if merged is None:
            merged = [start, end]
        elif start <= merged[1]:
            merged[1] = max(merged[1], end)
        else:
            hours += merged[1]-merged[0]
            merged = [start, end]
    return (hours+(merged[1]-merged[0] if merged else 0))/3600


@contextmanager
def gpu_run(out, research_id, config, inputs=(), *, ledger=None, limit=48):
    """Record a GPU interval, enforcing admission against the shared allowance.

    Long-running callers must also call budget_remaining at safe checkpoint boundaries.
    A crashed/unclosed interval conservatively continues consuming the allowance.
    """
    import fcntl
    import torch
    manifest = provenance(research_id, config, inputs)
    dedicated_device()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(ledger or os.environ.get('GO2NAV_GPU_LEDGER', REPO/'outputs/gated/gpu_intervals.json'))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex

    def update(end=None):
        with ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = json.loads(ledger.read_text()) if ledger.exists() else []
            if end is None:
                if device_hours(rows) >= limit:
                    raise RuntimeError('GPU budget exhausted')
                rows.append(dict(id=run_id, research_id=research_id, out=str(out), start=time.time(), end=None))
            else:
                next(r for r in rows if r['id'] == run_id)['end'] = end
            atomic_json(ledger, rows)
    update()
    manifest.update(run_id=run_id, gpu_ledger=str(ledger), budget_hours=limit, status='running',
                    device=torch.cuda.get_device_name(0), hip=torch.version.hip,
                    process_memory_ceiling=.85)
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'manifests'/f'{run_id}.json', manifest)
    try:
        yield manifest
        result_path = out/'result.json'
        outcome = json.loads(result_path.read_text()).get('status') if result_path.exists() else manifest.get('outcome')
        manifest['status'] = 'budget_incomplete' if outcome == 'budget_incomplete' else 'awaiting_literature_review'
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


def budget_remaining(manifest):
    return manifest['budget_hours']-device_hours(json.loads(Path(manifest['gpu_ledger']).read_text()))
