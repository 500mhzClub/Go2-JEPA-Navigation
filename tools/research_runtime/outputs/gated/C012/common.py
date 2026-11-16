"""C012 immutable inputs and single-row validation-reserve supervision."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time
import uuid

from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/C012')
NAME = 'o0_n3_s0'


def inherited():
    base = Path('outputs/gated/C007')
    assert require_review(base/'software/review.json')['status'] == 'passed'
    software = json.loads((base/'software/result.json').read_text())
    verify_files(base, software['source_sha256'])
    spec = importlib.util.spec_from_file_location('c012_inherited_c007', base/'supervision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    record, _ = require_record('C012')
    p = record['protocol']
    science = Path('outputs/gated/C004')/NAME
    failed = require_review(science/'review.json')
    assert failed['status'] == 'technical_blocked'
    assert digest(science/'result.json') == p['failed_result_sha256']
    assert digest(science/'review.json') == p['failed_review_sha256']
    manifest = json.loads((science/'manifest.json').read_text())
    computational = {k:v for k,v in manifest['source_sha256'].items()
                     if k.startswith(('go2nav/', 'configs/'))}
    verify_files(Path.cwd(), computational)
    files = {Path(f).resolve() for f in manifest['inputs']}
    verify_files(Path.cwd(), manifest['inputs'])
    for folder in ['C004', 'C007/runs', 'C001', 'C005/v2/runs', 'C008/runs']:
        path = Path('outputs/gated')/folder/NAME
        review = require_review(path/'review.json')
        assert review['status'] == ('technical_blocked' if folder == 'C004' else 'passed')
        files.update([path/'review.json', path/review['result']])
        if (path/'manifest.json').exists():
            files.add(path/'manifest.json')
    assert digest(Path('outputs/gated/C001')/NAME/'model.pt') == p['checkpoint_sha256']
    cache = Path('outputs/gated/C001/extract_n3_s0')
    assert digest(cache/'index.json') == p['cache_index_sha256']
    index = json.loads((cache/'index.json').read_text())
    assert (index['n'], index['seed']) == (3, 0)
    assert digest(index['encoder']) == index['encoder_sha256'] == p['encoder_sha256']
    verify_files(cache, {f'{s}.npy':h for s,h in index['splits'].items()})
    files.update([cache/'index.json', cache/'train.npy', cache/'val.npy', Path(index['encoder'])])
    for folder in [cache, Path('outputs/gated/C004/slots_n3_s0')]:
        assert require_review(folder/'review.json')['status'] == 'passed'
        files.update([folder/'review.json', folder/'result.json'])
    original = json.loads(Path('outputs/gated/C007/pre_o0_n3_s0.json').read_text())['data_sha256']
    expected = {f'outputs/data/balls/n3/{s}/{f}' for s in ['train','val']
                for f in ['frames.npy','labels.npz','meta.json']}
    assert set(original) == expected
    verify_files(Path.cwd(), original)
    files.update(map(Path, original))
    files.update([Path('docs/research/records/C012.json'),
                  Path('outputs/gated/C007/software/review.json'),
                  Path('outputs/gated/A016/software/review.json')])
    return {str(f.resolve()):digest(f) for f in sorted(files)}, computational


def admit(ledger, out, limit=300, cleanup=45):
    assert limit == 300 and cleanup == 45
    base = inherited()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):
            raise ValueError('Open GPU interval')
        now = time.time()
        remaining = base.remaining_seconds(rows, now)
        work = min(limit, *remaining.values())-cleanup
        if work <= 0:
            raise ValueError('Insufficient reserve/global time')
        row = dict(id=uuid.uuid4().hex, research_id='C004', supervisor='C012',
                   out=str(out.resolve()), start=now, end=None)
        rows.append(row)
        atomic_json(ledger, rows)
    return row, remaining, work


def close(ledger, row_id, child):
    if child is not None and child.poll() is None:
        raise RuntimeError('Child remains live')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        found = [r for r in rows if r['id'] == row_id]
        if len(found) != 1 or found[0]['research_id'] != 'C004' or found[0].get('supervisor') != 'C012' or found[0]['end'] is not None:
            raise ValueError('Wrong C012 interval identity/state')
        assert time.time() >= found[0]['start']
        found[0].update(end=time.time(), closure='C012: exact owned child terminal confirmed')
        atomic_json(ledger, rows)
