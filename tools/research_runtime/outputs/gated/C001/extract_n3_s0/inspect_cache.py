"""One-time C001 terminal cache inspection; no model execution or result approval."""
import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_review, verify_files

began = time.monotonic()
root = Path.cwd()
base = root / 'outputs/gated'
science = base / 'C001/extract_n3_s0'
operation = base / 'C005/v2/runs/extract_n3_s0'
load = lambda p: json.loads(p.read_text())
raw = load(science / 'result.json')
index = load(science / 'index.json')
manifest = load(science / 'manifest.json')
op = load(operation / 'result.json')
execution = load(operation / 'manifest.json')
launch = load(base / 'C005/v2/launch_extract_n3_s0.json')
pre_path = base / 'C005/v2/pre_extract_n3_s0.json'
pre = load(pre_path)
assert raw['status'] == 'awaiting_literature_review'
assert raw['cache_sha256'] == digest(science / 'index.json')
assert index['n'] == 3 and index['seed'] == 0
assert op['returncode'] == 0 and op['child_terminal']
assert not op['timed_out'] and not op['errors'] and not op['actions']
assert digest(operation / 'manifest.json') == op['manifest_sha256']
verify_files(science, op['child_artifact_sha256'])
assert digest(pre_path) == execution['preconsultation_sha256'] == launch['preconsultation_sha256']
assert execution['source_commit'] == launch['source_commit'] == manifest['code_commit']
verify_files(root, manifest['source_sha256'])
verify_files(root, manifest['inputs'])
verify_files(base / 'C005/v2', execution['source_sha256'])
original_data = load(base / 'C005/pre_slots_n3_s0.json')['data_sha256']
assert pre['data_sha256'] == original_data
verify_files(root, original_data)
verify_files(root / 'outputs/data/balls/n3', index['data_metadata'])
assert Path(index['encoder']).resolve() == (base / 'C001/slots_n3_s0/model.pt').resolve()
assert index['encoder_sha256'] == digest(index['encoder']) == pre['checkpoint_sha256']
for folder in ['C001/slots_n3_s0', 'C008/runs/slots_n3_s0', 'C005/runs/slots_n3_s0',
               'C004/slots_n3_s0', 'C007/runs/slots_n3_s0']:
    assert require_review(base / folder / 'review.json')['status'] == 'passed'
assert digest(base / 'C004/slots_n3_s0/review.json') == pre['parent_review_sha256']
for identity in [execution, launch]:
    pid, created = identity['pid'], identity['creation_time']
    assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time() != created
rows = load(base / 'gpu_intervals.json')
owned = [r for r in rows if r['id'] in [manifest['run_id'], execution['outer_interval_id']]]
assert len(owned) == 2 and {r['research_id'] for r in owned} == {'C001', 'C005'}
assert all(r['end'] is not None and r['end'] >= r['start'] for r in owned)
assert all(Path(r['out']).resolve() in [science.resolve(), operation.resolve()] for r in owned)
splits = {}
for split, episodes in [('train', 1000), ('val', 100)]:
    path = science / f'{split}.npy'
    assert digest(path) == index['splits'][split]
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    frames = np.load(root / f'outputs/data/balls/n3/{split}/frames.npy', mmap_mode='r', allow_pickle=False)
    assert values.shape == (episodes, 100, 5, 128)
    assert values.shape[:2] == frames.shape[:2]
    assert values.dtype == np.dtype('float32') and not values.flags.writeable
    assert path.stat().st_size == values.offset + values.nbytes
    for offset in range(0, episodes, 32):
        assert np.isfinite(values[offset:offset+32]).all()
    splits[split] = dict(shape=list(values.shape), dtype=str(values.dtype),
                         elements=int(values.size), all_finite=True,
                         array_bytes=int(values.nbytes), file_bytes=path.stat().st_size,
                         minimum=float(values.min()), maximum=float(values.max()),
                         sha256=digest(path), frames_shape=list(frames.shape))
assert not list(science.glob('*.partial.npy'))
assert 'torch' not in sys.modules
result = dict(research_id='C001', name='extract_n3_s0',
              inspected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scientific_result_sha256=digest(science/'result.json'),
              operational_result_sha256=digest(operation/'result.json'),
              index_sha256=digest(science/'index.json'), encoder_sha256=index['encoder_sha256'],
              source_commit=execution['source_commit'], source_files=len(manifest['source_sha256']),
              original_data_sha256=original_data, splits=splits, closed_owned_intervals=owned,
              exact_owned_processes_absent=True, no_partial_cache_files=True,
              all_declared_artifact_checks_passed=True, torch_imported=False,
              script_sha256=digest(__file__), elapsed_seconds=time.monotonic()-began,
              limitation='Full saved-array structure/finiteness and byte/source/provenance inspection. '
                         'No independent re-encoding, semantic binding test or model execution. '
                         'Sequential episode carry and no future-frame dependency follow the inspected bound source; '
                         'later independently initialized histories and goals still require evaluation.')
out = science / 'inspection.json'
assert not out.exists()
atomic_json(out, result)
print(json.dumps(result), flush=True)
