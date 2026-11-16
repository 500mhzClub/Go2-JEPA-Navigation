"""C014 selected-case lineage and bounded numerical execution."""
import fcntl
from functools import cache
import importlib.util
import json
from pathlib import Path
import time
import uuid

from go2nav.research import REPO, atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/C014')
HELPERS = ('common.py', 'math_checks.py', 'validation.py', 'run.py', 'inspect_result.py', 'check_numerics.py')


def need(value, message):
    if not value:
        raise ValueError(message)


def case(name):
    record, _ = require_record('C014')
    allowed = record['protocol']['encoder_cases']+record['protocol']['predictor_cases']
    need(name in allowed, 'Unplanned C014 case')
    kind, count, seed = name.split('_')
    return dict(name=name, kind=kind, n=int(count[1:]), seed=int(seed[1:]))


def code_hashes():
    return {str(p.relative_to(REPO)): digest(p) for base in ('go2nav', 'configs')
            for p in sorted((REPO/base).rglob('*')) if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}


def reviewed(directory, research, allowed=('passed',)):
    directory = Path(directory)
    review = require_review(directory/'review.json')
    raw = json.loads((directory/review['result']).read_text())
    need(review['research_id'] == raw['research_id'] == research and review['status'] in allowed,
         'Wrong or nonpassing reviewed evidence')
    return raw, review


@cache
def inherited():
    prior = Path('outputs/gated/C007')
    software, _ = reviewed(prior/'software', 'C007')
    verify_files(prior, software['source_sha256'])
    spec = importlib.util.spec_from_file_location('c014_owned_c007', prior/'supervision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(bound, paths):
    for p in paths:
        p = Path(p)
        need(p.is_file(), f'Required input absent: {p}')
        bound[str(p.resolve())] = digest(p)


def training_inputs(name):
    identity = case(name)
    kind, n, seed = (identity[k] for k in ('kind', 'n', 'seed'))
    directory = Path('outputs/gated/C001')/name
    raw, review = reviewed(directory, 'C001')
    need(all(raw[k] == identity[k] for k in ('kind', 'n', 'seed')), 'Training identity differs')
    audit_dir = Path('outputs/gated/C008/runs')/name
    audit, _ = reviewed(audit_dir, 'C008')
    need(audit['name'] == name and audit['scientific_result_sha256'] == digest(directory/'result.json'), 'Training audit differs')
    need((REPO/review['final_audit_review']).resolve() == (audit_dir/'review.json').resolve()
         and review['final_audit_review_sha256'] == digest(audit_dir/'review.json'), 'Unbound terminal audit')
    candidates = [Path('outputs/gated/C005')/prefix/'runs'/name for prefix in ('', 'v2')]
    candidates = [p for p in candidates if (p/'review.json').exists()]
    need(len(candidates) == 1, 'Missing or ambiguous terminal operation')
    opdir = candidates[0]
    operation, _ = reviewed(opdir, 'C005')
    need(operation['returncode'] == 0 and operation['child_terminal'] and
         not any(operation[k] for k in ('timed_out', 'actions', 'errors')), 'Nonterminal operation')
    need(audit['operational_result_sha256'] == digest(opdir/'result.json'), 'Operation differs from audit')
    verify_files(directory, operation['child_artifact_sha256'])
    verify_files(REPO, audit['input_sha256'])
    manifest = json.loads((directory/'manifest.json').read_text())
    operational = json.loads((opdir/'manifest.json').read_text())
    need(operation['manifest_sha256'] == digest(opdir/'manifest.json'), 'Operation manifest differs')
    # Training provenance is checked against its historical checkout, not silently
    # rewritten to pretend the new inference code was used for training.
    runtime = Path(operational['cwd'])
    original = {p: h for p, h in manifest['source_sha256'].items() if p.startswith(('go2nav/', 'configs/'))}
    need(original and manifest['code_commit'] == operational['source_commit'], 'Missing training source')
    verify_files(runtime, original)
    for p in ('go2nav/models/balls_slots.py', 'go2nav/training/balls_slots.py', 'go2nav/training/balls.py'):
        need(digest(REPO/p) == original[p], 'Training/model computation changed')
    config = json.loads((directory/'config.json').read_text())
    need(all(config[k] == identity[k] for k in ('kind', 'n', 'seed')), 'Training configuration identity differs')
    need(manifest['research_sha256'] == config['research_sha256'] == require_record('C001')[1], 'Training record differs')
    data = operational['preconsultation']['data_sha256']
    expected = {f'outputs/data/balls/n{n}/{s}/{f}' for s in ('train', 'val') for f in ('frames.npy', 'labels.npz', 'meta.json')}
    need(set(data) == expected, 'Incomplete original dataset binding')
    verify_files(REPO, data)
    verify_files(REPO, config['parents'])
    bound = {}
    bind(bound, [directory/f for f in ('model.pt', 'checkpoint.pt', 'config.json', 'metrics.jsonl', 'manifest.json', 'result.json', 'review.json')])
    bind(bound, [audit_dir/f for f in ('result.json', 'review.json')])
    bind(bound, [opdir/f for f in ('manifest.json', 'result.json', 'review.json')])
    bind(bound, [REPO/p for p in data])
    bind(bound, [REPO/p for p in config['parents']])
    if kind != 'slots':
        indices = [REPO/p for p in config['parents'] if Path(p).name == 'index.json']
        need(len(indices) == 1, 'Ambiguous cache parent')
        index_path = indices[0]
        index = json.loads(index_path.read_text())
        need((index['n'], index['seed']) == (n, seed), 'Cache identity differs')
        verify_files(index_path.parent, {f'{s}.npy': h for s, h in index['splits'].items()})
        need(digest(index['encoder']) == index['encoder_sha256'], 'Encoder parent differs')
        bind(bound, [index_path, Path(index['encoder']), *[index_path.parent/f'{s}.npy' for s in ('train', 'val')]])
    return identity, bound


def inputs(name):
    identity, bound = training_inputs(name)
    _, rh = require_record('C014')
    if identity['kind'] != 'slots':
        encoder = f"slots_n{identity['n']}_s{identity['seed']}"
        from go2nav.eval.slots_inference import require_numerical_review
        require_numerical_review(Path('outputs/gated/C001')/encoder/'model.pt', BASE/encoder/'review.json')
        for directory in (BASE/encoder, BASE/'operations'/encoder):
            reviewed(directory, 'C014')
            bind(bound, [p for p in directory.glob('*') if p.is_file()])
        require_review(BASE/encoder/'inspection_review.json')
    loader, _ = reviewed(BASE/'software_loader', 'C014')
    need(loader['research_sha256'] == rh, 'Loader review record differs')
    verify_files(REPO, loader['implementation_sha256'])
    need(digest(BASE/'check_loader.py') == loader['helper_sha256'], 'Loader verification changed')
    bind(bound, [BASE/'software_loader/result.json', BASE/'software_loader/review.json', BASE/'check_loader.py',
                 REPO/'docs/research/records/C014.json', BASE/'pre_numerical_implementation.json'])
    # Immutable previous evidence supplies gradient-name identities and provenance
    # of the reused math/process helpers, including the original failed case.
    for stage, directory in [('C004', Path('outputs/gated/C004/slots_n3_s0')),
                             ('C004', Path('outputs/gated/C004/o0_n3_s0')),
                             ('C013', Path('outputs/gated/C013/retry_001/o0_n3_s0')),
                             ('C013', Path('outputs/gated/C013/retry_001/operations/o0_n3_s0'))]:
        reviewed(directory, stage, ('passed', 'technical_blocked'))
        bind(bound, [directory/'result.json', directory/'review.json'])
    for directory in [Path('outputs/gated/C007'), Path('outputs/gated/A016')]:
        sw, _ = reviewed(directory/'software', directory.name)
        need(digest(directory/'supervision.py') == sw['source_sha256']['supervision.py'], 'Inherited process code changed')
        bind(bound, [directory/'supervision.py', directory/'software/result.json', directory/'software/review.json'])
    prior = Path('outputs/gated/C013')
    sw, _ = reviewed(prior/'software', 'C013')
    verify_files(prior, sw['source_sha256'])
    bind(bound, [prior/'validation.py', prior/'software/result.json', prior/'software/review.json'])
    return identity, bound


def admit(ledger, out):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        need(all(r['end'] is not None for r in rows), 'Open GPU interval')
        remaining = inherited().remaining_seconds(rows, time.time())
        work = min(1800, *remaining.values())-45
        need(work > 0, 'Insufficient reserve/global time')
        row = dict(id=uuid.uuid4().hex, research_id='C004', supervisor='C014', out=str(out.resolve()), start=time.time(), end=None)
        rows.append(row)
        atomic_json(ledger, rows)
    return row, remaining, work


def close(ledger, row_id, child):
    need(child is None or child.poll() is not None, 'Owned child remains live')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        found = [r for r in rows if r['id'] == row_id]
        need(len(found) == 1 and found[0]['research_id'] == 'C004' and found[0].get('supervisor') == 'C014'
             and found[0]['end'] is None, 'Wrong interval identity or state')
        need(time.time() >= found[0]['start'], 'Ledger clock reversal')
        found[0].update(end=time.time(), closure='C014: exact owned child confirmed terminal')
        atomic_json(ledger, rows)
