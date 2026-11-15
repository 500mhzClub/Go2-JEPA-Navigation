"""D002: reproduce the complete historical M5 offline summary on its CPU reference path."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from go2nav.research import atomic_json, digest, provenance, require_record


def compare(expected, actual, prefix='', atol=1e-4):
    """Return every mismatch; key sets and counts are exact, reported medians have a fixed tolerance."""
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [dict(path=prefix, reason='expected mapping')]
        if expected.keys() != actual.keys():
            differences.append(dict(path=prefix, reason='keys differ', missing=sorted(expected.keys()-actual.keys()),
                                    extra=sorted(actual.keys()-expected.keys())))
        for key in expected.keys() & actual.keys():
            differences.extend(compare(expected[key], actual[key], f'{prefix}/{key}', atol))
    elif isinstance(expected, (int, float)):
        tolerance = 0 if prefix.endswith('/count') else atol
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or abs(expected-actual) > tolerance+1e-12:
            differences.append(dict(path=prefix, expected=expected,
                                    actual=actual if isinstance(actual, (int, float)) and math.isfinite(actual) else repr(actual)))
    elif actual != expected:
        differences.append(dict(path=prefix, expected=expected, actual=actual))
    return differences


def run(corpus, mazes, model, reference, out, research_id):
    record, _ = require_record(research_id)
    cfg = record['protocol']
    corpus, mazes, model, reference, out = map(Path, (corpus, mazes, model, reference, out))
    metadata = sorted((corpus/'val').glob('*.json'))
    if len(metadata) != cfg['units']:
        raise ValueError('Legacy corpus count differs from the preregistered reference')
    if os.environ.get('HIP_VISIBLE_DEVICES') != '' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly hide HIP and CUDA for the original CPU reference evaluation')
    os.environ['GO2NAV_RING_CORPUS'] = str(corpus.resolve())
    os.environ['GO2NAV_MAZES_ROOT'] = str(mazes.resolve())
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from go2nav.eval import object_measures
    out.mkdir(parents=True, exist_ok=False)
    inputs = [model, reference, *metadata, *(p.with_suffix('.npz') for p in metadata)]
    manifest = provenance(research_id, cfg, inputs)
    manifest.update(status='running', pid=os.getpid(), device='cpu')
    atomic_json(out/'manifest.json', manifest)
    began = time.monotonic()
    try:
        print(f"Replaying {len(metadata)} legacy units with {cfg['workers']} CPU workers", flush=True)
        report = object_measures.offline(model.resolve(), every=cfg['every'], workers=cfg['workers'])
        atomic_json(out/'summary.json', report)
        differences = compare(json.loads(reference.read_text()), report)
        atomic_json(out/'result.json', dict(research_id=research_id, status='awaiting_literature_review',
                    units=len(metadata), reference_sha256=digest(reference), summary_sha256=digest(out/'summary.json'),
                    regression_passed=not differences, differences=differences))
        manifest['status'] = 'awaiting_literature_review'
        print(json.dumps(dict(regression_passed=not differences, mismatches=len(differences))), flush=True)
    except BaseException as exc:
        manifest.update(status='technical_blocked', error=repr(exc))
        atomic_json(out/'result.json', dict(research_id=research_id, status='technical_blocked', error=repr(exc)))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-began
        atomic_json(out/'manifest.json', manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'mazes', 'model', 'reference', 'out', 'research'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    run(a.corpus, a.mazes, a.model, a.reference, a.out, a.research)


if __name__ == '__main__':
    main()
