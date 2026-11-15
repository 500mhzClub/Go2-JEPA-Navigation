"""A002 numerical precision intervention on the exact A001 snapshots."""
import argparse
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.eval.balls_inference import invariance, parity, windows
from go2nav.research import atomic_json, digest, gpu_run, require_record
from go2nav.training.balls import load


def run(root, snapshots, out, research_id):
    record, _ = require_record(research_id)
    paths = sorted(Path(snapshots).glob('*/calibrated.pt'))
    if len(paths) != 6:
        raise ValueError('All six fixed A001 snapshots are required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with gpu_run(out, research_id, record['protocol'], paths):
        torch.cuda.set_per_process_memory_fraction(.10)
        result = dict(research_id=research_id, cases={})
        for path in paths:
            model, _, state = load(path, 'cuda:0')
            p, a = next(windows(root, state['n'], 'val', 64, 42))
            p, a = p[:8], a[:8]
            row = dict(source_sha256=digest(path), modes={})
            for mode, dtype in (('float32_math', torch.float32), ('float64_math', torch.float64)):
                model.to(dtype=dtype)
                with sdpa_kernel(SDPBackend.MATH):
                    inv = invariance(model, p.to(dtype), a.to(dtype))
                    par = parity(model, p[:4].to(dtype), a[:4].to(dtype))
                passed = par['passed'] and all(x['passed'] for m in inv.values() for x in m.values())
                row['modes'][mode] = dict(invariance=inv, parity=par, passed=passed)
                print(path.parent.name, mode, passed, flush=True)
            result['cases'][path.parent.name] = row
        result['status'] = 'awaiting_literature_review'
        atomic_json(out/'result.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'snapshots', 'out', 'research'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    run(a.data, a.snapshots, a.out, a.research)


if __name__ == '__main__':
    main()
