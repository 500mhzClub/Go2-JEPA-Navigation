"""T6: fit C1's command-history motion model (closed-form weighted ridge; no seed).

    python -m go2nav.training.command_model --out DIR [--contexts data/training/c1_command_model/contexts.jsonl]

Each training context (data/training/c1_command_model/contexts.jsonl) gives the known commands, the 420-value
applied-command history and the true motion at 100-800 ms; its weight is how often the original training schedule drew
it (7,200 draws over 4,694 contexts). Features and target per horizon h follow `go2nav.models.command`:
- commands known up to h (a tape shorter than h + 1 uses all its commands);
- the nominal integration at h, and the history;
- target: true motion minus the nominal integration, yaw wrapped to (-pi, pi].
The fit is deterministic: the same contexts give the same model, which reproduces the shipped C1 exactly.
Variation for confidence intervals comes from resampling contexts (`--bootstrap SEED`).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from go2nav.models.command import FEATURES, HORIZONS, CommandHistoryModel, nominal
from go2nav.training.common import write_lineage

REPO = Path(__file__).resolve().parents[2]
CONTEXTS = REPO/'data/training/c1_command_model/contexts.jsonl'


def design_matrix(contexts):
    """x (N, 8, 447), residual (N, 8, 3), valid (N, 8), weights (N,)."""
    n = len(contexts)
    x, residual = np.zeros((n, HORIZONS, FEATURES)), np.full((n, HORIZONS, 3), np.nan)
    valid, weights = np.zeros((n, HORIZONS), bool), np.zeros(n)
    for i, c in enumerate(contexts):
        commands = np.asarray(c['commands'], float).reshape(-1, 3)
        base, history = nominal(commands), np.asarray(c['history'])
        for h in range(HORIZONS):
            known = np.zeros((HORIZONS, 3))
            k = min(h+1, len(commands))
            known[:k] = commands[:k]
            x[i, h] = np.concatenate((known.ravel(), base[h], history))
            if c['motions'][h] is not None:
                delta = np.asarray(c['motions'][h])-base[h]
                delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                residual[i, h], valid[i, h] = delta, True
        weights[i] = c['weight']
    return x, residual, valid, weights


def fit(contexts):
    return CommandHistoryModel.fit(*design_matrix(contexts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--contexts', default=str(CONTEXTS))
    p.add_argument('--bootstrap', type=int, help='resample contexts with replacement using this seed')
    a = p.parse_args()
    contexts = [json.loads(line) for line in Path(a.contexts).read_text().splitlines() if line.strip()]
    if a.bootstrap is not None:
        rng = np.random.default_rng(a.bootstrap)
        contexts = [contexts[i] for i in rng.integers(0, len(contexts), len(contexts))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fit(contexts).save(out/'command_only.npz')
    write_lineage(out, component='t6_c1_command_model', seed=a.bootstrap, parents=[],
                  config=dict(penalty=1.0, bootstrap=a.bootstrap is not None),
                  data=dict(contexts=str(a.contexts), count=len(contexts)), checkpoint=out/'command_only.npz')
    print(out/'command_only.npz')


if __name__ == '__main__':
    main()
