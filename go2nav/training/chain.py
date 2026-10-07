"""Retrain the whole learned chain from scratch, resumably, and write the model set that missions and cohorts accept.

    python -m go2nav.training.chain OUT --render-root RENDER --recordings-root ROOT [--recordings-root ROOT ...]
                                    [--features DIR] [--seed-offset K] [--c1-bootstrap SEED]

Stages, each in its own process (`go2nav.training.*`), with its log in OUT/logs:
- encoder features, shared by every chain (--features, default OUT/features): T1's frames of the May corpus render
  (RENDER: the render folder of `go2nav.corpus.datagen`, or the May render), and the native and readout features of
  the collected recordings (ROOT: the collections' folders);
- T1 temporal predictor -> T2 native adaptation -> T3 balanced start -> T4 horizon predictor;
- T5-0 -> T5a full-heading readout -> T5b maze-view readout;
- T7: the decoder cache from T4 (about 63 GB; the driver checks the disk keeps its 12-GiB reserve), three
  decoder-fix fits from T5a, and the median-score choice; T8: the stage-2 refit on the same cache;
- T6: the C1 command model (closed form: it has no seed; --c1-bootstrap refits on a bootstrap resample).

Seeds are the original ones plus --seed-offset (0 reproduces the shipped chain's training). A stage whose checkpoint
exists is skipped; a partial stage folder is cleared and the stage rerun. The last step writes OUT/models.yaml, the
artefact-name -> checkpoint map for `python -m go2nav.eval.mission --models` and the cohort option `models`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from go2nav.training import native_predictor as native, readouts, t1_temporal

REPO = Path(__file__).resolve().parents[2]
T7_SEEDS, T8_SEED = (2026092205, 2026093011, 2026093012), 2026093012
CACHE_BYTES, RESERVE_BYTES = 70*1024**3, 12*1024**3


class Chain:
    def __init__(self, out, *, render_root, recordings, features=None, seed_offset=0, c1_bootstrap=None):
        self.out = Path(out).resolve()
        self.features = Path(features).resolve() if features else self.out/'features'
        self.render_root, self.recordings = Path(render_root), [str(Path(r)) for r in recordings]
        self.offset, self.c1_bootstrap = seed_offset, c1_bootstrap
        (self.out/'logs').mkdir(parents=True, exist_ok=True)

    def seed(self, original):
        return original+self.offset

    def stage(self, name, module, args, product, *, mark=False):
        """Run `python -m go2nav.training.<module> args` unless `product` exists, and return it. With `mark`, `product`
        is a completion marker the driver writes after the stage succeeds (for folders without a final file)."""
        product = Path(product)
        if product.exists():
            print(f'[skip] {name}', flush=True)
            return product
        if product.parent.exists():
            shutil.rmtree(product.parent)  # a partial run of this stage
        print(f'[run] {name}', flush=True)
        log = self.out/'logs'/f'{name}.log'
        with open(log, 'w') as stream:
            code = subprocess.run([sys.executable, '-m', f'go2nav.training.{module}', *map(str, args)], cwd=REPO,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        if code == 0 and mark:
            product.touch()
        if code or not product.exists():
            raise RuntimeError(f'stage {name} failed (exit {code}); see {log}')
        return product

    def run(self):
        o, roots = self.out, [a for r in self.recordings for a in ('--recordings-root', r)]
        features = {}
        for name, module, args in (('t1', 't1_temporal', ['--render-root', self.render_root]),
                                   ('native', 'native_predictor', roots), ('readouts', 'readouts', roots)):
            folder = self.features/name  # encoder features: seed-independent, shared by every chain
            self.stage(f'features_{name}', module, ['features', *args, '--out', folder], folder/'.done', mark=True)
            features[name] = folder
        t1_features, native_features, readout_features = features['t1'], features['native'], features['readouts']
        seed = self.seed(t1_temporal.SEED)
        t1 = self.stage('t1', 't1_temporal', ['train', '--features', t1_features, '--out', o/'t1', '--seed', seed],
                        o/'t1'/f't1_seed{seed}_epoch{t1_temporal.CHECKPOINT_EPOCH}.pt')
        t2 = self.stage('t2', 'native_predictor', ['t2', '--parent', t1, '--features', native_features, '--out', o/'t2',
                                                   '--seed', self.seed(native.T2_SEED)], o/'t2'/'t2.pt')
        t3 = self.stage('t3', 'native_predictor', ['t3', '--parent', t2, '--features', native_features, '--out', o/'t3',
                                                   '--seed', self.seed(native.T3_SEED)], o/'t3'/'t3.pt')
        t4 = self.stage('t4', 'native_predictor', ['t4', '--parent', t3, '--features', native_features, '--out', o/'t4',
                                                   '--seed', self.seed(native.T4_SEED)], o/'t4'/'t4.pt')
        t5_0 = self.stage('t5_0', 'readouts', ['t5-0', '--features', readout_features, '--out', o/'t5_0',
                                               '--seed', self.seed(readouts.T5_0_SEED)], o/'t5_0'/'readout.pt')
        t5a = self.stage('t5a', 'readouts', ['t5a', '--parent', t5_0, '--features', readout_features, '--out', o/'t5a',
                                             '--seed', self.seed(readouts.T5A_SEED)], o/'t5a'/'readout.pt')
        t5b = self.stage('t5b', 'readouts', ['t5b', '--parent', t5a, '--features', readout_features, '--out', o/'t5b',
                                             '--seed', self.seed(readouts.T5B_SEED)], o/'t5b'/'readout.pt')
        cache = o/'t7_cache'
        if not (cache/'.done').exists() and shutil.disk_usage(o).free < CACHE_BYTES+RESERVE_BYTES:
            raise RuntimeError(f'the T7 cache needs about {CACHE_BYTES/1024**3:.0f} GiB beyond the 12-GiB reserve; '
                               f'{shutil.disk_usage(o).free/1024**3:.0f} GiB free')
        self.stage('t7_cache', 'decoder', ['cache', '--predictor', t4, *roots, '--out', cache], cache/'.done', mark=True)
        fits = []
        for original in T7_SEEDS:
            seed = self.seed(original)
            folder = o/f't7_s{seed}'
            self.stage(f't7_s{seed}', 'decoder', ['fit', '--cache', cache, '--base-readout', t5a, '--out', folder,
                                                  '--seed', seed], folder/f'fit_decoder_fix_s{seed}.pt')
            fits.append(folder)
        scores = sorted((json.loads((f/'evaluation.json').read_text())['score_mm'], f) for f in fits)
        chosen = scores[len(scores)//2][1]
        (o/'t7_choice.json').write_text(json.dumps(dict(chosen=str(chosen), scores_mm={str(f): s for s, f in scores}),
                                                   indent=1)+'\n')
        seed = self.seed(T8_SEED)
        t8 = self.stage('t8_stage2', 'decoder', ['fit', '--cache', cache, '--base-readout', t5a, '--recipe', 'stage2',
                                                 '--out', o/'t8_stage2', '--seed', seed],
                        o/'t8_stage2'/f'fit_stage2_s{seed}.pt')
        t6 = self.stage('t6', 'command_model', ['--out', o/'t6', *(['--bootstrap', self.c1_bootstrap]
                                                                  if self.c1_bootstrap is not None else [])],
                        o/'t6'/'command_only.npz')
        models = dict(t4_horizon_dense_predictor=t4, t5b_maze_view_readout=t5b,
                      t7_c3_decoder_and_c4=next(chosen.glob('fit_decoder_fix_s*.pt')), t8_stage2_c3_c4_refit=t8,
                      t6_c1_command_model=t6)
        (o/'models.yaml').write_text(f'# Retrained chain, seed offset {self.offset}\n'+yaml.safe_dump(
            {name: str(path) for name, path in models.items()}, sort_keys=False))
        print(f'chain done: {o/"models.yaml"}', flush=True)
        return models


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--render-root', type=Path, required=True)
    p.add_argument('--recordings-root', type=Path, action='append', required=True)
    p.add_argument('--features', type=Path, help='shared encoder features (default OUT/features)')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--c1-bootstrap', type=int)
    a = p.parse_args()
    Chain(a.out, render_root=a.render_root, recordings=a.recordings_root, features=a.features, seed_offset=a.seed_offset,
          c1_bootstrap=a.c1_bootstrap).run()


if __name__ == '__main__':
    main()
