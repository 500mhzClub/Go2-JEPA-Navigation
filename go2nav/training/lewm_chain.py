"""Train the C5/C6 chain from the LeWM corpus, resumably, and write the model set that missions and cohorts accept.

    python -m go2nav.training.lewm_chain OUT --corpus ROOT --recordings-root ROOT [--recordings-root ROOT ...]
                                         [--config configs/lewm/go2.yaml] [--set KEY=VALUE ...] [--seed-offset K]
                                         [--base-readout contexts|corpus|none]

Stages, each in its own process with its log in OUT/logs:
- the corpus window indices (`go2nav.data.lewm_corpus index`; shared by every chain on the corpus);
- the world model (`go2nav.training.lewm train`, OUT/world): it checkpoints itself, so an interrupted stage resumes
  where it stopped (this driver never clears it);
- the latent cache of the decoder contexts (`go2nav.training.lewm_decoder cache`, OUT/cache);
- C5's base readout (`go2nav.training.lewm_readout`, OUT/readout_<source>): by default from T7's own training
  contexts (`contexts`, like for like with C3's decoder, grown from T5b; adopted 18 October, docs/lewm/STATUS.md);
  `corpus` pretrains on the corpus instead, `none` fits from scratch as before;
- three C5/C6 fits with T7's seeds grown from it (OUT/fit_<source>_s<seed>; from scratch OUT/fit_s<seed>), then T7's
  rule: the median of C5's scores (OUT/choice.json);
- OUT/models.yaml: lewm_world_model and lewm_c5_decoder_and_c6, for `go2nav.eval.mission --models` and the cohort
  option `models`.
A stage whose product exists is skipped. --seed-offset K adds K to the world model's seed and the fits' seeds (0: the
configured seeds).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from go2nav.training.lewm import EVAL_STEPS, load_config
from go2nav.training.lewm_decoder import SEEDS

REPO = Path(__file__).resolve().parents[2]


def stage(out, name, module, args, product):
    product = Path(product)
    if product.exists():
        print(f'[skip] {name}', flush=True)
        return product
    print(f'[run] {name}', flush=True)
    log = out/'logs'/f'{name}.log'
    with open(log, 'a') as stream:
        code = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=REPO, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    if code or not product.exists():
        raise RuntimeError(f'stage {name} failed (exit {code}); see {log}')
    return product


def run(out, *, corpus, recordings, config_path, overrides=(), seed_offset=0, base_readout='contexts'):
    out = Path(out).resolve()
    (out/'logs').mkdir(parents=True, exist_ok=True)
    config = load_config(config_path, overrides)
    world_seed = config['seed']+seed_offset
    history = config['model']['history']
    for length in sorted({config['data']['length'], history+EVAL_STEPS}):
        stage(out, f'index_{length}', 'go2nav.data.lewm_corpus', ['index', corpus, '--length', length],
              Path(corpus)/f'index_{length}.npz')
    sets = [a for o in overrides for a in ('--set', o)]
    world = stage(out, 'world', 'go2nav.training.lewm',
                  ['train', '--corpus', corpus, '--out', out/'world', '--config', config_path, *sets, '--set', f'seed={world_seed}'],
                  out/'world'/'lewm.pt')
    roots = [a for r in recordings for a in ('--recordings-root', r)]
    stage(out, 'cache', 'go2nav.training.lewm_decoder', ['cache', '--world-model', world, *roots, '--out', out/'cache'],
          out/'cache'/'rollout.done')
    base, prefix = [], 'fit'
    if base_readout != 'none':
        source = ['--corpus', corpus] if base_readout == 'corpus' else ['--cache', out/'cache']
        readout = stage(out, f'readout_{base_readout}', 'go2nav.training.lewm_readout',
                        [base_readout, '--world-model', world, *source, '--out', out/f'readout_{base_readout}'],
                        out/f'readout_{base_readout}'/'.done').parent/'readout.pt'
        base, prefix = ['--base-readout', readout], f'fit_{base_readout}'
    fits = []
    for original in SEEDS:
        seed = original+seed_offset
        folder = out/f'{prefix}_s{seed}'
        stage(out, f'{prefix}_s{seed}', 'go2nav.training.lewm_decoder',
              ['fit', '--cache', out/'cache', '--out', folder, '--seed', seed, *base], folder/'.done')
        fits.append(folder)
    scored = sorted((json.loads((f/'evaluation.json').read_text())['score_mm'], f) for f in fits)
    chosen = scored[len(scored)//2][1]
    (out/'choice.json').write_text(json.dumps(dict(chosen=str(chosen), scores_mm={str(f): s for s, f in scored},
                                                   probe=json.loads((chosen/'evaluation.json').read_text())['probe']),
                                              indent=1)+'\n')
    models = dict(lewm_world_model=str(world), lewm_c5_decoder_and_c6=str(next(chosen.glob('fit_*.pt'))))
    (out/'models.yaml').write_text(f'# LeWM chain ({Path(config_path).name}, seed offset {seed_offset}, '
                                   f'base readout {base_readout})\n'
                                   + yaml.safe_dump(models, sort_keys=False))
    print(f'chain done: {out/"models.yaml"}', flush=True)
    return models


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--recordings-root', action='append', required=True)
    p.add_argument('--config', type=Path, default=REPO/'configs/lewm/go2.yaml')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--base-readout', choices=('contexts', 'corpus', 'none'), default='contexts')
    a = p.parse_args()
    run(a.out, corpus=a.corpus, recordings=a.recordings_root, config_path=a.config.resolve(), overrides=a.set,
        seed_offset=a.seed_offset, base_readout=a.base_readout)


if __name__ == '__main__':
    main()
