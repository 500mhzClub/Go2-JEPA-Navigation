"""Train H-JEPA (`models.hjepa`) on the LeWM corpus, every level jointly and resumably, and evaluate its gates.

    python -m go2nav.training.hjepa train --corpus ROOT --out DIR [--config configs/hjepa/go2_3level.yaml] [--set K=V]
    python -m go2nav.training.hjepa evaluate CHECKPOINT --corpus ROOT [--out FILE]

The corpus needs window indices of the clip length (257 frames) and of 12 frames (`lewm_corpus index --length N`).

**Configuration** (configs/hjepa/): `level1` holds the LeWM model and objective (span8w's), `upper` one entry per upper
level (stride, context and the loss weights), plus `sigreg`, `data`, `optimiser` and `runtime` as in `training.lewm`.
`--set` overrides a value; the resolved config is saved with the run.

**Training.** The joint objective of `models.hjepa.losses`, AdamW with linear warm-up then cosine decay, one learning
rate, gradient clipping per level, bf16 autocast (SIGReg in float32), the loss compiled (`runtime.compile`). Batches
(`data.hjepa_dataset`: level 1 exactly span8w's windows, the upper levels clips of mission and explore units from the
same mazes) are a pure function of (seed, step) and come from parallel loader workers; SIGReg's projections from a generator keyed by
(seed, step). Every `checkpoint_every` steps the model, optimiser, scheduler, step and RNG states are written
atomically (DIR/checkpoint.pt), and rerunning the same command resumes from it with the same data stream.

**Validation** (every `validate_every` steps, at the end, and `evaluate`):
- level 1, exactly as `training.lewm.validate` on the same fixed windows (gates L1 health and L2 action sensitivity),
  and on the held-out look;
- each upper level, on fixed clips of the validation mazes' mission and explore units: the teacher-forced prediction
  loss, SIGReg, inverse dynamics and state health (z spread, effective rank) (gate H1); and the last transition's
  prediction error with the true macro-action, against a shuffled one (permuted across clips) and persistence (the
  previous state). Gate H2 passes when the true error is at most 0.7 x the shuffled one and below persistence.

`runtime.memory_fraction` (default 0.85) caps this process's share of GPU memory: on the development machine the
training GPU also drives the display, and running it out of memory crashed the desktop session.
`runtime.checkpoint_blocks`: the first n of level 1's ViT blocks recompute their activations in the backward pass
(`models.lewm.ViTEncoder`), which is what fits the joint batch (128 windows of 12 frames and 128 clips of 7) in it.

Outputs: DIR/hjepa.pt (`models.hjepa.save_hjepa`), DIR/lewm.pt (level 1 alone, `models.hjepa.export_level1`: what
C7/C8 and the LeWM readout tools load), config.yaml, metrics.jsonl, evaluation.json, lineage.json.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
import yaml

from go2nav.data.hjepa_dataset import HJEPABatches, clip_crops, upper_layout
from go2nav.data.lewm_dataset import Corpus, keyed_rng
from go2nav.models.hjepa import HJEPA, HJEPAConfig, UpperConfig, export_level1, losses, save_hjepa
from go2nav.models.lewm import SIGReg, normalise_frames
from go2nav.training.common import sha256_file, write_lineage
from go2nav.training.lewm import (EVAL_STEPS, TRANSFER_WINDOWS, VALIDATION_SEED, VALIDATION_WINDOWS, atomic_save,
                                  keyed_seed, load_config, model_config as level1_config, schedule, validate)

REPO = Path(__file__).resolve().parents[2]
UPPER_VALIDATION_CLIPS = 1024
LOSS_WEIGHTS = ('sigreg_weight', 'inverse_weight', 'action_sigreg_weight')
MAX_NONFINITE = 20
MEMORY_FRACTION = .85


def model_config(config):
    upper = tuple(UpperConfig(**{k: v for k, v in u.items() if k not in LOSS_WEIGHTS}) for u in config['upper'])
    level1 = level1_config(dict(model=config['level1']['model']))
    level1 = replace(level1, **{k: tuple(v) for k, v in asdict(level1).items() if isinstance(v, list)})
    return HJEPAConfig(level1=level1, upper=upper)


def weights(config):
    """The keyword arguments of `models.hjepa.losses` for the level-1 and upper-level terms."""
    objective = config['level1']['objective']
    level1 = dict(rollout_steps=objective['rollout_steps'], lam=objective['sigreg_weight'],
                  inverse_weight=objective.get('inverse_weight', 0.))
    upper = [dict(sigreg=u['sigreg_weight'], inverse=u['inverse_weight'], action_sigreg=u['action_sigreg_weight'])
             for u in config['upper']]
    return level1, upper


def to_device(batch, device):
    """uint8 frames to normalised (B, T, 3, H, W) floats, everything on the device."""
    return dict(frames=[normalise_frames(f.to(device, non_blocking=True)) for f in batch['frames']],
                actions=[a.to(device, non_blocking=True) for a in batch['actions']])


def health(states):
    """Per-dimension spread and effective rank of states (n, D)."""
    z = states.double()
    eig = torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=0)
    p = eig/eig.sum()
    std = z.std(0)
    return dict(z_std_min=float(std.min()), z_std_mean=float(std.mean()),
                effective_rank=float(torch.exp(-(p[p > 0]*p[p > 0].log()).sum())))


@torch.no_grad()
def validate_upper(model, corpus, indices, *, device, seed=VALIDATION_SEED, batch=128):
    """Each upper level's metrics on fixed clips (see the module docstring)."""
    was_training = model.training
    model.eval()
    sigreg = SIGReg().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    rng = keyed_rng(seed, 'hjepa_validation')
    layout = upper_layout(model.config)
    sums = [defaultdict(float) for _ in model.upper]
    last = [[] for _ in model.upper]
    count = 0
    for start in range(0, len(indices), batch):
        frames, actions = clip_crops(corpus, indices[start:start+batch], rng, layout)
        b = len(frames[0])
        order = torch.randperm(b, generator=shuffle).to(device)
        for k, level in enumerate(model.upper):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                z1 = model.level1.encode(normalise_frames(frames[k].to(device)))
                states = model.states(z1, k+2).float()
                macros = model.macros(actions[k].to(device), k+2).float()
                pred = level.predict(states[:, :-1], macros).float()
                shuffled = macros.clone()
                shuffled[:, -1] = macros[order, -1]
                pred_shuffled = level.predict(states[:, :-1], shuffled).float()[:, -1]
                inverse = level.infer(states).float()
            target = states[:, -1]
            s = sums[k]
            s['pred'] += float((pred-states[:, 1:]).pow(2).mean())*b
            s['sigreg'] += float(sigreg(states.transpose(0, 1), generator))*b
            s['inverse'] += float((inverse-macros).pow(2).mean())*b
            s['last_true'] += float((pred[:, -1]-target).pow(2).mean())*b
            s['last_shuffled'] += float((pred_shuffled-target).pow(2).mean())*b
            s['last_persistence'] += float((states[:, -2]-target).pow(2).mean())*b
            last[k].append(target.cpu())
        count += b
    out = {}
    for k, s in enumerate(sums):
        row = {key: value/count for key, value in s.items()} | health(torch.cat(last[k]))
        row['h2_action_sensitivity'] = bool(row['last_true'] <= .7*row['last_shuffled']
                                            and row['last_true'] < row['last_persistence'])
        out[f'l{k+2}'] = row
    model.train(was_training)
    return out


def validation_sets(corpus_root, config, upper_kinds=('mission', 'explore')):
    """Level 1's validation and held-out-look windows, and the upper levels' validation clips, with fixed indices."""
    history = config.level1.history
    validation = Corpus(corpus_root, length=history+EVAL_STEPS, roles=('lewm_val',), kinds=('explore', 'mission', 'free'))
    transfer = Corpus(corpus_root, length=history+EVAL_STEPS, roles=('lewm_val',), kinds=('transfer',))
    clips = Corpus(corpus_root, length=config.clip_length, roles=('lewm_val',), kinds=tuple(upper_kinds))
    return dict(validation=(validation, validation.fixed(VALIDATION_WINDOWS, VALIDATION_SEED)),
                transfer=(transfer, transfer.fixed(TRANSFER_WINDOWS, VALIDATION_SEED)),
                clips=(clips, clips.fixed(UPPER_VALIDATION_CLIPS, VALIDATION_SEED)))


def evaluate_all(model, sets, device):
    return dict(validation=validate(model.level1, *sets['validation'], rollout_steps=EVAL_STEPS, device=device),
                transfer=validate(model.level1, *sets['transfer'], rollout_steps=EVAL_STEPS, device=device),
                upper=validate_upper(model, *sets['clips'], device=device))


def train(config, corpus_root, out, *, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = out/'config.yaml'
    if resolved.exists():
        if yaml.safe_load(resolved.read_text()) != config:
            raise ValueError(f'{out} was started with another config; use a new folder')
    else:
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed, data, optim, runtime = (config[k] for k in ('seed', 'data', 'optimiser', 'runtime'))
    if device.type == 'cuda':  # the R9700 also drives the desktop: an overrun must fail here, not starve the compositor
        torch.cuda.set_per_process_memory_fraction(runtime.get('memory_fraction', MEMORY_FRACTION), device)
    mconfig = model_config(config)
    level1_weights, upper_weights = weights(config)
    windows = Corpus(corpus_root, length=mconfig.level1.history+level1_weights['rollout_steps'], roles=('lewm_train',))
    clips = Corpus(corpus_root, length=mconfig.clip_length, roles=('lewm_train',), kinds=tuple(data['upper_kinds']))
    sets = validation_sets(corpus_root, mconfig, data['upper_kinds'])

    model = HJEPA.seeded(seed, mconfig).to(device).train()
    model.level1.encoder.checkpoint_blocks = runtime.get('checkpoint_blocks', 0)
    sigreg = SIGReg(knots=config['sigreg'].get('knots', 17), num_proj=config['sigreg']['projections']).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=optim['lr'], weight_decay=optim['weight_decay'], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(optim['warmup_steps'], optim['steps']))
    first, nonfinite, skipped = 0, 0, []
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state(state['rng_cuda'], device)
        first, skipped = state['step']+1, state['skipped']
        rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line.strip()] \
            if (out/'metrics.jsonl').exists() else []
        (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step'] <= state['step']))
        print(json.dumps(dict(resumed_at=first)), flush=True)
    if first >= optim['steps'] and (out/'hjepa.pt').exists():
        return out/'hjepa.pt'

    loss_fn = torch.compile(losses, mode=runtime['compile']) if runtime.get('compile') else losses
    loader = torch.utils.data.DataLoader(
        HJEPABatches(windows, clips, mconfig, seed=seed, first=first, last=optim['steps'],
                     mazes_per_batch=data['mazes_per_batch'], windows_per_maze=data['windows_per_maze'],
                     clips_per_maze=data['clips_per_maze']),
        batch_size=None, shuffle=False, num_workers=runtime['workers'], pin_memory=True, prefetch_factor=runtime['prefetch'],
        persistent_workers=False, multiprocessing_context='forkserver')
    generator = torch.Generator(device=device)
    groups = model.parameter_groups()
    log, started, window = open(out/'metrics.jsonl', 'a'), time.time(), defaultdict(float)
    step = first-1
    for step, batch in enumerate(loader, start=first):
        batch = to_device(batch, device)
        generator.manual_seed(keyed_seed(seed, 'sigreg', step))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            terms = loss_fn(model, batch, level1=level1_weights, upper=upper_weights, sigreg=sigreg, generator=generator)
        optimiser.zero_grad(set_to_none=True)
        if not torch.isfinite(terms['loss']):
            nonfinite += 1
            skipped.append(step)
            if nonfinite >= MAX_NONFINITE:
                raise RuntimeError(f'{MAX_NONFINITE} non-finite losses in a row (last at step {step})')
            scheduler.step()
            continue
        nonfinite = 0
        terms['loss'].backward()
        norms = [torch.nn.utils.clip_grad_norm_(g, optim['grad_clip']) for g in groups]
        optimiser.step()
        scheduler.step()
        for key, value in terms.items():
            window[key] += float(value.detach())
        for level, norm in enumerate(norms, 1):
            window[f'l{level}/grad_norm'] += float(norm)
        window['n'] += 1
        last = step == optim['steps']-1
        if (step+1) % runtime['log_every'] == 0 or last:
            n = window.pop('n')
            row = dict(step=step, lr=scheduler.get_last_lr()[0], s_per_step=round((time.time()-started)/n, 3),
                       **{k: v/n for k, v in window.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(row), flush=True)
            window, started = defaultdict(float), time.time()
        if (step+1) % runtime['validate_every'] == 0 or last:
            row = dict(step=step, **evaluate_all(model, sets, device))
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(dict(step=step, l2_gate=row['validation']['l2_action_sensitivity'],
                                  l1_rank=round(row['validation']['effective_rank'], 1),
                                  **{f'{k}_h2': v['h2_action_sensitivity'] for k, v in row['upper'].items()},
                                  **{f'{k}_rank': round(v['effective_rank'], 1) for k, v in row['upper'].items()},
                                  **{f'{k}_pred': round(v['pred'], 4) for k, v in row['upper'].items()})), flush=True)
        if (step+1) % runtime['checkpoint_every'] == 0 or last:
            atomic_save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                             step=step, skipped=skipped, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device)),
                        checkpoint)
        if (step+1) % runtime['milestone_every'] == 0:
            save_hjepa(out/f'hjepa_step{step+1}.pt', model, extra=dict(step=step+1))
    log.close()
    if step != optim['steps']-1:
        raise RuntimeError(f'training stopped at step {step} of {optim["steps"]}')
    final = out/'hjepa.pt'
    save_hjepa(final, model, extra=dict(step=optim['steps'], skipped=skipped, training_config=config))
    export_level1(model, out/'lewm.pt', extra=dict(step=optim['steps'], training_config=config))
    evaluation = evaluate_all(model, sets, device)
    (out/'evaluation.json').write_text(json.dumps(evaluation, indent=1)+'\n')
    indices = {name: sha256_file(Path(corpus_root)/f'index_{c.length}.npz') for name, c in (('windows', windows), ('clips', clips))}
    write_lineage(out, component='hjepa', seed=seed, config=config, parents=[],
                  data=dict(corpus=str(corpus_root), index_sha256=indices, windows=len(windows), clips=len(clips),
                            mazes=int(len(windows.mazes)), skipped_steps=skipped),
                  checkpoint=final, metrics=evaluation)
    return final


def evaluate(checkpoint, corpus_root, *, device='cuda:0'):
    from go2nav.models.hjepa import load_hjepa
    device = torch.device(device)
    model = load_hjepa(checkpoint, device)
    return evaluate_all(model, validation_sets(corpus_root, model.config), device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--config', type=Path, default=REPO/'configs/hjepa/go2_3level.yaml')
    t.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    e = sub.add_parser('evaluate')
    e.add_argument('checkpoint', type=Path)
    e.add_argument('--corpus', required=True, type=Path)
    e.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.command == 'train':
        print(train(load_config(a.config, a.set), a.corpus, a.out))
    else:
        text = json.dumps(evaluate(a.checkpoint, a.corpus), indent=1)
        if a.out:
            a.out.write_text(text+'\n')
        print(text)


if __name__ == '__main__':
    main()
