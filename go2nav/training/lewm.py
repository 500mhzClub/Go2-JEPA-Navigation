"""Train the LeWorldModel (C5/C6's world model) on the LeWM corpus, resumably, and evaluate its training gates.

    python -m go2nav.training.lewm train --corpus ROOT --out DIR [--config configs/lewm/go2.yaml] [--set KEY=VALUE ...]
    python -m go2nav.training.lewm evaluate CHECKPOINT --corpus ROOT [--out FILE]

**Configs** (configs/lewm/): `go2.yaml` (the Go2 configuration: 4-frame history, 8-step rollout loss, 16 mazes x 8
windows per batch), `paper.yaml` (the paper's objective and i.i.d. batches, for comparison) and the
inverse-dynamics variants `go2_inverse.yaml` and `go2_idm100.yaml` (`objective.inverse_weight` 0.1 and 100,
`model.inverse_dynamics`), with longer spans `go2_idm_span8.yaml`, `go2_idm_spans.yaml`, `go2_idm_span8w.yaml`
(`model.inverse_spans`, `inverse_span_weights`), and `go2_idm_span8m.yaml`, which also decodes the body-frame motion
from the corpus poses (`model.motion_dynamics`, `objective.motion_weight`). `--set` overrides a
value (`--set objective.sigreg_weight=0.03`); the resolved config is saved with the run.

**Training.** AdamW with linear warm-up then cosine decay, gradient clipping, bf16 autocast (SIGReg in float32), the
whole loss compiled (`torch.compile`, autotuned GEMMs; 453 ms per step on the R9700 at batch 128 x 12 frames against
756 eager). Batches come from `go2nav.data.lewm_dataset` in parallel loader workers, as a pure function of
(seed, step); SIGReg's projections come from a generator seeded by (seed, step). The final step is fixed in advance
(no early stopping or checkpoint selection on a metric).

**Resume.** Every `checkpoint_every` steps the model, optimiser, scheduler, step and RNG states are written atomically
(DIR/checkpoint.pt; milestones every `milestone_every` keep the model only). Rerunning the same command continues from
the last checkpoint with the same data stream, and metrics.jsonl is cut back to it. A non-finite loss skips its update
(recorded); 20 in a row stop the run.

**Validation** (every `validate_every` steps, at the end, and `evaluate`), on fixed windows of the validation mazes
(explore, mission and free units) and, separately, the transfer units (the held-out look):
- teacher-forced prediction loss and SIGReg;
- rollout error at each of the 8 steps with the true actions, and the gates' baselines: shuffled future actions
  (permuted across the batch), zero future commands, and persistence (the last history frame's z);
- z health: per-dimension standard deviation (minimum and mean) and effective rank.
Gate L2 (action sensitivity) passes when, from step 3 on, the true-action error is at most 0.7 x the
shuffled-action error and below persistence.

Outputs: DIR/lewm.pt (`models.lewm.save_world_model`), config.yaml, metrics.jsonl, evaluation.json, lineage.json.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from go2nav.data.lewm_dataset import Corpus, WindowBatches
from go2nav.models.lewm import LeWMConfig, LeWorldModel, SIGReg, losses, normalise_frames, save_world_model
from go2nav.training.common import sha256_file, write_lineage

REPO = Path(__file__).resolve().parents[2]
VALIDATION_WINDOWS, TRANSFER_WINDOWS, VALIDATION_SEED = 2048, 1024, 2026101702
EVAL_STEPS = 8  # validation always rolls out the forecaster's 8 steps
MAX_NONFINITE = 20


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, value = item.partition('=')
        node = config
        *parents, leaf = key.split('.')
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return config


def model_config(config):
    m = config['model']
    return LeWMConfig.named(m.get('size', 'tiny'), history=m['history'],
                            **{k: tuple(v) if k == 'image_hw' else v for k, v in m.items() if k not in ('size', 'history')})


def keyed_seed(*keys):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'big') % (2**63-1)


def schedule(warmup, total):
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    return factor


def atomic_save(value, path):
    tmp = Path(str(path)+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def zero_command_actions(actions, corpus):
    """Action blocks of zero commands, normalised like the corpus's."""
    zero = torch.as_tensor(-corpus.mean/corpus.std, dtype=actions.dtype, device=actions.device)
    return zero.repeat(actions.shape[-1]//2).expand_as(actions)


@torch.no_grad()
def validate(model, corpus, indices, *, rollout_steps, device, seed=VALIDATION_SEED, batch=128):
    """Validation metrics on the windows `indices` of `corpus` (see the module docstring)."""
    was_training = model.training
    model.eval()
    n, k = model.config.history, rollout_steps
    sums, count, zs = defaultdict(float), 0, []
    per_step = {name: np.zeros(k) for name in ('rollout', 'shuffled', 'zero', 'persistence')}
    sigreg = SIGReg().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    for start in range(0, len(indices), batch):
        frames, actions, _ = corpus.batch(indices[start:start+batch])
        frames, actions = frames.to(device, non_blocking=True), actions.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            z = model.encode(normalise_frames(frames)).float()
            pred = model.predict(z[:, :n], actions[:, :n]).float()
            target = z[:, n:n+k]
            variants = dict(rollout=actions)
            permuted = actions.clone()
            permuted[:, n-1:] = actions[torch.randperm(len(actions), generator=shuffle).to(device), n-1:]
            variants['shuffled'] = permuted
            zeroed = actions.clone()
            zeroed[:, n-1:] = zero_command_actions(actions[:, n-1:], corpus)
            variants['zero'] = zeroed
            for name, acts in variants.items():
                rolled = model.rollout(z[:, :n], acts, k).float()
                per_step[name] += (rolled-target).pow(2).mean(-1).sum(0).cpu().numpy()
        per_step['persistence'] += (z[:, n-1:n]-target).pow(2).mean(-1).sum(0).cpu().numpy()
        b = len(frames)
        sums['pred'] += float((pred-z[:, 1:n+1]).pow(2).mean())*b
        sums['sigreg'] += float(sigreg(z.transpose(0, 1), generator))*b
        zs.append(z[:, n-1].cpu())
        count += b
    z = torch.cat(zs).double()
    eig = torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=0)
    p = eig/eig.sum()
    std = z.std(0)
    out = dict(windows=count, pred=sums['pred']/count, sigreg=sums['sigreg']/count,
               z_std_min=float(std.min()), z_std_mean=float(std.mean()),
               effective_rank=float(torch.exp(-(p[p > 0]*p[p > 0].log()).sum())),
               **{f'{name}_by_step': (v/count).tolist() for name, v in per_step.items()})
    true, shuffled, persistence = (np.asarray(out[f'{x}_by_step']) for x in ('rollout', 'shuffled', 'persistence'))
    out['l2_action_sensitivity'] = bool(np.all(true[2:] <= .7*shuffled[2:]) and np.all(true[2:] < persistence[2:]))
    model.train(was_training)
    return out


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
    seed, data, objective, optim, runtime = (config[k] for k in ('seed', 'data', 'objective', 'optimiser', 'runtime'))
    mconfig = model_config(config)
    length = mconfig.history+objective['rollout_steps']
    if data['length'] != length:
        raise ValueError(f"data.length must be history + rollout_steps = {length}")
    motion_weight = objective.get('motion_weight', 0.)
    if bool(motion_weight) != mconfig.motion_dynamics:
        raise ValueError('objective.motion_weight > 0 and model.motion_dynamics go together')
    corpus = Corpus(corpus_root, length=length, roles=tuple(data.get('train_roles', ('lewm_train',))), poses=bool(motion_weight))
    val_length = mconfig.history+EVAL_STEPS
    validation = Corpus(corpus_root, length=val_length, roles=('lewm_val',), kinds=('explore', 'mission', 'free'))
    transfer = Corpus(corpus_root, length=val_length, roles=('lewm_val',), kinds=('transfer',))
    val_indices, transfer_indices = validation.fixed(VALIDATION_WINDOWS, VALIDATION_SEED), transfer.fixed(TRANSFER_WINDOWS, VALIDATION_SEED)

    inverse_weight = objective.get('inverse_weight', 0.)
    if bool(inverse_weight) != mconfig.inverse_dynamics:
        raise ValueError('objective.inverse_weight > 0 and model.inverse_dynamics go together')
    model = LeWorldModel.seeded(seed, mconfig).to(device).train()
    sigreg = SIGReg(knots=objective.get('sigreg_knots', 17), num_proj=objective['sigreg_projections']).to(device)
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
    if first >= optim['steps'] and (out/'lewm.pt').exists():
        return out/'lewm.pt'

    loss_fn = torch.compile(losses, mode=runtime['compile']) if runtime.get('compile') else losses
    action_blind = objective.get('action_blind', False)
    loader = torch.utils.data.DataLoader(
        WindowBatches(corpus, seed=seed, first=first, last=optim['steps'], mazes_per_batch=data['mazes_per_batch'],
                      windows_per_maze=data['windows_per_maze'], batch_size=data['batch_size']),
        batch_size=None, shuffle=False, num_workers=runtime['workers'], pin_memory=True, prefetch_factor=runtime['prefetch'],
        persistent_workers=False, multiprocessing_context='forkserver')
    generator = torch.Generator(device=device)
    log, started, window = open(out/'metrics.jsonl', 'a'), time.time(), defaultdict(float)
    step = first-1
    for step, (frames, actions, _, *poses) in enumerate(loader, start=first):
        frames, actions = frames.to(device, non_blocking=True), actions.to(device, non_blocking=True)
        poses = poses[0].to(device, non_blocking=True) if poses else None
        if action_blind:
            actions = torch.zeros_like(actions)
        generator.manual_seed(keyed_seed(seed, 'sigreg', step))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            terms = loss_fn(model, normalise_frames(frames), actions, rollout_steps=objective['rollout_steps'],
                            sigreg=sigreg, lam=objective['sigreg_weight'], inverse_weight=inverse_weight, generator=generator,
                            poses=poses, motion_weight=motion_weight)
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
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), optim['grad_clip'])
        optimiser.step()
        scheduler.step()
        for key, value in terms.items():
            window[key] += float(value.detach())
        window['grad_norm'] += float(grad)
        window['n'] += 1
        last = step == optim['steps']-1
        if (step+1) % runtime['log_every'] == 0 or last:
            n = window.pop('n')
            row = dict(step=step, lr=scheduler.get_last_lr()[0], images_per_s=round(n*frames.shape[0]*frames.shape[1]/(time.time()-started), 1),
                       **{k: v/n for k, v in window.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(row), flush=True)
            window, started = defaultdict(float), time.time()
        if (step+1) % runtime['validate_every'] == 0 or last:
            row = dict(step=step, validation=validate(model, validation, val_indices, rollout_steps=EVAL_STEPS, device=device),
                       transfer=validate(model, transfer, transfer_indices, rollout_steps=EVAL_STEPS, device=device))
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(dict(step=step, l2=row['validation']['l2_action_sensitivity'],
                                  rollout=np.round(row['validation']['rollout_by_step'], 4).tolist(),
                                  shuffled=np.round(row['validation']['shuffled_by_step'], 4).tolist(),
                                  effective_rank=round(row['validation']['effective_rank'], 1))), flush=True)
        if (step+1) % runtime['checkpoint_every'] == 0 or last:
            atomic_save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                             step=step, skipped=skipped, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device)),
                        checkpoint)
        if (step+1) % runtime['milestone_every'] == 0:
            save_world_model(out/f'lewm_step{step+1}.pt', model, extra=dict(step=step+1))
    log.close()
    if step != optim['steps']-1:
        raise RuntimeError(f'training stopped at step {step} of {optim["steps"]}')
    final = out/'lewm.pt'
    save_world_model(final, model, extra=dict(step=optim['steps'], skipped=skipped, training_config=config))
    evaluation = dict(validation=validate(model, validation, val_indices, rollout_steps=EVAL_STEPS, device=device),
                      transfer=validate(model, transfer, transfer_indices, rollout_steps=EVAL_STEPS, device=device))
    (out/'evaluation.json').write_text(json.dumps(evaluation, indent=1)+'\n')
    index = Path(corpus_root)/f'index_{length}.npz'
    write_lineage(out, component='lewm_world_model', seed=seed, config=config, parents=[],
                  data=dict(corpus=str(corpus_root), index=index.name, index_sha256=sha256_file(index), windows=len(corpus),
                            mazes=int(len(corpus.mazes)), skipped_steps=skipped),
                  checkpoint=final, metrics=evaluation)
    return final


def evaluate(checkpoint, corpus_root, *, rollout_steps=EVAL_STEPS, device='cuda:0'):
    from go2nav.models.lewm import load_world_model
    device = torch.device(device)
    model = load_world_model(checkpoint, device)
    length = model.config.history+rollout_steps
    validation = Corpus(corpus_root, length=length, roles=('lewm_val',), kinds=('explore', 'mission', 'free'))
    transfer = Corpus(corpus_root, length=length, roles=('lewm_val',), kinds=('transfer',))
    return dict(validation=validate(model, validation, validation.fixed(VALIDATION_WINDOWS, VALIDATION_SEED),
                                    rollout_steps=rollout_steps, device=device),
                transfer=validate(model, transfer, transfer.fixed(TRANSFER_WINDOWS, VALIDATION_SEED),
                                  rollout_steps=rollout_steps, device=device))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--config', type=Path, default=REPO/'configs/lewm/go2.yaml')
    t.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    e = sub.add_parser('evaluate')
    e.add_argument('checkpoint', type=Path)
    e.add_argument('--corpus', required=True, type=Path)
    e.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.command == 'train':
        print(train(load_config(a.config, a.set), a.corpus, a.out))
    else:
        result = evaluate(a.checkpoint, a.corpus)
        text = json.dumps(result, indent=1)
        if a.out:
            a.out.write_text(text+'\n')
        print(text)


if __name__ == '__main__':
    main()
