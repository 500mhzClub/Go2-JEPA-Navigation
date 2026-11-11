"""Training the ring world model (`models.ring_jepa`; docs/dynamic/PLAN.md P21) on P20's corpus.

    python -m go2nav.training.ring_jepa train --corpus CORPUS --out DIR --config configs/ring/r0.yaml [--set key=value ...]

CORPUS is the ring corpus (`data.ring_corpus`, packed). Both splits are held on the GPU (float16 ranges: 6 GB), and a
batch is drawn there: `batch` windows uniformly over every frame that starts `history + rollout` frames inside one
episode.

**Loss** (per window of H history frames and K after them):
- `pred`: the next-latent error of the teacher-forced pass over the history (position i predicts frame i+1), plus
  that of the K-step rollout from the history on the model's own predictions (`RingJEPA.rollout`), averaged over steps;
- `sigreg`: SIGReg of the tokens, each sector of each of `sigreg_frames` frames drawn at random taken as its own
  distribution over the batch (every token as a sample needs a 4-GB projection tensor), times `sigreg_weight`;
- `decode`: the decoded log range's absolute error on beams that returned, on every frame's encoded latent and on every
  rolled-out one, times `decoder_weight`.
AdamW with linear warm-up and cosine decay, gradient clipping, bf16 autocast (SIGReg in float32), the predictor's
blocks recomputed in the backward pass (`checkpoint_predictor`; a 16-step rollout at batch 128 keeps about 15 GB of
activations otherwise). The process may use 80% of the GPU's memory: the card drives the desktop. Resumable: the
checkpoint (`DIR/checkpoint.pt`, every `checkpoint_every` steps) holds the model, optimiser, scheduler and step, and a
step's batch and SIGReg projections are drawn from generators seeded by the step.

**Validation** (`validate`, every `validate_every` steps, `DIR/validation.jsonl`): fixed windows of the val split
(layouts the training split never saw), rolled out `validation_steps` frames. At steps 1, 8, 16, 32 (those within
it): the latent error, the decoded scan's error on every beam, and on the beams that changed by more than
`CHANGED` in log range since the last history frame (what moved: the robot's own motion and the walkers), beside the
same error of persistence (the last history frame's scan). The model's final weights: `DIR/model.pt`.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from go2nav.models.lewm import SIGReg
from go2nav.models.ring_jepa import RingConfig, RingJEPA, command_input, scan_input

CHANGED = .02
MEMORY_FRACTION = .8
REPORT_STEPS = (1, 8, 16, 32, 64)
DEFAULTS = dict(seed=2026111102, model={}, data=dict(batch=128, rollout=16),
                objective=dict(sigreg_weight=.08, sigreg_projections=1024, sigreg_frames=2, decoder_weight=5.),
                optimiser=dict(lr=2e-4, weight_decay=1e-3, warmup=1000, steps=40000, grad_clip=1.),
                runtime=dict(log_every=100, validate_every=2000, checkpoint_every=1000, validation_windows=4096,
                             validation_steps=32))


def merged(base, extra):
    out = dict(base)
    for key, value in extra.items():
        out[key] = merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def load_config(path, overrides=()):
    config = merged(DEFAULTS, yaml.safe_load(Path(path).read_text()) if path else {})
    for item in overrides:
        key, _, value = item.partition('=')
        node = config
        *parents, last = key.split('.')
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return config


class Split:
    """One packed split on the device: ranges (N, 720) float16, commands (N, 3), and the window starts that keep
    `frames` frames inside one episode."""

    def __init__(self, corpus, split, frames, device):
        packed = Path(corpus)/'packed'
        self.ranges = torch.from_numpy(np.load(packed/f'{split}_ranges.npy', mmap_mode='r')[:]).to(device)
        self.requested = torch.from_numpy(np.load(packed/f'{split}_requested.npy')).to(device)
        starts = np.load(packed/f'{split}_starts.npy')
        self.episode_starts = starts
        self.windows = self.starts_for(frames)
        self.device = device

    def starts_for(self, frames):
        s = self.episode_starts
        return np.concatenate([np.arange(a, b-frames+1) for a, b in zip(s[:-1], s[1:]) if b-a >= frames]).astype(np.int64)

    def batch(self, starts, frames):
        """Inputs (B, F, 720, 2) and commands (B, F, 3) of windows starting at `starts` (B,) int64."""
        index = torch.as_tensor(starts, device=self.device)[:, None]+torch.arange(frames, device=self.device)
        return scan_input(self.ranges[index]), command_input(self.requested[index])


def losses(model, x, a, rollout, *, sigreg, generator, weights):
    """The training loss and its terms for inputs (B, H+K, 720, 2) and commands (B, H+K, 3)."""
    h = model.config.history
    b, f = x.shape[:2]
    z = model.encode(x.flatten(0, 1)).unflatten(0, (b, f))                                  # (B, F, S, D)
    teacher = model.predict(z[:, :h], a[:, :h])
    pred_teacher = (teacher-z[:, 1:h+1]).pow(2).mean()
    rolled = model.rollout(z[:, :h], a[:, :h+rollout-1])                                    # (B, K, S, D)
    pred_roll = (rolled-z[:, h:h+rollout]).pow(2).mean()
    frames = torch.randperm(f, device=z.device, generator=generator)[:weights['sigreg_frames']]
    with torch.autocast('cuda', enabled=False):
        sig = sigreg(z[:, frames].float().permute(1, 2, 0, 3).flatten(0, 1), generator)     # (frames x S, B, D)
    valid = x[..., 1] > 0
    target = x[..., 0]
    decoded = model.decode(z)
    decode_enc = ((decoded.float()-target).abs()*valid).sum()/valid.sum().clamp(min=1)
    decoded_roll = model.decode(rolled)
    v = valid[:, h:h+rollout]
    decode_roll = ((decoded_roll.float()-target[:, h:h+rollout]).abs()*v).sum()/v.sum().clamp(min=1)
    total = pred_teacher+pred_roll+weights['sigreg_weight']*sig+weights['decoder_weight']*(decode_enc+decode_roll)
    return total, dict(pred_teacher=pred_teacher, pred_roll=pred_roll, sigreg=sig, decode_enc=decode_enc,
                       decode_roll=decode_roll)


@torch.no_grad()
def validate(model, split, starts, steps, batch=256):
    """See the module docstring. Returns a dict of the metrics at each reported step."""
    model.eval()
    h = model.config.history
    frames = h+steps
    sums, count = {}, 0
    for i in range(0, len(starts), batch):
        x, a = split.batch(starts[i:i+batch], frames)
        b = x.shape[0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model.encode(x.flatten(0, 1)).unflatten(0, (b, frames))
            rolled = model.rollout(z[:, :h], a[:, :frames-1])
            decoded = model.decode(rolled).float()
        target, valid = x[:, h:, :, 0], x[:, h:, :, 1] > 0
        last = x[:, h-1:h, :, 0]
        changed = valid & ((target-last).abs() > CHANGED)
        for k in REPORT_STEPS:
            if k > steps:
                continue
            j = k-1
            err = (decoded[:, j]-target[:, j]).abs()
            rows = dict(latent=float((rolled[:, j].float()-z[:, h+j].float()).pow(2).mean())*b,
                        scan=float((err*valid[:, j]).sum()/valid[:, j].sum().clamp(min=1))*b,
                        changed=float((err*changed[:, j]).sum()/changed[:, j].sum().clamp(min=1))*b,
                        changed_persistence=float(((last[:, 0]-target[:, j]).abs()*changed[:, j]).sum()
                                                  / changed[:, j].sum().clamp(min=1))*b,
                        changed_share=float(changed[:, j].float().mean())*b)
            for name, value in rows.items():
                sums[f'{name}@{k}'] = sums.get(f'{name}@{k}', 0.)+value
        count += b
    model.train()
    return {key: round(value/count, 5) for key, value in sums.items()}


def schedule(warmup, total):
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    return factor


def train(corpus, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)   # the card also drives the desktop
    torch.manual_seed(config['seed'])
    run, optim, objective = config['runtime'], config['optimiser'], config['objective']
    model_config = RingConfig(**config['model'])
    model = RingJEPA(model_config).to(device)
    model.predictor.checkpoint = run.get('checkpoint_predictor', True)    # a 16-step rollout keeps 15 GB otherwise
    rollout, batch = config['data']['rollout'], config['data']['batch']
    h = model_config.history
    train_split = Split(corpus, 'train', h+rollout, device)
    val_split = Split(corpus, 'val', h+run['validation_steps'], device)
    val_starts = np.random.default_rng(config['seed']).choice(val_split.windows, run['validation_windows'], replace=False)
    sigreg = SIGReg(num_proj=objective['sigreg_projections']).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=optim['lr'], weight_decay=optim['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(optim['warmup'], optim['steps']))
    step = 0
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
    log = (out/'train.jsonl').open('a')
    vlog = (out/'validation.jsonl').open('a')
    windows = torch.as_tensor(train_split.windows, device=device)
    generator = torch.Generator(device=device)
    sums, started, n = {}, time.time(), 0
    model.train()
    while step < optim['steps']:
        generator.manual_seed(config['seed']*1_000_003+step)
        pick = torch.randint(len(windows), (batch,), device=device, generator=generator)
        x, a = train_split.batch(windows[pick], h+rollout)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total, terms = losses(model, x, a, rollout, sigreg=sigreg, generator=generator, weights=objective)
        if not torch.isfinite(total):
            raise FloatingPointError(f'loss {float(total)} at step {step}')
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim['grad_clip'])
        optimiser.step()
        scheduler.step()
        step += 1
        n += 1
        for key, value in terms.items():
            sums[key] = sums.get(key, 0.)+float(value)
        sums['grad_norm'] = sums.get('grad_norm', 0.)+float(norm)
        if step % run['log_every'] == 0:
            row = dict(step=step, lr=scheduler.get_last_lr()[0], s_per_step=round((time.time()-started)/n, 4),
                       **{k: round(v/n, 5) for k, v in sums.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            sums, started, n = {}, time.time(), 0
        if step % run['validate_every'] == 0 or step == optim['steps']:
            vlog.write(json.dumps(dict(step=step, **validate(model, val_split, val_starts, run['validation_steps'])))+'\n')
            vlog.flush()
        if step % run['checkpoint_every'] == 0 or step == optim['steps']:
            partial = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step, config=config), partial)
            partial.rename(checkpoint)
    torch.save(dict(model=model.state_dict(), config=model_config.to_dict(), step=step), out/'model.pt')
    return step


def load_model(path, device='cpu'):
    """A trained ring model (`DIR/model.pt`) in eval mode."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = RingJEPA(RingConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--config', type=Path)
    t.add_argument('--set', action='append', default=[], help='an override, key.path=value (YAML)')
    a = p.parse_args()
    print(train(a.corpus, a.out, load_config(a.config, a.set)))


if __name__ == '__main__':
    main()
