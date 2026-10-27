"""C5/C6 for the corpus-v2 world models (docs/dynamic/PLAN.md, P5's hybrid): LeWM's motion decoder (C5) and its direct
head (C6) fitted on corpus v2's own windows, saved in `models.lewm.load_heads`'s format, so arms C5 and C6 load them
through `models` (`lewm_c5_decoder_and_c6`).

    python -m go2nav.training.lewm_heads_v2 WORLD.pt --corpus ROOT --out DIR [--train-units 400] [--val-units 60]
                                            [--transfer-units 20] [--stride 2] [--updates 20000] [--device cpu]

T7's decoder contexts (`training.lewm_decoder`) hold the six primitives and no lateral command; the lattice planner
asks for 25 continuous tapes, which corpus v2 covers. Windows: every `stride`-th window of units drawn with a fixed
seed, from the training mazes (fit) and from the validation mazes and transfer units (evaluation). The inputs are
built as the forecaster builds them at run time (`controllers.lewm`):
- z of frames t-3..t ([CLS]'s, for a grid model);
- C5: the 8-step rollout under the last three applied commands and the executed tape, each tick's command (the first
  of its block) five times;
- C6: the 1.5-s applied-command history and the tape.
Targets: the body-frame motion from t to t+h, h = 1..8, from the poses. Both heads are fitted together as
`lewm_corpus_probe`'s (AdamW 3e-4, cosine, batch 512 of (window, horizon) pairs, clip 1), in normalised target units
(the training targets' mean and spread), over all eight horizons.

Evaluation (`evaluation.json`): per split, at 500 and 800 ms, each head's median XY error and RMS yaw, overall and on
windows whose tape holds a lateral command or switches at least twice.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from go2nav.data.lewm_corpus import body_motion
from go2nav.models.lewm import (COMMAND_AXES, LatentDirectPredictor, LatentMotionDecoder, control_stats,
                                load_world_model, normalise_frames)
from go2nav.training.common import sha256_file

HORIZONS = 8
REPORTED = (5, 8)
SEED = 2026102701


@torch.inference_mode()
def unit_windows(model, folder, starts, mean, std, device, batch=512):
    """One unit's windows (frame t = start + 3): `zh` (W, 4, D), `rolled` (W, 8, D), `past` (W, 15, C) and `tape`
    (W, 8, C) normalised, `y` (W, 8, 3), and the `lateral` and `switching` flags of the tape."""
    frames = np.load(folder/'frames.npy', mmap_mode='r')
    steps = np.load(folder/'steps.npz')
    axes = COMMAND_AXES[model.config.command_dim]
    t = starts+3
    kind = torch.device(device).type
    autocast = lambda: torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == 'cuda')
    needed = np.unique(t[:, None]+np.arange(-3, 1))
    state = (len(frames), model.config.tokens, model.config.latent) if model.config.grid else \
        (len(frames), model.config.latent)
    z = torch.zeros(state)
    for s in range(0, len(needed), batch):
        chunk = needed[s:s+batch]
        x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
        with autocast():
            z[chunk] = model.encode(normalise_frames(x)[:, None])[:, 0].float().cpu()
    zh = torch.stack([z[t+k] for k in range(-3, 1)], 1)
    history = (steps['applied_history'][t][:, :, axes]-mean)/std                       # (W, 15, C)
    commands = steps['executed'][t[:, None]+np.arange(HORIZONS), 0][:, :, axes]       # (W, 8, C) as commanded
    tape = (commands-mean)/std
    actions = torch.from_numpy(np.concatenate((history[:, -3:], tape), 1).astype(np.float32))
    actions = actions[:, :, None].expand(-1, -1, 5, -1).flatten(2)                     # (W, 11, 5 C)
    rolled = []
    for s in range(0, len(t), batch):
        with autocast():
            rolled.append(model.cls(model.rollout(zh[s:s+batch].to(device), actions[s:s+batch].to(device),
                                                  HORIZONS)).float().cpu())
    switches = (np.abs(np.diff(commands, axis=1)) > 1e-6).any(-1).sum(1)
    return dict(zh=model.cls(zh), rolled=torch.cat(rolled), past=torch.from_numpy(history.astype(np.float32)),
                tape=torch.from_numpy(tape.astype(np.float32)),
                y=torch.from_numpy(body_motion(steps['pose'], t[:, None], t[:, None]+np.arange(1, HORIZONS+1))
                                   .astype(np.float32)),
                lateral=torch.from_numpy(np.abs(steps['executed'][t[:, None]+np.arange(HORIZONS), 0, 1]).max(1) > 1e-6),
                switching=torch.from_numpy(switches >= 2))


def gather(model, root, *, train_units, val_units, transfer_units, stride, device, seed=SEED):
    """The windows of each split, units drawn as `lewm_corpus_probe` draws them (index_12)."""
    rng = np.random.default_rng(seed)
    units = json.loads((root/'index_12.json').read_text())['units']
    index = np.load(root/'index_12.npz')

    def pick(role, kinds, n):
        ids = [i for i, u in enumerate(units) if u['role'] == role and u['kind'] in kinds]
        return sorted(rng.choice(ids, min(n, len(ids)), replace=False).tolist())
    chosen = dict(train=pick('lewm_train', ('explore', 'mission', 'free'), train_units),
                  val=pick('lewm_val', ('explore', 'mission', 'free'), val_units),
                  transfer=pick('lewm_val', ('transfer',), transfer_units))
    mean, std = control_stats(model.config.command_dim)
    data, started = {}, time.time()
    for split, ids in chosen.items():
        parts = [unit_windows(model, root/units[u]['path'], index['start'][index['unit'] == u][::stride], mean, std,
                              device) for u in ids if (index['unit'] == u).any()]
        data[split] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        print(json.dumps(dict(split=split, units=len(ids), windows=len(data[split]['y']),
                              wall_s=round(time.time()-started))), flush=True)
    return data, chosen


def predict(c5, c6, d, h, rows=slice(None)):
    """Both heads' motion at horizon h (1-8) for the windows `rows` of a split."""
    zh = d['zh'][rows]
    horizon = torch.full((len(zh),), h)
    return c5(zh, d['rolled'][rows, h-1]), c6(zh, d['past'][rows], d['tape'][rows], horizon)


def fit(data, latent, command_dim, *, updates, batch=512, seed=SEED):
    train = data['train']
    y = train['y'].flatten(0, 1)
    mean, scale = y.mean(0), y.std(0)
    c5, c6 = LatentMotionDecoder(latent, mean, scale).train(), LatentDirectPredictor(latent, mean, scale,
                                                                                     command_dim=command_dim).train()
    params = list(c5.parameters())+list(c6.parameters())
    opt = torch.optim.AdamW(params, 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates)
    g = torch.Generator().manual_seed(seed)
    for step in range(updates):
        rows = torch.randint(len(train['y']), (batch,), generator=g)
        h = torch.randint(1, HORIZONS+1, (batch,), generator=g)
        target = (train['y'][rows, h-1]-mean)/scale
        zh = train['zh'][rows]
        loss5 = (c5.normalized(zh, train['rolled'][rows, h-1])-target).pow(2).mean()
        loss6 = (c6.normalized(zh, train['past'][rows], train['tape'][rows], h)-target).pow(2).mean()
        opt.zero_grad()
        (loss5+loss6).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        sched.step()
        if (step+1) % 2000 == 0:
            print(json.dumps(dict(step=step+1, c5=round(float(loss5), 4), c6=round(float(loss6), 4))), flush=True)
    return c5.eval(), c6.eval(), mean, scale


@torch.no_grad()
def evaluate(c5, c6, data):
    out = {}
    for split, d in data.items():
        for h in REPORTED:
            masks = dict(all=torch.ones(len(d['y']), dtype=bool), lateral=d['lateral'], switching=d['switching'])
            p5, p6 = predict(c5, c6, d, h)
            y = d['y'][:, h-1]
            row = {}
            for head, p in (('c5', p5), ('c6', p6)):
                xy = (p[:, :2]-y[:, :2]).norm(dim=1)*1000
                yaw = torch.rad2deg(p[:, 2]-y[:, 2])
                row[head] = {name: dict(n=int(m.sum()), median_xy_mm=float(xy[m].median()) if m.any() else None,
                                        rmse_yaw_deg=float(yaw[m].pow(2).mean().sqrt()) if m.any() else None)
                             for name, m in masks.items()}
            row['median_true_mm'] = float((y[:, :2].norm(dim=1)*1000).median())
            out[f'{split}@{h*100}ms'] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('world', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--train-units', type=int, default=400)
    p.add_argument('--val-units', type=int, default=60)
    p.add_argument('--transfer-units', type=int, default=20)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.manual_seed(SEED)
    model = load_world_model(a.world, a.device)
    data, chosen = gather(model, a.corpus, train_units=a.train_units, val_units=a.val_units,
                          transfer_units=a.transfer_units, stride=a.stride, device=a.device)
    c5, c6, mean, scale = fit(data, model.config.latent, model.config.command_dim, updates=a.updates)
    evaluation = evaluate(c5, c6, data)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(c5=c5.state_dict(), c6=c6.state_dict(), latent=model.config.latent, target_mean=mean,
                    target_scale=scale, command_dim=model.config.command_dim, world_model=str(a.world),
                    world_model_sha256=sha256_file(a.world), units=chosen, stride=a.stride, updates=a.updates),
               a.out/'heads.pt')
    (a.out/'evaluation.json').write_text(json.dumps(evaluation, indent=1)+'\n')
    print(json.dumps({k: {h: v[h]['all']['median_xy_mm'] for h in ('c5', 'c6')} for k, v in evaluation.items()}))


if __name__ == '__main__':
    main()
