"""C5/C6: fit the LeWM motion decoder (C5) and its matched direct head (C6) on the C3/C4 decoder contexts, exactly as
T7 fits C3's decoder and C4, so that only the representation differs.

    python -m go2nav.training.lewm_decoder cache --world-model LEWM.pt --recordings-root ROOT [...] --out CACHE
    python -m go2nav.training.lewm_decoder fit --cache CACHE --out DIR --seed 2026093011 [--recipe decoder_fix]
    python -m go2nav.training.lewm_decoder evaluate --cache CACHE --checkpoint FIT.pt
    python -m go2nav.training.lewm_decoder choose DIR [DIR ...]

**Contexts**: data/training/decoder/contexts.jsonl (`go2nav.training.decoder`: groups, recipes, evaluation sets). For
LeWM a context's frames are t-300, t-200, t-100 and t (the 100-ms neighbours of its frame t, in the same recording or
replay); its past action blocks are the applied-command history's last three values (the commands executed after
frames t-3, t-2 and t-1, as the packet records them), and its future blocks the executed tape, each command
repeated for the tick's five policy steps: what the forecaster sees at run time.

**Cache** (`cache`): z of every frame (PNG decoding in parallel workers, encoding on the GPU), then, per context, the
world model's 8-step rollout with the executed tape. Stages leave .done markers; a rerun redoes only unfinished ones.

**Fit** (`fit`): T7's recipe: the same 64-example batches drawn by the recipe's (group, movement type) cells at one
of each context's horizons, 3,520 AdamW updates (lr 3e-4, weight decay 1e-4, clip 1), MSE in normalised motion units
(normalisation fitted on the recipe's training targets). C5 (`models.lewm.LatentMotionDecoder`) reads the rolled-out
future; C6 (`LatentDirectPredictor`) reads the frames' z, command history and candidate tape. Alongside, the
**probe** (gate L3) is C5's decoder trained and evaluated on the *true* future frame's z instead of the rollout: what
the encoder holds, as against what the predictor forecasts. A fit folder is written atomically, so a stopped fitting
resumes by seed. **Choice** (`choose`): T7's rule on C5's score (median 800-ms XY
error on eval_onpolicy, averaged over the six movement types); of three seeds, the median.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from PIL import Image
import torch

from go2nav.models.lewm import (LatentDirectPredictor, LatentMotionDecoder, control_stats, load_world_model,
                                normalise_frames, resize_frame)
from go2nav.training.common import sha256_file, write_lineage
from go2nav.training.decoder import (BATCH, CATEGORIES, EDGE_BINS, EVAL_SETS, GRAD_CLIP, LR, RECIPES, UPDATES,
                                     WEIGHT_DECAY, load_contexts, metrics, sampler, score)

SEEDS = (2026092205, 2026093011, 2026093012)  # T7's
HISTORY = 4


def neighbours(name, history=HISTORY, future=0):
    """The frame names at t-300 .. t ms for a context frame `name` (recording rgb_NNNN.png or replay NNNN.png), or with
    `future`, those at t+100 .. t+100*future ms."""
    folder, file = name.rsplit('/', 1)
    match = re.fullmatch(r'(rgb_)?(\d+)\.png', file)
    if match is None:
        raise ValueError(f'unrecognised frame name {name}')
    prefix, index, width = match.group(1) or '', int(match.group(2)), len(match.group(2))
    offsets = range(1, future+1) if future else range(-(history-1), 1)
    return [f'{folder}/{prefix}{index+k:0{width}d}.png' for k in offsets]


class _Frames(torch.utils.data.Dataset):
    def __init__(self, names, roots):
        self.names, self.roots = names, [Path(r) for r in roots]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        path = next((r/name for r in self.roots if (r/name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        return torch.from_numpy(resize_frame(Image.open(path)))


class LatentCache:
    """frames.f16 (frames, D), rollout.f16 (contexts, 8, D), frames.json, meta.json."""

    def __init__(self, folder, contexts):
        self.folder, self.contexts = Path(folder), contexts
        names = json.loads((self.folder/'frames.json').read_text())
        meta = json.loads((self.folder/'meta.json').read_text())
        self.latent = meta['latent']
        self.row = {n: i for i, n in enumerate(names)}
        self.z = np.memmap(self.folder/'frames.f16', dtype=np.float16, mode='r', shape=(len(names), self.latent))
        self.rollout = np.memmap(self.folder/'rollout.f16', dtype=np.float16, mode='r', shape=(len(contexts), 8, self.latent))
        self.mean, self.std = control_stats()

    def tensors(self, idx, horizons, device):
        rows = [self.row[f] for i in idx for f in neighbours(self.contexts[i]['frames'][2])]
        z = torch.from_numpy(np.asarray(self.z[rows], np.float32)).reshape(len(idx), HISTORY, self.latent).to(device)
        future = torch.from_numpy(np.stack([self.rollout[i, int(h)-1] for i, h in zip(idx, horizons)]).astype(np.float32)).to(device)
        true_rows = [self.row[neighbours(self.contexts[i]['frames'][2], future=8)[int(h)-1]] for i, h in zip(idx, horizons)]
        true = torch.from_numpy(np.asarray(self.z[true_rows], np.float32)).to(device)
        past = np.stack([(np.asarray(self.contexts[i]['past'], np.float32)[:, [0, 2]]-self.mean)/self.std for i in idx])
        tape = np.stack([(np.asarray(self.contexts[i]['tape'], np.float32)[:, [0, 2]]-self.mean)/self.std for i in idx])
        y = np.stack([np.asarray(self.contexts[i]['targets'], np.float32)[int(h)-1] for i, h in zip(idx, horizons)])
        return (z, future, torch.from_numpy(past).to(device), torch.from_numpy(tape).to(device), torch.from_numpy(y).to(device),
                true)

    @staticmethod
    @torch.inference_mode()
    def build(contexts, roots, world_model, out, *, batch=256, workers=16, device='cuda:0'):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        model = load_world_model(world_model, device)
        latent = model.config.latent
        names = sorted({f for c in contexts for f in neighbours(c['frames'][2])+neighbours(c['frames'][2], future=8)})
        row = {n: i for i, n in enumerate(names)}
        started = time.time()
        if not (out/'frames.done').exists():
            store = np.memmap(out/'frames.f16', dtype=np.float16, mode='w+', shape=(len(names), latent))
            loader = torch.utils.data.DataLoader(_Frames(names, roots), batch_size=batch, num_workers=workers, pin_memory=True,
                                                 prefetch_factor=4)
            done = 0
            for frames in loader:
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    z = model.encode(normalise_frames(frames.to(device, non_blocking=True))[:, None])[:, 0]
                store[done:done+len(frames)] = z.float().cpu().numpy()
                done += len(frames)
            store.flush()
            (out/'frames.json').write_text(json.dumps(names)+'\n')
            (out/'frames.done').touch()
            print(json.dumps(dict(frames=len(names), wall_s=round(time.time()-started))), flush=True)
        if not (out/'rollout.done').exists():
            z_all = torch.from_numpy(np.asarray(np.memmap(out/'frames.f16', dtype=np.float16, mode='r', shape=(len(names), latent)),
                                                np.float32)).to(device)
            mean, std = (torch.as_tensor(v, device=device) for v in control_stats())
            store = np.memmap(out/'rollout.f16', dtype=np.float16, mode='w+', shape=(len(contexts), 8, latent))
            for start in range(0, len(contexts), batch):
                part = contexts[start:start+batch]
                z = z_all[torch.tensor([[row[f] for f in neighbours(c['frames'][2])] for c in part], device=device)]
                past = torch.tensor(np.stack([np.asarray(c['past'], np.float32)[-3:, [0, 2]] for c in part]), device=device)
                tape = torch.tensor(np.stack([np.asarray(c['tape'], np.float32)[:, [0, 2]] for c in part]), device=device)
                commands = (torch.cat((past, tape), 1)-mean)/std                       # (B, 11, 2)
                actions = commands[:, :, None].expand(-1, -1, 5, -1).flatten(2)       # each tick's command, five times
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    store[start:start+len(part)] = model.rollout(z, actions, 8).float().cpu().numpy()
            store.flush()
            (out/'meta.json').write_text(json.dumps(dict(world_model=str(world_model), world_model_sha256=sha256_file(world_model),
                                                         latent=latent, contexts=len(contexts), frames=len(names)))+'\n')
            (out/'rollout.done').touch()
        return LatentCache(out, contexts)


def target_normalisation(contexts, recipe):
    """Mean and standard deviation of the recipe's training targets over all horizons."""
    groups = {key.split('/')[0] for key in RECIPES[recipe]}
    t = np.concatenate([np.asarray(c['targets'], np.float64) for c in contexts
                        if c['set'] == 'train' and ('*' in groups or c['group'] in groups)])
    return t.mean(0), t.std(0)


def update(model, optimiser, prediction, y):
    optimiser.zero_grad(set_to_none=True)
    loss = ((prediction-(y-model.target_mean)/model.target_scale)**2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()
    return float(loss.detach())


def fit(cache, out, *, seed, recipe='decoder_fix', updates=UPDATES, device='cuda:0'):
    out = Path(out)
    checkpoint = out/f'fit_{recipe}_s{seed}.pt'
    if (out/'.done').exists():
        return checkpoint
    partial = Path(str(out)+'.tmp')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started = time.time()
    draw, cells = sampler(cache.contexts, RECIPES[recipe], np.random.default_rng(seed))
    mean, scale = target_normalisation(cache.contexts, recipe)
    torch.manual_seed(seed)
    c5 = LatentMotionDecoder(cache.latent, mean, scale).to(device).train()
    c6 = LatentDirectPredictor(cache.latent, mean, scale).to(device).train()
    probe = LatentMotionDecoder(cache.latent, mean, scale).to(device).train()
    opt5 = torch.optim.AdamW(c5.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt6 = torch.optim.AdamW(c6.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optp = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for n in range(updates):
        idx, h = draw(BATCH)
        z, future, past, tape, y, true = cache.tensors(idx, h, device)
        loss5 = update(c5, opt5, c5.normalized(z, future), y)
        loss6 = update(c6, opt6, c6.normalized(z, past, tape, torch.from_numpy(h).to(device)), y)
        update(probe, optp, probe.normalized(z, true), y)
        if (n+1) % 100 == 0:
            history.append(dict(step=n+1, c5_loss=loss5, c6_loss=loss6))
    report = evaluate(cache, c5, c6, device, probe=probe)
    torch.save(dict(c5=c5.state_dict(), c6=c6.state_dict(), latent=cache.latent, target_mean=mean, target_scale=scale,
                    recipe=recipe, mix=RECIPES[recipe], world_model_sha256=json.loads((cache.folder/'meta.json').read_text())['world_model_sha256']),
               partial/checkpoint.name)
    (partial/'evaluation.json').write_text(json.dumps(dict(seed=seed, recipe=recipe, score_mm=score(report, 'C5'),
                                                           score_c6_mm=score(report, 'C6'), probe=probe_gate(report),
                                                           eval=report), indent=1)+'\n')
    write_lineage(partial, component=f'lewm_c5_decoder_and_c6_{recipe}', seed=seed, parents=[str(cache.folder/'meta.json')],
                  config=dict(recipe=recipe, mix=RECIPES[recipe], updates=updates, lr=LR, weight_decay=WEIGHT_DECAY),
                  data=dict(contexts='data/training/decoder/contexts.jsonl', cells=cells), checkpoint=partial/checkpoint.name,
                  metrics=dict(score_mm=score(report, 'C5'), score_c6_mm=score(report, 'C6'), history=history,
                               wall_s=round(time.time()-started, 1)))
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    (out/'.done').touch()
    return checkpoint


def probe_gate(report):
    """Gate L3: the probe (true future z) at 800 ms on eval_onpolicy, averaged over the movement types: median XY
    error at most 10 mm and RMS yaw error at most 1.5 degrees."""
    cells = report['eval_onpolicy@800ms/probe']
    xy = float(np.mean([cells[c]['median_xy_mm'] for c in CATEGORIES if c in cells]))
    yaw = float(np.mean([cells[c]['rmse_yaw_deg'] for c in CATEGORIES if c in cells]))
    return dict(median_xy_mm=xy, rmse_yaw_deg=yaw, l3=xy <= 10. and yaw <= 1.5)


@torch.no_grad()
def evaluate(cache, c5, c6, device='cuda:0', probe=None):
    c5.eval()
    c6.eval()
    report = {}
    for name, horizons in EVAL_SETS.items():
        idx_all = [i for i, c in enumerate(cache.contexts) if c['set'] == name]
        if not idx_all:
            continue
        cats = np.asarray([cache.contexts[i]['category'] for i in idx_all])
        bins = np.asarray([cache.contexts[i]['extra'].get('edge_bin', '') for i in idx_all])
        for h in horizons:
            preds = {'C5': [], 'C6': [], **({'probe': []} if probe is not None else {})}
            for k in range(0, len(idx_all), 512):
                idx = idx_all[k:k+512]
                z, future, past, tape, _, true = cache.tensors(idx, np.full(len(idx), h), device)
                preds['C5'].append(c5(z, future).cpu().numpy())
                preds['C6'].append(c6(z, past, tape, torch.full((len(idx),), h, device=device)).cpu().numpy())
                if probe is not None:
                    preds['probe'].append(probe.eval()(z, true).cpu().numpy())
            true = np.stack([np.asarray(cache.contexts[i]['targets'], np.float32)[h-1] for i in idx_all])
            for model, chunks in preds.items():
                p = np.concatenate(chunks)
                key = f'{name}@{100*h}ms/{model}'
                report[key] = metrics(p, true, cats)
                if name == 'eval_patch':
                    for b in EDGE_BINS:
                        if (bins == b).any():
                            report[key][f'bin:{b}'] = metrics(p[bins == b], true[bins == b], cats[bins == b])['all']
    c5.train()
    c6.train()
    return report


def choose(fit_dirs):
    """The median-scoring fit (C5's score) of several seeds."""
    scored = sorted((json.loads((Path(d)/'evaluation.json').read_text())['score_mm'], str(d)) for d in fit_dirs)
    return scored[len(scored)//2], scored


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--world-model', required=True, type=Path)
    c.add_argument('--recordings-root', action='append', required=True)
    c.add_argument('--out', required=True, type=Path)
    c.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    f = sub.add_parser('fit')
    f.add_argument('--cache', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--seed', type=int, required=True)
    f.add_argument('--recipe', choices=sorted(RECIPES), default='decoder_fix')
    e = sub.add_parser('evaluate')
    e.add_argument('--cache', required=True)
    e.add_argument('--checkpoint', required=True)
    ch = sub.add_parser('choose')
    ch.add_argument('fit_dirs', nargs='+')
    a = p.parse_args()
    if a.command == 'cache':
        LatentCache.build(load_contexts(), a.recordings_root, a.world_model, a.out, workers=a.workers)
    elif a.command == 'choose':
        print(json.dumps(choose(a.fit_dirs), indent=1))
    else:
        cache = LatentCache(a.cache, load_contexts())
        if a.command == 'fit':
            print(fit(cache, a.out, seed=a.seed, recipe=a.recipe))
        else:
            from go2nav.models.lewm import load_heads
            c5, c6 = load_heads(a.checkpoint, 'cuda:0')
            report = evaluate(cache, c5, c6)
            print(json.dumps(dict(score_mm=score(report, 'C5'), score_c6_mm=score(report, 'C6'), eval=report)))


if __name__ == '__main__':
    main()
