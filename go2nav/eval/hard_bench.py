"""P33's measures (docs/dynamic/PLAN.md P33) on the hard toy's validation episodes. Labels are for scoring only.

    python -m go2nav.eval.hard_bench stage1 --data ROOT                 # both estimators: found, precision, identity
    python -m go2nav.eval.hard_bench score --data ROOT --run RUN        # the head, then P27's measures: RUN/bench_mlp.json

**Stage 1** (`stage1`): over every validation frame, the obstacles that exist and are at least half visible (not under
a bar); one is found if a detection is within 1 px; precision is the found detections' mean error; identity: of the
obstacles found at a moment's history frame t0 (every validation frame from 2 on, every 4th), the share whose chain
(`data.hard_tracks.chain`) is within 1 px of the same obstacle at both earlier history frames, where it is visible.

**Score:** the read-out head is an MLP on one token (3 layers of 1,024; no slot identity, since slots are arbitrary),
trained on encoded training tokens against each token's true object (the agent for the first; for a slot, the nearest
existing obstacle within 1.5 px of its position). The forecaster rolls each window out and reads the tokens; each
existing obstacle is given the slot nearest it at t0 within 1.5 px (else it is not found and left out); then
`eval.recipe_bench.measure`. R: the head's error on the encoded tokens of validation frames, per object.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from go2nav.data import hard_tracks as ht
from go2nav.eval import recipe_bench as rb
from go2nav.sim import reactive_balls as world
from go2nav.training.balls import schedule
from go2nav.training.hard_tokens import HISTORY, HardWindows

FOUND_PX, MATCH_PX = 1., 1.5
HEAD = dict(steps=10000, batch=512, hidden=1024)


def stage1(root, estimator):
    tracks = ht.load(root, 'val', estimator)
    from go2nav.data import reactive_balls as data
    _, labels, _ = data.load(root, 'interactive', 'val', 't2')
    truth, live, vis = labels['obstacle'], labels['active'], labels['visible']
    xy = tracks['xy']
    d = np.linalg.norm(truth[:, :, :, None]-xy[:, :, None], axis=-1)                       # (E, T, K, S)
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.min(-1)
    counted = np.broadcast_to(live[:, None], best.shape) & (vis > .5)
    found = counted & (best < FOUND_PX)
    e, t0 = np.meshgrid(np.arange(xy.shape[0]), np.arange(HISTORY-1, xy.shape[1], 4), indexing='ij')
    e, t0 = e.ravel(), t0.ravel()
    pos, seen, present = ht.chain(xy, tracks['ahead'], tracks['back'], e, t0, before=HISTORY-1, after=0)
    kept = total = 0
    for k in range(world.K):
        at = found[e, t0, k]
        slot = d[e, t0, k].argmin(-1)
        ok = at.copy()
        for s in range(HISTORY-1):
            frame = t0-(HISTORY-1)+s
            near = np.linalg.norm(pos[np.arange(len(e)), s, slot]-truth[e, frame, k], axis=-1) < FOUND_PX
            ok &= near | ~(vis[e, frame, k] > .5)               # where it is under a bar, nothing is asked
        kept += int(ok.sum())
        total += int(at.sum())
    return dict(estimator=estimator, found=float(found.sum()/counted.sum()), precision_px=float(best[found].mean()),
                median_px=float(np.median(best[found])), identity_kept=kept/max(1, total), counted=int(counted.sum()))


class TokenHead(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


def truth_for(windows, b, frames):
    """Each token's true object at the batch's frames (B, T, 1+S, 2), NaN where none: the agent for the first; a slot's
    nearest existing obstacle within MATCH_PX."""
    L = windows.labels
    e = b['e'][:, None]
    agent = L['agent'][e, frames]
    obstacles = np.where(L['active'][b['e']][:, None, :, None], L['obstacle'][e, frames], np.nan)       # (B, T, K, 2)
    slots = b['located'][:, :, 1:].cpu().numpy()
    d = np.linalg.norm(slots[:, :, :, None]-obstacles[:, :, None], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    which = d.argmin(-1)
    near = np.take_along_axis(obstacles, which[..., None].repeat(2, -1), 2)
    near = np.where((np.take_along_axis(d, which[..., None], -1) < MATCH_PX) & b['seen'][:, :, 1:, None].cpu().numpy(), near, np.nan)
    return np.concatenate((agent[:, :, None], near), 2)


def train_head(model, root, estimator, device, steps, batch, hidden, cache=200, seed=0):
    windows = HardWindows(root, 'train', estimator, device)
    rng = np.random.default_rng(2026111934+seed)
    tokens, targets = [], []
    with torch.no_grad():
        for _ in range(cache):
            e = rng.integers(0, windows.episodes, 256)
            t0 = rng.integers(HISTORY-1, windows.length, 256)
            b = windows.batch(e, t0, 0)
            z = model.encode(b['pixels'], b['located'], b['seen'])
            truth = truth_for(windows, b, t0[:, None]+np.arange(1-HISTORY, 1))
            keep = torch.as_tensor(np.isfinite(truth).all(-1), device=device)
            tokens.append(z[keep].float())
            targets.append(((torch.as_tensor(truth, device=device).float()-rb.HALF)/rb.HALF)[keep])
    tokens, targets = torch.cat(tokens), torch.cat(targets)
    del windows
    torch.manual_seed(2026111935+seed)
    head = TokenHead(model.config.latent, hidden).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(200, steps))
    g = torch.Generator(device=device).manual_seed(2026111936+seed)
    for step in range(steps):
        pick = torch.randint(0, len(tokens), (batch,), generator=g, device=device)
        loss = (head(tokens[pick])-targets[pick]).pow(2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step+1) % 2000 == 0:
            print(f'head step {step+1}: rms {rb.HALF*float(loss.detach())**.5:.3f} px ({len(tokens)} tokens)', flush=True)
    return head.eval().requires_grad_(False)


def forecaster(model, head, windows, batch=512):
    """`eval.recipe_bench`'s forecaster interface over the hard toy's estimator tokens."""
    device = windows.device

    @torch.no_grad()
    def forecast(bench, e, t0, future):
        out = np.full((len(e), rb.STEPS, 1+world.K, 2), np.nan, np.float32)
        future = torch.as_tensor(np.asarray(future, np.float32), device=device)/world.MAX_ACTION
        for i in range(0, len(e), batch):
            rows = slice(i, min(i+batch, len(e)))
            b = windows.batch(e[rows], t0[rows], 0)
            z = model.encode(b['pixels'], b['located'], b['seen'])
            actions = torch.cat((b['actions'], future[rows]), 1)
            read = (head(model.rollout(z, actions, rb.STEPS).float())*rb.HALF+rb.HALF).cpu().numpy()   # (B, 16, 1+S, 2)
            out[rows, :, 0] = read[:, :, 0]
            slots = b['located'][:, -1, 1:].cpu().numpy()                                         # (B, S, 2) at t0
            present = b['seen'][:, -1, 1:].cpu().numpy()
            truth = windows.labels['obstacle'][e[rows], t0[rows]]
            live = windows.labels['active'][e[rows]]
            for r in range(len(slots)):
                taken = set()
                for k in np.flatnonzero(live[r]):
                    d = np.linalg.norm(np.nan_to_num(slots[r], nan=1e3)-truth[r, k], axis=-1)
                    d[~present[r]] = 1e3
                    for s in taken:
                        d[s] = 1e3
                    s = int(d.argmin())
                    if d[s] < MATCH_PX:
                        out[i+r, :, 1+k] = read[r, :, 1+s]
                        taken.add(s)
        return out
    return forecast


@torch.no_grad()
def retention(model, head, windows, every=4):
    """R: the head on the encoded tokens of validation frames, per object (mean px), over matched tokens."""
    errors = [[] for _ in range(1+world.K)]
    for start in range(0, windows.episodes, 50):
        e, t0 = np.meshgrid(np.arange(start, min(start+50, windows.episodes)), np.arange(HISTORY-1, windows.length, every),
                            indexing='ij')
        e, t0 = e.ravel(), t0.ravel()
        b = windows.batch(e, t0, 0)
        read = (head(model.encode(b['pixels'], b['located'], b['seen'])[:, -1].float())*rb.HALF+rb.HALF).cpu().numpy()
        errors[0].append(np.linalg.norm(read[:, 0]-windows.labels['agent'][e, t0], axis=-1))
        slots = b['located'][:, -1, 1:].cpu().numpy()
        for k in range(world.K):
            truth = windows.labels['obstacle'][e, t0, k]
            d = np.linalg.norm(np.nan_to_num(slots, nan=1e3)-truth[:, None], axis=-1)
            s = d.argmin(-1)
            ok = (d[np.arange(len(e)), s] < MATCH_PX) & windows.labels['active'][e, k]
            errors[1+k].append(np.linalg.norm(read[np.arange(len(e)), 1+s][ok]-truth[ok], axis=-1))
    names = ('agent',)+tuple(f'obstacle {k}' for k in range(world.K))
    return {n: float(np.concatenate(v).mean()) for n, v in zip(names, errors)}


def score(root, run):
    from go2nav.training.recipe_bench import load
    run, device = Path(run), torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.35)
    model, state = load(run/'model.pt', device)
    estimator = state['estimator']
    head = train_head(model, root, estimator, device, **HEAD)
    torch.save(head.state_dict(), run/'head_mlp.pt')
    windows = HardWindows(root, 'val', estimator, device)
    bench = rb.Bench(root, 't2', 'interactive', 'val')
    row = rb.measure(forecaster(model, head, windows), bench)
    row['R'] = retention(model, head, windows)
    row['meta'] = dict(recipe=state['recipe'], estimator=estimator, seed=state['seed'], steps=state['step'])
    (run/'bench_mlp.json').write_text(json.dumps(row, indent=1))
    print(rb.table({run.name: rb.keyed(json.loads(json.dumps(row)))}))
    print('R:', {k: round(v, 3) for k, v in row['R'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s1 = sub.add_parser('stage1')
    s1.add_argument('--data', required=True)
    s1.add_argument('--estimators', nargs='+', default=['blobs', 'keypoints'])
    sc = sub.add_parser('score')
    sc.add_argument('--data', required=True)
    sc.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.command == 'stage1':
        for est in args.estimators:
            print(json.dumps(stage1(args.data, est)))
    else:
        score(args.data, args.run)


if __name__ == '__main__':
    main()
