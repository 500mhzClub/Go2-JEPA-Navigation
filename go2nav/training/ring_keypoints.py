"""P31 stage 1 (docs/dynamic/PLAN.md P31 items 2, 5 and 6): P30's locator on the ring world's rasterised scans.

    python -m go2nav.training.ring_keypoints train --corpus CORPUS --scenes F004 --out DIR [--seed S] [--set k=v ...]

**Training:** pairs of frames 1-8 apart in one training episode of the ring corpus, both drawn in the earlier frame's
pose (`data.ring_rasters`), the images in [-1, 1]. `models.keypoints.Transporter` with one channel and K = 6; the
masks narrow from 8 to 2 px over the first half and the separation fades with them (P30's round 2). No labels and no
layout. DIR/checkpoint.pt every `checkpoint_every` steps; rerunning resumes.

**Measures** (`stage1`, labels for scoring only), on F004's validation scenes: the history's three scans (the
branch point t0 and 0.4 and 0.8 s before), drawn in the robot's pose at t0.
- **found:** at t0, the keypoints are matched to the walkers in sight within 3 m of the robot by least total
  distance; a walker is found if its match is within 0.3 m. Read over the walkers the tracker matched (F004's
  population) and over all walkers in sight within 3 m (beside the tracker's share of those).
- **precision:** the median distance of the found matches.
- **identity:** each step's keypoints are joined to the step before by least total distance (P31 item 3); a found
  walker keeps its identity if its joined keypoint is within 0.3 m of it at both earlier steps (where it is in sight).
DIR/locator.pt holds the keypoint network and the measures; DIR/measures.json the measures.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from go2nav.data import ring_rasters as rr
from go2nav.models.keypoints import KeypointConfig, Transporter, separation

DEFAULTS = dict(steps=10000, batch=64, lr=1e-3, warmup=500, gap=(1, 8), keypoints=6, sigma=2., sigma_start=8.,
                anneal=.5, separation_weight=.01, log_every=250, checkpoint_every=2000, memory_fraction=.45)
HISTORY_FRAMES = (-8, -4, 0)
FOUND_M, SIGHT_M = .3, 3.


def image(corpus, index, frame):
    """The scans at packed frames `index` (N,) drawn in `frame` (N, 3), in [-1, 1]."""
    return rr.rasterise(corpus.ranges[index], corpus.pose[index], frame)*2-1


@torch.no_grad()
def keypoints_world(net, corpus, index, frame, batch=2048):
    """The locator's keypoints (N, K, 2) in the world for the scans at `index` drawn in `frame`."""
    out = []
    for i in range(0, len(index), batch):
        px = net(image(corpus, index[i:i+batch], frame[i:i+batch]))
        out.append(rr.to_world(rr.to_local(px), frame[i:i+batch]))
    return torch.cat(out).cpu().numpy()


def join(a, b):
    """Keypoints a (K, 2) to b (K, 2) by least total distance: the index in b of each keypoint of a (K,)."""
    rows, cols = linear_sum_assignment(np.linalg.norm(a[:, None]-b[None], axis=-1))
    out = np.empty(len(a), np.int64)
    out[rows] = cols
    return out


def stage1(net, corpus, scenes, device):
    """The module's measures. `corpus`: the validation `data.ring_rasters.Corpus` with labels; `scenes`: F004's
    validation arrays (episode, t0, walker, visible_count)."""
    e, t0 = scenes['episode'].astype(np.int64), scenes['t0'].astype(np.int64)
    at = corpus.starts[e]+t0
    frame = corpus.pose[torch.as_tensor(at, device=device)]
    kp = {s: keypoints_world(net, corpus, torch.as_tensor(at+s, device=device), frame) for s in HISTORY_FRAMES}
    pose = corpus.pose[torch.as_tensor(at, device=device)].cpu().numpy()
    truth = {s: corpus.walkers_xy[at+s] for s in HISTORY_FRAMES}
    sight = {s: corpus.in_sight[at+s] for s in HISTORY_FRAMES}
    counts = dict(tracker_population=0, tracker_found=0, all_visible=0, all_found=0, kept=0, kept_counted=0)
    errors = []
    for n in range(len(e)):
        w_truth = truth[0][n]
        visible = sight[0][n] & np.isfinite(w_truth).all(-1) & (np.linalg.norm(w_truth-pose[n, :2], axis=-1) <= SIGHT_M)
        walkers = np.flatnonzero(visible)
        matched = {}
        if len(walkers):
            cost = np.linalg.norm(kp[0][n][:, None]-w_truth[walkers][None], axis=-1)
            rows, cols = linear_sum_assignment(cost)
            for k, j in zip(rows, cols):
                if cost[k, j] < FOUND_M:
                    matched[walkers[j]] = k
                    errors.append(cost[k, j])
        counts['all_visible'] += len(walkers)
        counts['all_found'] += len(matched)
        tracked = [int(w) for w in scenes['walker'][n] if w >= 0]
        counts['tracker_population'] += len(tracked)
        counts['tracker_found'] += sum(w in matched for w in tracked)
        back1, back2 = join(kp[0][n], kp[-4][n]), join(kp[-4][n], kp[-8][n])
        for w, k in matched.items():
            ok = True
            for s, k_s in ((-4, back1[k]), (-8, back2[back1[k]])):
                if sight[s][n][w] and np.isfinite(truth[s][n][w]).all():
                    ok &= np.linalg.norm(kp[s][n][k_s]-truth[s][n][w]) < FOUND_M
            counts['kept'] += int(ok)
            counts['kept_counted'] += 1
    tracker_all = float(scenes['walker'].__ge__(0).sum()/max(1, scenes['visible_count'].sum()))
    return dict(found_of_tracker=counts['tracker_found']/max(1, counts['tracker_population']),
                found_of_all=counts['all_found']/max(1, counts['all_visible']), tracker_of_all=tracker_all,
                identity_kept=counts['kept']/max(1, counts['kept_counted']),
                median_error_m=float(np.median(errors)) if errors else None,
                mean_error_m=float(np.mean(errors)) if errors else None, **counts)


def load_scenes(root):
    folder = Path(root)/'val'
    return {k: np.load(folder/f'{k}.npy') for k in ('episode', 't0', 'walker', 'visible_count')}


def train(corpus_root, scenes_root, out, seed=0, overrides=None):
    cfg = DEFAULTS | (overrides or {})
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_seed = 2026111890+seed
    torch.manual_seed(run_seed)
    config = KeypointConfig(keypoints=cfg['keypoints'], sigma=cfg['sigma'], channels=1, size=rr.SIZE)
    model = Transporter(config).to(device)
    corpus = rr.Corpus(corpus_root, 'train', device)
    episode = np.repeat(np.arange(len(corpus.starts)-1), np.diff(corpus.starts))
    end = torch.as_tensor(corpus.starts[1:][episode], device=device)                # each frame's episode end
    frames = len(episode)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    from go2nav.training.balls import schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(cfg['warmup'], cfg['steps']))
    step, checkpoint, metrics = 0, out/'checkpoint.pt', out/'metrics.jsonl'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        if metrics.exists():
            rows = [r for r in metrics.read_text().splitlines() if json.loads(r).get('step', 0) <= step]
            metrics.write_text(''.join(r+'\n' for r in rows))
    (out/'config.json').write_text(json.dumps(dict(train=cfg, model=asdict(config), seed=seed, run_seed=run_seed), indent=1))
    log, started = metrics.open('a'), time.time()
    model.train()
    while step < cfg['steps']:
        g = torch.Generator(device=device).manual_seed(run_seed*10_000_000+step)
        gap = torch.randint(cfg['gap'][0], cfg['gap'][1]+1, (cfg['batch'],), generator=g, device=device)
        s = torch.randint(0, frames, (cfg['batch'],), generator=g, device=device)
        s = torch.minimum(s, end[s]-1-gap)
        frame = corpus.pose[s]
        source, target = image(corpus, s, frame), image(corpus, s+gap, frame)
        wide = max(0., 1-step/(cfg['anneal']*cfg['steps']))
        sigma = cfg['sigma']+(cfg['sigma_start']-cfg['sigma'])*wide
        rebuilt, kp = model(source, target, sigma)
        rebuild, apart = F.mse_loss(rebuilt, target), separation(kp)
        loss = rebuild+cfg['separation_weight']*wide*apart
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        step += 1
        if step % cfg['log_every'] == 0:
            log.write(json.dumps(dict(step=step, loss=round(float(loss.detach()), 6), rebuild=round(float(rebuild.detach()), 6),
                                      separation=round(float(apart.detach()), 4), sigma=round(sigma, 2),
                                      steps_per_s=round(cfg['log_every']/(time.time()-started), 1)))+'\n')
            log.flush()
            started = time.time()
        if step % cfg['checkpoint_every'] == 0 and step < cfg['steps']:
            partial = checkpoint.with_suffix('.partial')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step), partial)
            os.replace(partial, checkpoint)
    del corpus
    torch.cuda.empty_cache()
    model.eval()
    result = stage1(model.keypoints, rr.Corpus(corpus_root, 'val', device, labels=True), load_scenes(scenes_root), device)
    torch.save(dict(config=asdict(config), keypoints=model.keypoints.state_dict(), transporter=model.state_dict(),
                    measures=result, step=step), out/'locator.pt')
    (out/'measures.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True)
    t.add_argument('--scenes', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--set', action='append', default=[], help='key=value (a DEFAULTS key; the value parsed as JSON)')
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    train(args.corpus, args.scenes, args.out, args.seed, overrides)


if __name__ == '__main__':
    main()
