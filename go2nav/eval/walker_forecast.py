"""The world model's walkers and free space through its rollouts, offline (docs/dynamic/PLAN.md, P6).

    python -m go2nav.eval.walker_forecast WORLD.pt --corpus ROOT --out REPORT.json [--windows 12000] [--device cuda:0]

Fixed windows of the validation mazes (`val`: explore, mission and free units; `transfer`: the held-out looks), four history
frames and eight rolled out under the executed commands. Labels are from `go2nav.data.mover_labels` and `free_labels`.

**Walkers.** Each walker in view at t and at t+h, tracked between them by the nearest labels. At h = 0 the head reads the
encoded state; at h = 8 (0.8 s) the rolled-out one, against that frame's labels. `found`: a detection within `MATCH_M`;
`error`: the median distance to the nearest detection, where found. The baselines start from the true positions, so they
are stronger than any tracker: `constant_velocity` carries the walker's world motion over the last 0.3 s on, and `persistence`
leaves it where it was; each is placed in the true body frame at t+h.

**Free space.** The free-space head's median absolute error on all rays and on near rays (true distance under `NEAR_M`), and
`false_clear`, the share of near rays read as more than `FALSE_CLEAR_MARGIN_M` further off than they are: the error a
clearance check cannot afford.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data.lewm_dataset import Corpus
from go2nav.models.lewm import FREE_SCALE_M, MOVER_SCALE_M, load_world_model, normalise_frames

HISTORY, STEPS = 4, 8
MATCH_M, TRACK_M = .5, .3
NEAR_M, FALSE_CLEAR_MARGIN_M = .6, .15
PRESENCE = .5
SEED = 2026110102
KINDS = dict(val=('explore', 'mission', 'free'), transfer=('transfer',))


def world(pose, xy):
    """Body-frame points (..., 2) of a planar pose (x, y, yaw) in the world."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((pose[0]+c*xy[..., 0]-s*xy[..., 1], pose[1]+s*xy[..., 0]+c*xy[..., 1]), -1)


def body(pose, xy):
    """World points (..., 2) in the body frame of a planar pose."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = xy-pose[:2]
    return np.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]), -1)


def walkers(row):
    """The labelled positions of a frame's columns (n, 2) from (columns, 3)."""
    return row[row[:, 0] > 0, 1:]


def nearest(point, others):
    """(distance, position) of the nearest of (n, 2), (inf, None) if there are none."""
    if not len(others):
        return np.inf, None
    gaps = np.linalg.norm(others-point, axis=1)
    k = int(gaps.argmin())
    return float(gaps[k]), others[k]


def tracked(labels, poses, horizon):
    """A window's walkers followed from t to t+`horizon`: rows (at t in the world, 0.3 s earlier, at t+h in its body frame)."""
    t, end = HISTORY-1, HISTORY-1+horizon
    rows = []
    for here in walkers(labels[t]):
        now = world(poses[t], here)
        gap, earlier = nearest(now, world(poses[t-3], walkers(labels[t-3])) if len(walkers(labels[t-3])) else np.zeros((0, 2)))
        if gap > TRACK_M+.3:
            continue
        ahead = walkers(labels[end])
        if not len(ahead):
            continue
        walked = (now-earlier)/.3*horizon*.1
        gap, there = nearest(body(poses[end], now+walked), ahead)
        if gap <= TRACK_M+.3:
            rows.append((now, earlier, there))
    return rows


def detections(out):
    """A head's (columns, 3) detections (n, 2) in metres."""
    keep = out[:, 0] > 0
    return out[keep, 1:]*MOVER_SCALE_M


def summary(errors):
    errors = np.array(errors, float)
    found = errors <= MATCH_M
    return dict(walkers=len(errors), found=float(found.mean()) if len(errors) else None,
                error_m=float(np.median(errors[found])) if found.any() else None)


def split(model, corpus, count, device, batch=48):
    """The report of one split."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(corpus), size=min(count, len(corpus)), replace=False)
    now, ahead, base, still = [], [], [], []
    free = {name: [] for name in ('encoded_error', 'encoded_near', 'encoded_false', 'rolled_error', 'rolled_near', 'rolled_false')}
    for s in range(0, len(chosen), batch):
        indices = chosen[s:s+batch]
        frames, actions, keys, movers, rays = corpus.batch(indices)
        poses = [corpus.poses(int(u))[int(k):int(k)+corpus.length] for u, k in keys.numpy()]
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            z = model.encode(normalise_frames(frames.to(device)))
            rolled = model.rollout(z[:, :HISTORY], actions.to(device), STEPS)
            encoded_out = model.movers(z[:, HISTORY-1]).float()
            rolled_out = model.movers(rolled[:, -1]).float()
            read = (model.free_rays(z[:, HISTORY-1]).float()*FREE_SCALE_M, model.free_rays(rolled[:, -1]).float()*FREE_SCALE_M) \
                if model.config.free_head else None
        encoded_out, rolled_out = encoded_out.cpu().numpy(), rolled_out.cpu().numpy()
        movers = movers.numpy()
        for i in range(len(indices)):
            seen = detections(encoded_out[i])
            for here in walkers(movers[i, HISTORY-1]):
                now.append(nearest(here, seen)[0])
            guess = detections(rolled_out[i])
            for at_t, earlier, there in tracked(movers[i], poses[i], STEPS):
                ahead.append(nearest(there, guess)[0])
                end = poses[i][HISTORY-1+STEPS]
                base.append(float(np.linalg.norm(body(end, at_t+(at_t-earlier)/.3*.8)-there)))
                still.append(float(np.linalg.norm(body(end, at_t)-there)))
        if read is not None:
            true = (rays[:, HISTORY-1].numpy(), rays[:, -1].numpy())
            for name, value, target in zip(('encoded', 'rolled'), read, true):
                value, near = value.cpu().numpy(), target < NEAR_M
                free[f'{name}_error'].extend(np.abs(value-target).ravel().tolist())
                free[f'{name}_near'].extend(np.abs(value-target)[near].tolist())
                free[f'{name}_false'].extend((value[near]-target[near] > FALSE_CLEAR_MARGIN_M).tolist())
    report = dict(windows=len(chosen), walkers_now=summary(now), forecast_0_8s=summary(ahead),
                  constant_velocity_0_8s=summary(base), persistence_0_8s=summary(still))
    if model.config.free_head:
        report['free'] = {n: dict(all_error_m=float(np.median(free[f'{n}_error'])), near_error_m=float(np.median(free[f'{n}_near'])),
                                  false_clear=float(np.mean(free[f'{n}_false'])), near_rays=len(free[f'{n}_near']))
                          for n in ('encoded', 'rolled')}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('world', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--windows', type=int, default=12000)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    device = torch.device(a.device)
    model = load_world_model(a.world, device)
    report = {}
    for name, kinds in KINDS.items():
        corpus = Corpus(a.corpus, length=HISTORY+STEPS, roles=('lewm_val',), kinds=kinds, command_dim=model.config.command_dim,
                        movers=True, free=True)
        report[name] = split(model, corpus, a.windows, device)
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text+'\n')


if __name__ == '__main__':
    main()
