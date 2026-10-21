"""Readout-free and low-capacity measures of a world model's latents, beside the learned decoders (L3, L4).

    python -m go2nav.eval.latent_measures lewm --world-model LEWM.pt --cache CHAIN_CACHE --out FILE

A learned decoder's errors are charged to the world model it reads (span8w's T7 decoder deadlocked C5 while the
same latents, read by another head, did not). These measures need no decoder, or one too small to hide much:

- **Linear probe**: ridge regression from [z_t, f - z_t, z_t - z_(t-1), z_t - z_(t-3)] to the body-frame motion, f
  either the true future frame's z (what the encoder holds) or the rollout's (what the predictor forecasts); fitted on
  T7's training contexts (all eight horizons), the ridge strength chosen on a fifth held out, scored like L3/L4
  (median XY and RMS yaw averaged over the six movement types) on eval_onpolicy and eval_fresh_c3.
- **Candidate ranking**: at each held-out decision (eval_onpolicy: C1's states; eval_fresh_c3: C3's), the six
  candidates' rollouts (the executed prefix, then each action's four ticks after the safety limits, as the planner
  forecasts them) at 700 ms, the end of the committed action; the executed candidate should land nearest the true
  future z. Top-1 accuracy and mean rank (chance 1/6 and 3.5), per movement type; decisions whose executed tape
  matches no candidate are skipped.
- **Rollout error against baselines** at 800 ms: the rollout's distance to the true z, against persistence (z_t), the
  zero-command rollout and a shuffled-tape rollout (tapes permuted across decisions), as ratios, per movement type.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.training.decoder import CATEGORIES, load_contexts

RANK_HORIZON = 7    # 700 ms, the end of the committed action
ERROR_HORIZON = 8   # 800 ms


def l3_style(pred, true, cats):
    out = {}
    for c in CATEGORIES:
        m = cats == c
        if m.any():
            e = np.linalg.norm(pred[m, :2]-true[m, :2], axis=1)*1000
            yaw = np.angle(np.exp(1j*(pred[m, 2]-true[m, 2])))
            out[c] = dict(n=int(m.sum()), median_xy_mm=float(np.median(e)), rmse_yaw_deg=float(np.degrees(np.sqrt(np.mean(yaw**2)))))
    out['mean'] = dict(median_xy_mm=float(np.mean([v['median_xy_mm'] for v in out.values()])),
                       rmse_yaw_deg=float(np.mean([v['rmse_yaw_deg'] for k, v in out.items() if k != 'mean'])))
    return out


def ridge(X, Y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-6
    Xs = (X-mu)/sd
    ym = Y.mean(0)
    A = Xs.T@Xs+lam*len(X)*np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T@(Y-ym))
    return lambda Z: ((Z-mu)/sd)@W+ym


def fit_probe(Xtr, Ytr, lams=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.)):
    fold = np.arange(len(Xtr)) % 5 == 0
    best = min(lams, key=lambda l: np.mean((ridge(Xtr[~fold], Ytr[~fold], l)(Xtr[fold])-Ytr[fold])**2))
    return ridge(Xtr, Ytr, best), best


def features(zh, f):
    cur = zh[:, -1]
    return np.concatenate([cur, f-cur, cur-zh[:, -2], cur-zh[:, 0]], 1)


def candidate_tapes(past, tape):
    """(6, 8, 3) candidate tapes as executed: the decision's executed prefix, each action for four ticks, then zero,
    after the safety limits from the prefix's last command. The index of the executed one, or None."""
    from go2nav.locomotion.safety import apply_limits
    from go2nav.planning.actions import ACTIONS, command
    requested = np.zeros((6, 8, 3), np.float32)
    requested[:, :3] = tape[:3]
    for i, a in enumerate(ACTIONS):
        requested[i, 3:7] = command(a)
    limited = apply_limits(requested[:, 3:], np.broadcast_to(tape[2], (6, 3)))
    cands = np.concatenate([np.broadcast_to(tape[:3], (6, 3, 3)), limited], 1)
    match = [i for i in range(6) if np.allclose(cands[i, 3:7], tape[3:7], atol=1e-6)]
    return cands, (match[0] if len(match) == 1 else None)


@torch.inference_mode()
def lewm(world_model, cache_folder, out, device='cuda:0'):
    from go2nav.models.lewm import control_stats, load_world_model
    from go2nav.training.lewm_decoder import LatentCache, neighbours
    contexts = load_contexts()
    cache = LatentCache(cache_folder, contexts)
    model = load_world_model(world_model, device)
    mean, std = (torch.as_tensor(v, device=device) for v in control_stats())
    z = lambda names: np.asarray(cache.z[[cache.row[n] for n in names]], np.float32)

    def context_arrays(idx):
        zh = z([f for i in idx for f in neighbours(contexts[i]['frames'][2])]).reshape(len(idx), 4, -1)
        true = z([f for i in idx for f in neighbours(contexts[i]['frames'][2], future=8)]).reshape(len(idx), 8, -1)
        roll = np.asarray(cache.rollout[idx], np.float32)
        y = np.stack([np.asarray(contexts[i]['targets'], np.float32) for i in idx])
        cats = np.asarray([contexts[i]['category'] for i in idx])
        return zh, true, roll, y, cats

    def rollout(zh, past, tapes):
        """zh (B, 4, D) np; past (B, 3, 3); tapes (B, 8, 3) -> (B, 8, D)."""
        out_ = []
        for s in range(0, len(zh), 512):
            commands = (torch.tensor(np.concatenate([past[s:s+512], tapes[s:s+512]], 1)[..., [0, 2]], device=device)-mean)/std
            actions = commands[:, :, None].expand(-1, -1, 5, -1).flatten(2)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                out_.append(model.rollout(torch.tensor(zh[s:s+512], device=device), actions, 8).float().cpu().numpy())
        return np.concatenate(out_)

    train = [i for i, c in enumerate(contexts) if c['set'] == 'train' and c['group'] in ('old', 'maze', 'onpolicy')]
    ztr, ttr, rtr, ytr, _ = context_arrays(train)
    report = dict(world_model=str(world_model), probe={}, ranking={}, rollout_error={})
    for kind, ftr in (('true', ttr), ('rollout', rtr)):
        X = np.concatenate([features(ztr, ftr[:, h]) for h in range(8)])
        Y = np.concatenate([ytr[:, h] for h in range(8)])
        probe, lam = fit_probe(X, Y)
        for name in ('eval_onpolicy', 'eval_fresh_c3'):
            idx = [i for i, c in enumerate(contexts) if c['set'] == name]
            zh, true, roll, y, cats = context_arrays(idx)
            f = true if kind == 'true' else roll
            for h in (5, 8):
                report['probe'][f'{name}@{100*h}ms/{kind}'] = dict(ridge=lam, **l3_style(probe(features(zh, f[:, h-1])), y[:, h-1], cats))
    rng = np.random.default_rng(0)
    for name in ('eval_onpolicy', 'eval_fresh_c3'):
        idx = [i for i, c in enumerate(contexts) if c['set'] == name]
        zh, true, roll, y, cats = context_arrays(idx)
        past = np.stack([np.asarray(contexts[i]['past'], np.float32)[-3:] for i in idx])
        tapes = np.stack([np.asarray(contexts[i]['tape'], np.float32) for i in idx])
        cands, which = zip(*(candidate_tapes(np.asarray(contexts[i]['past'], np.float32), t) for i, t in zip(idx, tapes)))
        keep = np.array([w is not None for w in which])
        sel = np.flatnonzero(keep)
        rolled = rollout(np.repeat(zh[sel], 6, 0), np.repeat(past[sel], 6, 0), np.concatenate([cands[i] for i in sel])).reshape(len(sel), 6, 8, -1)
        d = np.linalg.norm(rolled[:, :, RANK_HORIZON-1]-true[sel, None, RANK_HORIZON-1], axis=-1)
        rank = (d < d[np.arange(len(sel)), [which[i] for i in sel]][:, None]).sum(1)+1
        rc = cats[sel]
        report['ranking'][name] = dict(decisions=int(len(idx)), matched=int(len(sel)), top1=float(np.mean(rank == 1)),
                                       mean_rank=float(rank.mean()),
                                       by_type={c: dict(n=int((rc == c).sum()), top1=float(np.mean(rank[rc == c] == 1)),
                                                        mean_rank=float(rank[rc == c].mean())) for c in CATEGORIES if (rc == c).any()})
        zero = rollout(zh, past, np.zeros_like(tapes))
        shuffled = rollout(zh, past, tapes[rng.permutation(len(tapes))])
        target = true[:, ERROR_HORIZON-1]
        err, pers, zer, shu = (np.linalg.norm(f-target, axis=-1) for f in (roll[:, ERROR_HORIZON-1], zh[:, -1],
                                                                           zero[:, ERROR_HORIZON-1], shuffled[:, ERROR_HORIZON-1]))
        report['rollout_error'][name] = {c: dict(n=int(m.sum()), vs_persistence=float(np.median(err[m]/pers[m])),
                                                 vs_zero=float(np.median(err[m]/zer[m])), vs_shuffled=float(np.median(err[m]/shu[m])))
                                         for c in ('all',)+CATEGORIES for m in [np.ones(len(cats), bool) if c == 'all' else cats == c] if m.any()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    lw = sub.add_parser('lewm')
    lw.add_argument('--world-model', required=True)
    lw.add_argument('--cache', required=True)
    lw.add_argument('--out', required=True)
    a = p.parse_args()
    r = lewm(a.world_model, a.cache, a.out)
    for k, v in r['probe'].items():
        print(f"probe {k:32s} {v['mean']['median_xy_mm']:.1f} mm {v['mean']['rmse_yaw_deg']:.2f} deg (ridge {v['ridge']})")
    for k, v in r['ranking'].items():
        print(f"ranking {k}: top1 {v['top1']:.2f} mean rank {v['mean_rank']:.2f} ({v['matched']}/{v['decisions']})")
    for k, v in r['rollout_error'].items():
        print(f"rollout error {k}: " + ', '.join(f"{c} {x['vs_persistence']:.2f}/{x['vs_zero']:.2f}/{x['vs_shuffled']:.2f}" for c, x in v.items()))


if __name__ == '__main__':
    main()
