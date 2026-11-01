"""The grid readout of the mover probe (docs/dynamic/PLAN.md, P4c): the patch head on a grid world model's 3 x 4 grid
tokens, beside the probe's ridge and MLP heads on the flattened state.

    python -m go2nav.eval.grid_readout PROBE_DIR --world-model LEWM.pt --train TRAIN.npz --eval EVAL.npz [--device cpu]

**Inputs:** the latents `mover_probe probe` cached in PROBE_DIR (features_{train,eval}.pt) for a grid world model.
Per window, the grid tokens of frames t-3..t, and for h > 0 also the state at t+h: the true one (`true`, the
encoder ceiling) or the rollout (`rollout`, trained on the executed tape's and applied to the executed, hold,
advance and retreat tapes'), as the probe's heads. **Head:** `mover_ceiling.patch_network` (its convolutional head,
one input frame per token frame) predicting the mover's position at t (h = 0) or its displacement to t+h, trained as the
patch-token probe (`patches_go2_v2`): `EPOCHS` epochs, dropout and weight decay as below, the epoch with the lowest
loss on a tenth of the training units held out. **Scores:** `mover_probe.score` and, for the rollouts,
`mover_probe.response` and `mover_probe.give_way`, in PROBE_DIR/results_grid.json under the heads `patch_true` and
`patch_rollout`.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from go2nav.eval import mover_ceiling as mc
from go2nav.eval import mover_probe as mp

EPOCHS, DROPOUT, WEIGHT_DECAY = 20, .3, .05


def grid_tokens(features, key, rows, j, grid, latent):
    """(n, frames, rows, columns, latent) float16: frames t-3..t, then (j not None) the state `key` at horizon j."""
    import torch
    history = features['zh'][rows]
    stack = history if j is None else torch.cat((history, features[key][rows, j][:, None]), 1)
    tokens = stack.reshape(len(rows), stack.shape[1], 1+grid[0]*grid[1], latent)[:, :, 1:]
    return tokens.reshape(len(rows), stack.shape[1], *grid, latent).numpy()


def fit(x, y, units, *, device, seed=mc.SEED, epochs=EPOCHS):
    """The patch head on x (n, frames, rows, columns, latent) -> y (n, 2), with epoch selection on held-out units;
    returns a predictor of (m, ...) -> (m, 2)."""
    import torch
    rng = np.random.default_rng(seed)
    held = rng.permutation(np.unique(units))[:max(1, int(len(np.unique(units))*mc.SELECT_SHARE))]
    select, rfit = np.flatnonzero(np.isin(units, held)), np.flatnonzero(~np.isin(units, held))
    yt = torch.from_numpy(y).float()
    mean, scale = yt[rfit].mean(0), yt[rfit].std(0).clamp_min(1e-6)
    torch.manual_seed(seed)
    net = mc.patch_network(*x.shape[2:5], 2, DROPOUT, frames=x.shape[1]).to(device).to(memory_format=torch.channels_last)
    steps = epochs*int(np.ceil(len(rfit)/mc.BATCH))
    opt = torch.optim.AdamW(net.parameters(), lr=mc.LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, mc.LR, total_steps=steps, pct_start=.05)
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    normalised = lambda rows: ((yt[rows]-mean)/scale).to(device)
    best = (np.inf, None)
    for _ in range(epochs):
        net.train()
        for rows, tokens in mc.batches(x, rfit, rng):
            with autocast():
                loss = ((net(torch.from_numpy(tokens).to(device)).float()-normalised(rows))**2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            sched.step()
        net.eval()
        held_loss = 0.
        with torch.inference_mode():
            for rows, tokens in mc.batches(x, select):
                with autocast():
                    held_loss += float(((net(torch.from_numpy(tokens).to(device)).float()-normalised(rows))**2)
                                       .mean())*len(rows)
        if held_loss < best[0]:
            best = (held_loss, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    net.eval()

    def predict(x):
        out = np.zeros((len(x), 2), np.float32)
        with torch.inference_mode():
            for rows, tokens in mc.batches(x, np.arange(len(x))):
                with autocast():
                    out[rows] = (net(torch.from_numpy(tokens).to(device)).float().cpu()*scale+mean).numpy()
        return out
    return predict


def readout(probe_dir, world_model, train, evaluation, *, device='cpu', threads=None):
    import torch
    from go2nav.models.lewm import load_world_model
    if threads:
        torch.set_num_threads(threads)
    config = load_world_model(world_model).config
    if not config.grid:
        raise ValueError(f'{world_model} has no grid')
    probe_dir = Path(probe_dir)
    dtrain, deval = dict(np.load(train)), dict(np.load(evaluation))
    ftrain, feval = (torch.load(probe_dir/f'features_{n}.pt', weights_only=False) for n in ('train', 'eval'))
    rtrain, reval = np.flatnonzero(ftrain['ok'].numpy()), np.flatnonzero(feval['ok'].numpy())
    split = mp.kinds(deval)[reval]
    tokens = lambda f, rows, key, j: grid_tokens(f, key, rows, j, config.grid, config.latent)
    results = dict(world_model=str(world_model), train_windows=int(len(rtrain)), eval_windows=int(len(reval)),
                   epochs=EPOCHS, dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
                   horizons_s=[0.]+[h/10 for h in mp.HORIZONS], heads={})
    for source in ('true', 'rollout'):
        rows_out = {s: [] for s in ('val', 'transfer')}
        for j in (None, *range(len(mp.HORIZONS))):
            if j is None and source == 'rollout':
                continue
            key = 'executed' if source == 'rollout' else 'true'
            y = dtrain['xy_t'][rtrain] if j is None else dtrain['target'][rtrain, j]-dtrain['xy_t'][rtrain]
            predict = fit(tokens(ftrain, rtrain, key, j), y.astype(np.float32), dtrain['unit'][rtrain], device=device)
            start = 0. if j is None else deval['xy_t'][reval]
            p = {tape: start+predict(tokens(feval, reval, tape, j))
                 for tape in ((key,) if source == 'true' or j is None else mp.TAPES)}
            for s in rows_out:
                m = split == s
                entry = dict(horizon_s=0. if j is None else mp.HORIZONS[j]/10, **mp.score(deval, reval[m], p[key][m], j))
                if source == 'rollout':
                    entry['response'] = mp.response(deval, reval[m], p['hold'][m], p['advance'][m], j)
                    entry['give_way'] = mp.give_way(deval, reval[m], {t: v[m] for t, v in p.items()}, j)
                rows_out[s].append(entry)
            results['heads'][f'patch_{source}'] = rows_out
            (probe_dir/'results_grid.json').write_text(json.dumps(results, indent=1)+'\n')
            print(json.dumps(dict(source=source, horizon=None if j is None else mp.HORIZONS[j],
                                  val=rows_out['val'][-1].get('median_m'))), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('probe_dir')
    parser.add_argument('--world-model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    readout(args.probe_dir, args.world_model, args.train, args.eval, device=args.device, threads=args.threads)


if __name__ == '__main__':
    main()
