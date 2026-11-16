"""C001 train-only linear position probes, four-step predictions and controls.

White targets are sorted lexicographically at each time. The decoder never uses
test labels to choose a slot assignment, checkpoint, coefficient or regularizer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.data.balls import load as load_data
from go2nav.research import atomic_json, budget_remaining, digest, gpu_run, require_training_slot, reviewed_artifact
from go2nav.training.balls import load, to_pixels


def targets(red, white):
    order = np.lexsort((white[..., 1], white[..., 0]), axis=-1)
    ordered = np.take_along_axis(white, order[..., None], axis=-2)
    return np.concatenate((red, ordered.reshape(*red.shape[:-1], -1)), -1)


def correlation(x, y):
    x, y = x-x.mean(0), y-y.mean(0)
    denom = np.sqrt((x*x).sum(0)*(y*y).sum(0))
    return np.divide((x*y).sum(0), denom, out=np.zeros_like(denom), where=denom > 0)


def probe_summary(predicted, truth, episodes, repeats=2000):
    unique = np.unique(episodes)
    groups = [np.flatnonzero(episodes == e) for e in unique]
    rng = np.random.default_rng(2026111510)
    corr = correlation(predicted, truth)
    # Retain all windows from each sampled episode, never resample correlated windows independently.
    samples = []
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        samples.append(correlation(predicted[indices], truth[indices]))
    return dict(pearson=corr.tolist(), pearson_ci95=np.quantile(samples, [.025, .975], axis=0).T.tolist(),
                rmse_px=np.sqrt(np.mean((predicted-truth)**2, axis=0)).tolist(),
                minimum_pearson=float(corr.min()), mean_pearson=float(corr.mean()))


def fit_linear(x, y):
    """Fixed ridge alpha 1e-3 on train-standardized features, unpenalized intercept; GPU only in experiments."""
    x, y = x.double(), y.double()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-8)
    design = torch.cat(((x-mean)/std, x.new_ones(len(x), 1)), -1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype, device=x.device)*1e-3
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(design.T@design/len(x)+penalty, design.T@y/len(x))
    return mean, std, weights


def apply_linear(x, decoder):
    mean, std, weights = decoder
    z = (x.double()-mean)/std
    return torch.cat((z, z.new_ones(len(z), 1)), -1)@weights


@torch.no_grad()
def features(model, frames, actions, episodes, starts, batch=32, manifest=None):
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    result = {key: [] for key in ('current', 'rollout', 'shuffled', 'no_rollout')}
    for i in range(0, len(episodes), batch):
        if manifest is not None and budget_remaining(manifest) <= 0:
            return None
        e, t = episodes[i:i+batch, None], starts[i:i+batch, None]
        pixels = to_pixels(torch.from_numpy(np.array(frames[e, t+np.arange(-2, 1)])).to(device)).to(dtype)
        a = torch.from_numpy(np.array(actions[e, t+np.arange(-2, 4)])).to(device=device, dtype=dtype)/3
        z = model.encode(pixels)
        rolled = model.rollout(z, a, 4)[:, -1]
        shuffled = a.clone()
        # Fixed reversal of the four future actions, applied identically to every model and split.
        shuffled[:, 2:] = a[:, 2:].flip(1)
        altered = model.rollout(z, shuffled, 4)[:, -1]
        for key, values in (('current', z[:, -1]), ('no_rollout', z[:, -1]), ('rollout', rolled), ('shuffled', altered)):
            result[key].append(values.flatten(1).cpu())
    return {k: torch.cat(v) for k, v in result.items()}


def evaluate(root, n, checkpoint, out, research_id, parent_review):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['n'] != n:
        raise ValueError('Checkpoint ball count does not match probe data')
    field = 'checkpoint_sha256' if state['kind'] in ('o0', 'o1') else 'calibrated_sha256'
    reviewed_artifact(parent_review, checkpoint, field)
    require_training_slot()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with gpu_run(out, research_id, dict(stage='probes', n=n, train_windows=4096, test_windows_per_episode=5,
                 ridge_alpha=.001), [Path(checkpoint), Path(parent_review)]) as manifest, sdpa_kernel(SDPBackend.MATH):
        if state['kind'] in ('o0', 'o1'):
            from go2nav.eval.slots_inference import load_reviewed_world
            model = load_reviewed_world(checkpoint, parent_review)
        else:
            model, _, _ = load(checkpoint, 'cuda:0')
        dataset = {}
        for split in ('train', 'test'):
            frames, labels = load_data(root, n, split)
            rng = np.random.default_rng([2026111511, n, split == 'test'])
            episodes = rng.integers(len(frames), size=4096) if split == 'train' else np.repeat(np.arange(len(frames)), 5)
            starts = rng.integers(2, frames.shape[1]-4, size=len(episodes))
            x = features(model, frames, labels['actions'], episodes, starts, manifest=manifest)
            if x is None:
                atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete', split=split))
                return
            y = targets(labels['red'][episodes, starts], labels['white'][episodes, starts])
            future = targets(labels['red'][episodes, starts+4], labels['white'][episodes, starts+4])
            dataset[split] = (x, y, future, episodes)
        x, y, _, _ = dataset['train']
        decoder = fit_linear(x['current'].cuda(), torch.from_numpy(y).cuda())
        torch.save(dict(mean=decoder[0].cpu(), std=decoder[1].cpu(), weights=decoder[2].cpu(),
                        research_id=research_id, checkpoint_sha256=digest(checkpoint)), out/'probe.pt')
        x, current, future, episodes = dataset['test']
        report, predictions = {}, {}
        for key in x:
            predicted = apply_linear(x[key].cuda(), decoder).cpu().numpy()
            target = current if key == 'current' else future
            report[key] = probe_summary(predicted, target, episodes)
            predictions[key] = predicted
        np.savez_compressed(out/'cases.npz', episodes=episodes, current_truth=current, future_truth=future, **predictions)
        atomic_json(out/'result.json', dict(research_id=research_id, n=n, kind=state['kind'], measures=report,
                   cases_sha256=digest(out/'cases.npz'), status='awaiting_literature_review'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'out', 'research', 'parent-review'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--n', type=int, required=True)
    a = p.parse_args()
    evaluate(a.data, a.n, a.checkpoint, a.out, a.research, a.parent_review)


if __name__ == '__main__':
    main()
