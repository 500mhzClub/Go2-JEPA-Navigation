"""A007: bounded validation-only probes of frozen intermediate n3 checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.data.balls import load as load_data
from go2nav.eval.balls_probes import apply_linear, fit_linear, probe_summary, targets
from go2nav.research import atomic_json, device_hours, digest, gpu_run, reviewed_artifact
from go2nav.training.balls import load, to_pixels


def sample(episodes, frames, split):
    if split not in ('train', 'val'):
        raise ValueError('A007 never reads the test split')
    rng = np.random.default_rng([2026111518, 3, split == 'val'])
    e = rng.integers(episodes, size=4096) if split == 'train' else np.repeat(np.arange(episodes), 5)
    return e, rng.integers(2, frames-4, size=len(e))


def action_controls(actions):
    reverse, zero = actions.clone(), actions.clone()
    reverse[:, 2:] = actions[:, 2:].flip(1)
    zero[:, 2:] = 0
    return dict(rollout=actions, reversed=reverse, zero=zero)


def check_budget(manifest):
    rows = json.loads(Path(manifest['gpu_ledger']).read_text())
    if device_hours(rows) >= 48 or device_hours([r for r in rows if r['research_id'] == 'A007']) >= .25:
        raise TimeoutError('A007 diagnostic budget exhausted')


@torch.no_grad()
def encode(model, frames, actions, episodes, starts, *, future, batch=32, manifest=None):
    device = next(model.parameters()).device
    output = {key: [] for key in (('current', 'rollout', 'reversed', 'zero', 'no_rollout') if future else ('current',))}
    for i in range(0, len(episodes), batch):
        if manifest is not None:
            check_budget(manifest)
        e, t = episodes[i:i+batch, None], starts[i:i+batch, None]
        offsets = np.arange(-2, 1) if future else np.array([0])
        pixels = to_pixels(torch.from_numpy(np.array(frames[e, t+offsets])).to(device)).double()
        z = model.encode(pixels)
        output['current'].append(z[:, -1].cpu())
        if future:
            a = torch.from_numpy(np.array(actions[e, t+np.arange(-2, 4)])).to(device).double()/3
            output['no_rollout'].append(z[:, -1].cpu())
            for key, control in action_controls(a).items():
                output[key].append(model.rollout(z, control, 4)[:, -1].cpu())
    result = {k: torch.cat(v) for k, v in output.items()}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite interim features')
    return result


def paired_gain(baseline, predicted, truth, episodes):
    """Positive gain means factual rollout beats the specified control in squared pixels."""
    delta = ((baseline-truth)**2-(predicted-truth)**2).reshape(len(truth), 4, 2).sum(-1)
    by_episode = np.stack([delta[episodes == e].mean(0) for e in np.unique(episodes)])
    rng = np.random.default_rng(2026111518)
    draws = by_episode[rng.integers(len(by_episode), size=(2000, len(by_episode)))].mean(1)
    return {name: dict(mean=float(by_episode[:, idx].mean()),
                      ci95=np.quantile(draws[:, idx].mean(-1), [.025, .975]).tolist())
            for name, idx in (('red', [0]), ('white', [1, 2, 3]))}


@torch.no_grad()
def evaluate(root, checkpoint, parent_review, out):
    reviewed_artifact(parent_review, checkpoint, 'calibrated_sha256')
    checkpoint_hash = digest(checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with gpu_run(out, 'A007', dict(stage='validation_probes', train_windows=4096,
                 validation_windows_per_episode=5, ridge_alpha=.001),
                 [Path(checkpoint), Path(parent_review)]) as manifest, sdpa_kernel(SDPBackend.MATH):
        torch.cuda.set_per_process_memory_fraction(.10)
        model, _, state = load(checkpoint, 'cuda:0')
        if state['n'] != 3 or state['kind'] not in ('l0', 'l1') or state.get('inference_dtype') != 'float64':
            raise ValueError('A007 requires calibrated float64 n3 L0/L1 checkpoints')
        weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        try:
            train_frames, train_labels = load_data(root, 3, 'train')
            te, tt = sample(len(train_frames), train_frames.shape[1], 'train')
            train = encode(model, train_frames, train_labels['actions'], te, tt, future=False, manifest=manifest)['current']
            y = targets(train_labels['red'][te, tt], train_labels['white'][te, tt])
            decoder = fit_linear(train.cuda(), torch.from_numpy(y).cuda())
            design = torch.cat(((train.cuda()-decoder[0])/decoder[1], torch.ones(len(train), 1, device='cuda', dtype=torch.float64)), -1)
            penalty = torch.eye(design.shape[1], device='cuda', dtype=torch.float64)*.001
            penalty[-1, -1] = 0
            a, b = design.T@design/len(train)+penalty, design.T@torch.from_numpy(y).cuda().double()/len(train)
            residual = float(torch.linalg.vector_norm(a@decoder[2]-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
            if residual > 1e-8 or not all(torch.isfinite(v).all() for v in decoder):
                raise ValueError('Invalid ridge solution')
            frames, labels = load_data(root, 3, 'val')
            ve, vt = sample(len(frames), frames.shape[1], 'val')
            x = encode(model, frames, labels['actions'], ve, vt, future=True, manifest=manifest)
            # Check actual diagnostic batch grouping, including all rollout controls.
            single = encode(model, frames, labels['actions'], ve[:8], vt[:8], future=True, batch=1)
            reverse = encode(model, frames, labels['actions'], ve[:8][::-1].copy(), vt[:8][::-1].copy(), future=True, batch=8)
            grouping = {}
            for key in x:
                torch.testing.assert_close(x[key][:8], single[key], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(x[key][:8], reverse[key].flip(0), rtol=1e-4, atol=1e-5)
                grouping[key] = dict(passed=True, single_max_abs=float((x[key][:8]-single[key]).abs().max()),
                                     reversed_max_abs=float((x[key][:8]-reverse[key].flip(0)).abs().max()))
            current = targets(labels['red'][ve, vt], labels['white'][ve, vt])
            future = targets(labels['red'][ve, vt+4], labels['white'][ve, vt+4])
            predictions = {key: apply_linear(value.cuda(), decoder).cpu().numpy() for key, value in x.items()}
            measures = {}
            for key, predicted in predictions.items():
                if not np.isfinite(predicted).all():
                    raise ValueError('Nonfinite probe prediction')
                truth = current if key == 'current' else future
                measures[key] = probe_summary(predicted, truth, ve)
                for label, idx in (('red', slice(0, 2)), ('white', slice(2, None))):
                    measures[key][label+'_mean_pearson'] = float(np.mean(measures[key]['pearson'][idx]))
                    measures[key][label+'_position_rmse_px'] = float(np.sqrt(np.mean(
                        ((predicted[:, idx]-truth[:, idx])**2).reshape(len(truth), -1, 2).sum(-1))))
            gains = {key: paired_gain(predictions[key], predictions['rollout'], future, ve)
                     for key in ('no_rollout', 'zero', 'reversed')}
            unchanged = digest(checkpoint) == checkpoint_hash and all(torch.equal(v.cpu(), weights[k]) for k, v in model.state_dict().items())
            if not unchanged:
                raise ValueError('Frozen checkpoint or learned state changed')
            np.savez_compressed(out/'cases.npz', episodes=ve, starts=vt, current_truth=current,
                                future_truth=future, **predictions)
            torch.save(dict(mean=decoder[0].cpu(), std=decoder[1].cpu(), weights=decoder[2].cpu(),
                            checkpoint_sha256=checkpoint_hash), out/'probe.pt')
            atomic_json(out/'result.json', dict(research_id='A007', status='awaiting_literature_review',
                        n=3, kind=state['kind'], step=state['step'], checkpoint_sha256=checkpoint_hash,
                        train_windows=len(te), validation_windows=len(ve), validation_episodes=len(np.unique(ve)),
                        numerical_grouping=grouping, ridge_relative_residual=residual,
                        weights_unchanged=unchanged, measures=measures, paired_squared_error_gains=gains,
                        cases_sha256=digest(out/'cases.npz')))
        except TimeoutError as exc:
            atomic_json(out/'result.json', dict(research_id='A007', status='budget_incomplete', error=str(exc)))
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'parent-review', 'out'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    evaluate(a.data, a.checkpoint, a.parent_review, a.out)


if __name__ == '__main__':
    main()
