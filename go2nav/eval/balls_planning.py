"""B001: fixed, paired balls tasks, optimizer controls and latent planning.

python -m go2nav.eval.balls_planning --data DATA --n 3 --out NEW --research B001
Default arms are recorded/direct/true_cem/hold/random. Learned evaluation requires
an explicitly reviewed inference artifact and passing controls from the same tasks.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.data import balls as data
from go2nav.research import (atomic_json, budget_remaining, digest, gpu_run, provenance, require_record,
                            require_review, reviewed_artifact, require_training_slot)
from go2nav.sim import balls as world
from go2nav.training.balls import load, to_pixels

TASK_SEED, CEM_SEED = 2026111503, 2026111504


@dataclass(frozen=True)
class Inputs:
    """Only observations and already executed actions enter a learned planner."""
    history: np.ndarray
    history_actions: np.ndarray
    goal: np.ndarray


def task_start(n, episode, horizon, length=100):
    return int(np.random.default_rng((TASK_SEED, n, episode, horizon)).integers(2, length-horizon))


def task_inputs(frames, actions, episode, start, horizon):
    return Inputs(np.array(frames[episode, start-2:start+1]),
                  np.array(actions[episode, start-2:start]), np.array(frames[episode, start+horizon]))


def cap(actions):
    return actions*np.minimum(1, 3/np.maximum(np.linalg.norm(actions, axis=-1, keepdims=True), 1e-12))


def red_path(start, actions):
    x, path = np.broadcast_to(start, actions.shape[:-2]+(2,)).copy(), []
    for t in range(actions.shape[-2]):
        x = np.clip(x+cap(actions[..., t, :]), world.LOW, world.HIGH)
        path.append(x.copy())
    return np.stack(path, -2)


def cem(cost, horizon, seed, candidates=100, elites=10, iterations=20, std=2.):
    """Diagonal CEM, paired random stream; execute the final elite mean (C-JEPA Appendix G)."""
    rng = np.random.default_rng(seed)
    mean, scale = np.zeros((horizon, 2), np.float32), np.full((horizon, 2), std, np.float32)
    for _ in range(iterations):
        proposals = cap(mean+scale*rng.standard_normal((candidates, horizon, 2)).astype(np.float32))
        values = np.asarray(cost(proposals))
        if values.shape != (candidates,) or not np.isfinite(values).all():
            raise FloatingPointError('Invalid CEM costs')
        order = np.argsort(values, kind='stable')
        selected = proposals[order[:elites]]
        mean, scale = selected.mean(0), np.maximum(selected.std(0), 1e-3)
    return cap(mean)


def slot_distance(predicted, goal):
    """Sum squared latent distance after optimal assignment; no object labels."""
    costs = (predicted[..., :, None, :]-goal[..., None, :, :]).square().sum(-1)
    shape = costs.shape[:-2]
    flat = costs.reshape(-1, *costs.shape[-2:])
    values = []
    for matrix in flat:
        row, col = linear_sum_assignment(matrix.detach().cpu().numpy())
        values.append(matrix[row, col].sum())
    return torch.stack(values).reshape(shape)


@torch.no_grad()
def learned_actions(model, inputs, horizon, seed):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    history = model.encode(to_pixels(torch.from_numpy(inputs.history).to(device)).to(dtype)[None])
    # Goal is encoded in an independent invocation; no carry/history can reveal the intervening frames.
    goal = model.encode(to_pixels(torch.from_numpy(inputs.goal).to(device)).to(dtype)[None, None])[:, 0]
    past = torch.from_numpy(inputs.history_actions).to(device=device, dtype=dtype)[None]/3

    def cost(actions):
        future = torch.from_numpy(actions).to(device=device, dtype=dtype)/3
        joined = torch.cat((past.expand(len(future), -1, -1), future), 1)
        z = history.expand(len(future), *history.shape[1:])
        predicted = model.rollout(z, joined, horizon)[:, -1]
        loss = slot_distance(predicted, goal) if predicted.ndim == 3 else (predicted-goal).square().sum(-1)
        return loss.cpu().numpy()
    return cem(cost, horizon, seed)


def score(start, goal, actions, white):
    path = red_path(start, actions)
    error = float(np.linalg.norm(path[-1]-goal))
    collision = bool((np.linalg.norm(path[:, None]-white, axis=-1) < 2*world.RADIUS).any())
    return dict(endpoint_error=error, success=error <= 1., collision=collision,
                collision_free_success=error <= 1. and not collision)


def summarize(rows):
    summary = {}
    for arm in sorted({r['arm'] for r in rows}):
        summary[arm] = {}
        for h in sorted({r['horizon'] for r in rows}):
            cases = [r for r in rows if r['arm'] == arm and r['horizon'] == h]
            values = np.array([r['success'] for r in cases], float)
            rng = np.random.default_rng(2026111505)
            # One task per episode at this horizon: resampling rows is episode-cluster resampling.
            bootstrap = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            summary[arm][str(h)] = dict(count=len(cases), success=float(values.mean()),
                success_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                endpoint_median=float(np.median([r['endpoint_error'] for r in cases])),
                endpoint_mean=float(np.mean([r['endpoint_error'] for r in cases])),
                collision_free_success=float(np.mean([r['collision_free_success'] for r in cases])),
                latency_mean_s=float(np.mean([r['latency_s'] for r in cases])),
                groups={str(g): float(np.mean([r['success'] for r in cases if r['group'] == g]))
                        for g in sorted({r['group'] for r in cases})})
    controls_passed = all(arm in summary and all(v['success'] >= threshold for v in summary[arm].values())
                          for arm, threshold in (('recorded', 1.), ('direct', 1.), ('true_cem', .95)))
    return dict(arms=summary, controls_passed=controls_passed)


def evaluate(root, n, out, research_id, *, checkpoint=None, inference_review=None, controls=None):
    record, _ = require_record(research_id)
    out = Path(out)
    if out.exists():
        raise FileExistsError('Use a new output directory to preserve prior evidence')
    frames, labels = data.load(root, n, 'test')
    if len(frames) != record['protocol']['episodes_per_n']:
        raise ValueError('Test count disagrees with preregistration')
    inputs = [Path(root)/f'n{n}'/'test'/'meta.json']
    model = None
    if checkpoint:
        require_training_slot()
        review = require_review(inference_review)
        result = json.loads((Path(inference_review).parent/review['result']).read_text())
        if review['status'] != 'passed' or digest(checkpoint) not in (result.get('calibrated_sha256'), result.get('checkpoint_sha256')):
            raise ValueError('Checkpoint has no passing inference review')
        control_review = require_review(Path(controls).with_name('review.json'), controls)
        control = json.loads(Path(controls).read_text())
        if control_review['status'] != 'passed' or not control['controls_passed'] or control['n'] != n \
                or control['task_seed'] != TASK_SEED or control['cem_seed'] != CEM_SEED:
            raise ValueError('Matching simulator controls must pass first')
        control_manifest = json.loads(Path(controls).with_name('manifest.json').read_text())
        if digest(inputs[0]) not in control_manifest['inputs'].values():
            raise ValueError('Test data metadata changed since control evaluation')
        inputs += [Path(checkpoint), Path(inference_review), Path(controls)]
        with gpu_run(out, research_id, record['protocol'], inputs) as manifest:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['n'] != n:
                raise ValueError('Checkpoint ball count does not match evaluation tasks')
            if state['kind'] in ('o0', 'o1'):
                from go2nav.eval.slots_inference import load_reviewed_world
                model = load_reviewed_world(checkpoint, inference_review)
            else:
                model, _, _ = load(checkpoint, 'cuda:0')
            with sdpa_kernel(SDPBackend.MATH):
                _evaluate(frames, labels, n, out, research_id, model, manifest, state['kind'])
    else:
        out.mkdir(parents=True)
        manifest = provenance(research_id, record['protocol'], inputs)
        atomic_json(out/'manifest.json', manifest)
        _evaluate(frames, labels, n, out, research_id, model, manifest)


def _evaluate(frames, labels, n, out, research_id, model, manifest, kind='learned'):
    rows = []
    arms = [kind] if model is not None else ['recorded', 'direct', 'true_cem', 'hold', 'random']
    with (out/'cases.jsonl').open('w') as f:
        for episode in range(len(frames)):
            if model is not None and budget_remaining(manifest) <= 0:
                atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete', episodes=episode))
                return
            for h in (4, 6, 8):
                t = task_start(n, episode, h, frames.shape[1])
                start, goal = labels['red'][episode, t], labels['red'][episode, t+h]
                seed = [CEM_SEED, n, episode, h]
                for arm in arms:
                    began = time.perf_counter()
                    if arm == 'recorded':
                        actions = labels['actions'][episode, t:t+h]
                    elif arm == 'direct':
                        actions = np.broadcast_to((goal-start)/h, (h, 2)).copy()
                    elif arm == 'true_cem':
                        actions = cem(lambda a: ((red_path(start, a)[:, -1]-goal)**2).sum(-1), h, seed)
                    elif arm == 'hold':
                        actions = np.zeros((h, 2), np.float32)
                    elif arm == 'random':
                        actions = cap(np.random.default_rng(seed).normal(0, 2, (h, 2)).astype(np.float32))
                    else:
                        actions = learned_actions(model, task_inputs(frames, labels['actions'], episode, t, h), h, seed)
                    row = dict(research_id=research_id, n=n, episode=episode, group=episode//200,
                               horizon=h, start_frame=t, goal_frame=t+h, start_position=start.tolist(),
                               goal_position=goal.tolist(), arm=arm, seed=seed, actions=actions.tolist(),
                               latency_s=time.perf_counter()-began,
                               **score(start, goal, actions, labels['white'][episode, t+1:t+h+1]))
                    f.write(json.dumps(row)+'\n')
                    rows.append(row)
            if episode % 100 == 0:
                f.flush()
                print(f'n{n}: {episode+1}/{len(frames)} episodes', flush=True)
    result = summarize(rows) | dict(research_id=research_id, n=n, task_seed=TASK_SEED, cem_seed=CEM_SEED,
                                  cases_sha256=digest(out/'cases.jsonl'), status='awaiting_literature_review')
    atomic_json(out/'result.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--n', type=int, choices=[1, 3, 6], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--research', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--inference-review')
    p.add_argument('--controls')
    a = p.parse_args()
    evaluate(a.data, a.n, a.out, a.research, checkpoint=a.checkpoint,
             inference_review=a.inference_review, controls=a.controls)


if __name__ == '__main__':
    main()
