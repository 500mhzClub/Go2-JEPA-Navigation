"""C004: real-input numerical validation of immutable selected slot checkpoints."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from go2nav.eval.balls_inference import difference
from go2nav.eval.balls_planning import cap
from go2nav.eval.slots_recovery import finite_values, numerics
from go2nav.models.balls_slots import ObjectMaskedPredictor
from go2nav.research import (atomic_json, budget_remaining, digest, gpu_run, require_record,
                            require_training_slot, reviewed_artifact, verify_files)
from go2nav.training.balls import to_pixels
from go2nav.training.balls_slots import load_encoder

SEED = 2026111516


def windows(shape, n, split, count=16):
    if split not in ('train', 'val') or shape[1] < 4:
        raise ValueError('Four-frame train/val windows are required')
    starts = shape[1]-3
    flat = np.random.default_rng((SEED, n, ('train', 'val').index(split))).choice(shape[0]*starts, count, replace=False)
    return flat//starts, flat % starts


def cache_parent(state, data):
    cache = Path(state['cache'])
    if digest(cache/'index.json') != state['cache_sha256']:
        raise ValueError('Predictor cache index changed')
    index = json.loads((cache/'index.json').read_text())
    if (index['n'], index['seed']) != (state['n'], state['seed']):
        raise ValueError('Predictor cache n/seed mismatch')
    if digest(index['encoder']) != index['encoder_sha256']:
        raise ValueError('Predictor encoder changed')
    verify_files(cache, {f'{s}.npy': h for s, h in index['splits'].items()})
    verify_files(Path(data)/f"n{state['n']}", index['data_metadata'])
    return cache, index


@torch.no_grad()
def rollout_grouping(model, history, actions, check_budget=lambda: None):
    outputs = {}
    for mode in ('batched', 'repeat', 'reversed', 'separate'):
        check_budget()
        if mode == 'separate':
            parts = []
            for i in range(0, len(history), 2):
                check_budget()
                parts.append(model.rollout(history[i:i+2], actions[i:i+2], 8))
            values = torch.cat(parts)
        elif mode == 'reversed':
            values = model.rollout(history.flip(0), actions.flip(0), 8).flip(0)
        else:
            values = model.rollout(history, actions, 8)
        finite_values({mode: values}, 'planning batch')
        outputs[mode] = values
    comparisons = {k: difference(outputs['batched'], outputs[k]) for k in ('repeat', 'reversed', 'separate')}
    return dict(checks=comparisons, candidates=len(history), steps=8,
                passed=all(v['passed'] for v in comparisons.values()))


def run(data, checkpoint, parent_review, out, research_id, encoder_review=None):
    record, _ = require_record(research_id)
    if record['id'] != 'C004':
        raise ValueError('Selected-slot validation requires C004')
    require_training_slot()
    reviewed_artifact(parent_review, checkpoint)
    checkpoint, data, out = Path(checkpoint), Path(data), Path(out)
    original_hash = digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['kind'] not in ('slots', 'o0', 'o1'):
        raise ValueError('C004 only validates selected slot checkpoints')
    n, seed = state['n'], state['seed']
    parents = [checkpoint, Path(parent_review)]
    for split in ('train', 'val'):
        parents += [data/f'n{n}'/split/name for name in ('frames.npy', 'labels.npz', 'meta.json')]
    cache = index = None
    if state['kind'] != 'slots':
        cache, index = cache_parent(state, data)
        if encoder_review is None:
            raise ValueError('Predictor validation requires its C004 encoder review')
        evidence = reviewed_artifact(encoder_review, index['encoder'])
        if evidence.get('research_id') != 'C004' or not evidence.get('checks_passed') or evidence.get('kind') != 'slots':
            raise ValueError('The encoder needs passing selected-checkpoint numerical evidence')
        parents += [cache/'index.json', Path(index['encoder']), Path(encoder_review)]
    out.mkdir(parents=True, exist_ok=False)
    result = dict(research_id=research_id, n=n, seed=seed, kind=state['kind'], checkpoint_sha256=original_hash,
                  status='awaiting_literature_review', windows={}, planning_batch={})
    with gpu_run(out, research_id, record['protocol'], parents) as manifest:
        torch.cuda.set_per_process_memory_fraction(.8)
        start = time.monotonic()
        def check_budget():
            if budget_remaining(manifest) <= 0 or time.monotonic()-start >= record['resources']['gpu_hours_max']*3600:
                raise TimeoutError('Selected-checkpoint validation allowance exhausted')
        try:
            if state['kind'] == 'slots':
                model, _ = load_encoder(checkpoint)
                encoder, numerical_kind = model, 'rgb_slots'
            else:
                model = ObjectMaskedPredictor(**state['config']).cuda().eval()
                model.load_state_dict(state['model'])
                encoder, encoder_state = load_encoder(index['encoder'])
                numerical_kind = 'predictor'
            finite_values(model.state_dict(), 'selected weights')
            for split in ('train', 'val'):
                frames = np.load(data/f'n{n}'/split/'frames.npy', mmap_mode='r')
                with np.load(data/f'n{n}'/split/'labels.npz') as labels:
                    actions = labels['actions']
                episodes, starts = windows(frames.shape, n, split)
                states = np.load(cache/f'{split}.npy', mmap_mode='r') if cache else None
                if states is not None and states.shape != (*frames.shape[:2], n+2, 128):
                    raise ValueError('Frozen cache shape differs from the actual frame cohort')
                result['windows'][split] = []
                for i in range(0, len(episodes), 2):
                    check_budget()
                    e, t = episodes[i:i+2, None], starts[i:i+2, None]
                    cases = dict(episodes=e[:, 0].tolist(), starts=t[:, 0].tolist())
                    if numerical_kind == 'rgb_slots':
                        values = [to_pixels(torch.from_numpy(np.array(frames[e, t+np.arange(3)])))]
                    else:
                        z = torch.from_numpy(np.array(states[e, t+np.arange(4)]))
                        a = torch.from_numpy(np.array(actions[e, t+np.arange(3)]))/3
                        values = [z, a, torch.zeros(2, n+2, dtype=torch.bool)]
                    cases['unmasked'] = numerics(numerical_kind, model, check_budget, data=values)
                    if state['kind'] == 'o1':
                        values[-1][0, 1] = True
                        values[-1][1, -2:] = True
                        cases['masked'] = numerics(numerical_kind, model, check_budget, data=values)
                    result['windows'][split].append(cases)
                    atomic_json(out/'result.json', result)
                if numerical_kind == 'predictor':
                    check_budget()
                    e, t = int(episodes[0]), int(starts[0])
                    pixels = to_pixels(torch.from_numpy(np.array(frames[e:e+1, t:t+3])).cuda())
                    with torch.no_grad():
                        history = encoder(pixels).expand(100, -1, -1, -1)
                    finite_values({'history': history}, 'independently encoded history')
                    rng = np.random.default_rng((SEED, n, ('train', 'val').index(split), 100))
                    future = cap((2*rng.standard_normal((100, 8, 2))).astype(np.float32))
                    past = np.broadcast_to(actions[e, t:t+2], (100, 2, 2))
                    commands = torch.from_numpy(np.concatenate((past, future), 1)).cuda()/3
                    result['planning_batch'][split] = rollout_grouping(model, history, commands, check_budget)
                    atomic_json(out/'result.json', result)
            unchanged = all(torch.equal(value.cpu(), state['model'][name]) for name, value in model.state_dict().items())
            result.update(weights_unchanged=unchanged, checkpoint_unchanged=digest(checkpoint) == original_hash)
            encoder_unchanged = index is None or (digest(index['encoder']) == index['encoder_sha256'] and
                all(torch.equal(value.cpu(), encoder_state['model'][name]) for name, value in encoder.state_dict().items()))
            result['encoder_unchanged'] = encoder_unchanged
            checks = [value['passed'] for rows in result['windows'].values() for row in rows
                      for key, value in row.items() if key in ('unmasked', 'masked')]
            checks += [row['passed'] for row in result['planning_batch'].values()]
            result['checks_passed'] = unchanged and encoder_unchanged and result['checkpoint_unchanged'] and all(checks)
            atomic_json(out/'result.json', result)
            print(dict(kind=state['kind'], n=n, seed=seed, checks_passed=result['checks_passed']), flush=True)
        except TimeoutError:
            result.update(status='budget_incomplete', checks_passed=False)
            atomic_json(out/'result.json', result)
        except BaseException as exc:
            result.update(status='technical_blocked', checks_passed=False, error=repr(exc))
            atomic_json(out/'result.json', result)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'parent-review', 'out', 'research'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--encoder-review')
    a = p.parse_args()
    run(a.data, a.checkpoint, a.parent_review, a.out, a.research, a.encoder_review)


if __name__ == '__main__':
    main()
