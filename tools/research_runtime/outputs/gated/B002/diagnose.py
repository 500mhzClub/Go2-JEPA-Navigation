"""Bounded B002 CPU feasibility check of the unchanged B001 learned planner."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from go2nav.data.balls import load as load_data
from go2nav.eval.balls_planning import CEM_SEED, cap, learned_actions, task_inputs, task_start
from go2nav.research import atomic_json, digest, provenance, require_record, require_review, reviewed_artifact, verify_files
from go2nav.training.balls import load, to_pixels

BASE = Path('outputs/gated/B002')
NAME = 'l0_n3_s0'
torch.set_num_threads(8)
torch.set_num_interop_threads(1)


def guard():
    if any(os.environ.get(k) != '' for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']):
        raise ValueError('Hidden GPU devices required')
    if torch.cuda.is_initialized():
        raise ValueError('B002 must not initialize CUDA')
    if psutil.Process().memory_info().rss > 16*2**30:
        raise MemoryError('B002 16GiB RSS limit')


def finite(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite or non-CPU tensor')
    elif not np.isfinite(value).all():
        raise ValueError('Nonfinite values')


def lineage():
    _, rh = require_record('B002')
    parent = Path('outputs/gated/A003_recovered')/NAME
    checkpoint = parent/'calibrated.pt'
    evidence = reviewed_artifact(parent/'review.json', checkpoint, 'calibrated_sha256')
    audit = Path('outputs/gated/A015/finals')/NAME
    assert require_review(audit/'review.json')['status'] == 'passed'
    final = json.loads((audit/'result.json').read_text())
    assert final['step'] == 303000 and evidence['source_sha256'] == final['model_sha256']
    op = Path('outputs/gated/A016/runs')/NAME/'review.json'
    assert require_review(op)['status'] == 'passed'
    config = Path('outputs/balls/recovery_A014')/NAME/'config.json'
    am = json.loads((audit/'manifest.json').read_text())
    assert digest(config) == am['snapshot_sha256']['config.json']
    cfg = json.loads(config.read_text())
    assert (cfg['n'], cfg['kind'], cfg['seed']) == (3, 'l0', 0)
    probe = Path('outputs/gated/probes')/NAME
    pr = require_review(probe/'review.json')
    assert pr['status'] == 'passed'
    presult = json.loads((probe/pr['result']).read_text())
    assert presult['checkpoint_sha256'] == digest(checkpoint)
    assert presult['manifest_sha256'] == digest(probe/'manifest.json')
    pm = json.loads((probe/'manifest.json').read_text())
    pre = Path('outputs/gated/C006')/f'pre_{NAME}.json'
    assert pm['inputs'][str(pre)] == digest(pre)
    bound = json.loads(pre.read_text())['data_sha256']
    files = [f'outputs/data/balls/n3/test/{f}' for f in ['frames.npy', 'labels.npz', 'meta.json']]
    data_hashes = {f: bound[f] for f in files}
    verify_files(Path.cwd(), data_hashes)
    controls = Path('outputs/gated/B001/n3')
    assert require_review(controls/'review.json')['status'] == 'passed'
    cr = json.loads((controls/'result.json').read_text())
    assert cr['controls_passed'] and cr['n'] == 3 and cr['cem_seed'] == CEM_SEED
    cm = json.loads((controls/'manifest.json').read_text())
    verify_files(Path.cwd(), cm['inputs'])
    inputs = [checkpoint, parent/'review.json', audit/'review.json', config, op,
              probe/'review.json', probe/'manifest.json', pre, controls/'review.json', controls/'result.json']
    inputs += [Path(f) for f in files]
    return rh, checkpoint, inputs, data_hashes, cfg


@torch.no_grad()
def candidate_check(model, inputs):
    h = 8
    z = model.encode(to_pixels(torch.from_numpy(inputs.history)).double()[None])
    goal = model.encode(to_pixels(torch.from_numpy(inputs.goal)).double()[None, None])[:, 0]
    past = torch.from_numpy(inputs.history_actions).double()[None]/3
    rng = np.random.default_rng([CEM_SEED, 3, 0, h])
    proposals = cap(np.full((h, 2), 2., np.float32)*rng.standard_normal((100, h, 2)).astype(np.float32))

    def evaluate(a):
        actions = torch.from_numpy(a).double()/3
        joined = torch.cat((past.expand(len(a), -1, -1), actions), 1)
        pred = model.rollout(z.expand(len(a), -1, -1), joined, h)[:, -1]
        cost = (pred-goal).square().sum(-1)
        finite(pred); finite(cost)
        return pred, cost

    pred, cost = evaluate(proposals)
    singles = [evaluate(proposals[i:i+1]) for i in range(8)]
    single_pred = torch.cat([x[0] for x in singles]); single_cost = torch.cat([x[1] for x in singles])
    reverse_pred, reverse_cost = evaluate(proposals[::-1].copy())
    for actual, expected in [(pred[:8], single_pred), (cost[:8], single_cost),
                             (pred, reverse_pred.flip(0)), (cost, reverse_cost.flip(0))]:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    checks = dict(single_latent_max_abs=float((pred[:8]-single_pred).abs().max()),
                  single_cost_max_abs=float((cost[:8]-single_cost).abs().max()),
                  reversed_latent_max_abs=float((pred-reverse_pred.flip(0)).abs().max()),
                  reversed_cost_max_abs=float((cost-reverse_cost.flip(0)).abs().max()))
    return checks, dict(proposals=proposals, latents=pred.numpy(), costs=cost.numpy(),
                        singleton_latents=single_pred.numpy(), singleton_costs=single_cost.numpy())


def software():
    guard(); started = time.monotonic(); rh, _, _, _, _ = lineage()
    out = BASE/'software'; out.mkdir(parents=True, exist_ok=False)
    checks = {}
    assert all(2 <= task_start(3, e, h) < 100-h for e in range(1000) for h in [4, 6, 8])
    checks['all_fixed_task_bounds'] = True
    frames = np.broadcast_to(np.arange(100, dtype=np.uint8)[None, :, None, None, None], (1, 100, 2, 2, 3)).copy()
    actions = np.arange(200, dtype=np.float32).reshape(1, 100, 2)/100

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            self.encodes, self.rollouts = [], []

        def encode(self, x):
            self.encodes.append(x.clone())
            return x.mean((-3, -2, -1))[..., None].expand(-1, -1, 2)

        def rollout(self, z, a, steps):
            self.rollouts.append((z.clone(), a.clone(), steps))
            return z[:, -1:] + a[:, -steps:].cumsum(1)

    for h in [4, 6, 8]:
        t = task_start(3, 0, h); inputs = task_inputs(frames, actions, 0, t, h)
        assert np.array_equal(inputs.history, frames[0, t-2:t+1])
        assert np.array_equal(inputs.history_actions, actions[0, t-2:t])
        assert np.array_equal(inputs.goal, frames[0, t+h])
        model = Recorder(); seed = [CEM_SEED, 3, 0, h]
        output = learned_actions(model, inputs, h, seed)
        assert len(model.encodes) == 2 and [x.shape[1] for x in model.encodes] == [3, 1]
        torch.testing.assert_close(model.encodes[0], to_pixels(torch.from_numpy(inputs.history)).double()[None])
        torch.testing.assert_close(model.encodes[1], to_pixels(torch.from_numpy(inputs.goal)).double()[None, None])
        assert len(model.rollouts) == 20
        for z, a, steps in model.rollouts:
            assert steps == h and z.shape == (100, 3, 2) and a.shape == (100, h+2, 2)
            torch.testing.assert_close(a[:, :2], torch.from_numpy(inputs.history_actions).double()[None].expand(100, -1, -1)/3)
        finite(output); assert output.shape == (h, 2) and np.linalg.norm(output, axis=-1).max() <= 3+1e-6
        np.testing.assert_array_equal(output, learned_actions(Recorder(), inputs, h, seed))
    checks['exact_history_actions_goal_and_no_intermediate_frames'] = True
    checks['twenty_batches_hundred_candidates_preserved_history_all_horizons'] = True
    checks['finite_capped_deterministic_actions'] = True
    try:
        finite(torch.tensor([float('nan')]))
        raise AssertionError('Nonfinite not rejected')
    except ValueError:
        pass
    old = os.environ['CUDA_VISIBLE_DEVICES']; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        try:
            guard(); raise AssertionError('Visible GPU not rejected')
        except ValueError:
            pass
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = old
    checks['nonfinite_and_hidden_device_guards'] = True
    assert digest('go2nav/eval/balls_planning.py') == '737da767014a0e524373764346edf4de6f4e071b374d8f2f5c27b2b4a4d235b3'
    checks['actual_reviewed_final_lineage_data_and_unchanged_planner'] = True
    guard()
    result = dict(research_id='B002', status='awaiting_literature_review', research_sha256=rh,
                  checks=checks, elapsed_seconds=time.monotonic()-started, cuda_initialized=False,
                  source_sha256={f: digest(BASE/f) for f in ['diagnose.py', 'run_one.py']},
                  repository_source_sha256={str(p): digest(p) for p in sorted(Path('go2nav').rglob('*.py'))},
                  a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'),
                  limitation='Fake-model execution and actual lineage checks; no trained-model planning or GPU experiment.')
    atomic_json(out/'result.json', result)
    print(json.dumps(dict(checks=checks, elapsed_seconds=result['elapsed_seconds'])), flush=True)


def actual():
    guard(); started = time.monotonic(); rh, checkpoint, inputs, data_hashes, cfg = lineage()
    assert require_review(BASE/'software/review.json')['status'] == 'passed'
    sw = json.loads((BASE/'software/result.json').read_text())
    assert sw['research_sha256'] == rh
    verify_files(BASE, sw['source_sha256']); verify_files(Path.cwd(), sw['repository_source_sha256'])
    out = BASE/'preflight_l0_n3_s0'; out.mkdir(parents=True, exist_ok=False)
    manifest = provenance('B002', dict(name=NAME, device='cpu', dtype='float64', episode=0, horizons=[4, 6, 8]), inputs)
    manifest.update(software_review_sha256=digest(BASE/'software/review.json'), helper_sha256=sw['source_sha256'])
    atomic_json(out/'manifest.json', manifest)
    result = dict(research_id='B002', name=NAME, status='awaiting_literature_review', episode=0,
                  checkpoint_sha256=digest(checkpoint), device='cpu', dtype='float64')
    try:
        with sdpa_kernel(SDPBackend.MATH):
            model, _, state = load(checkpoint, 'cpu')
            assert (state['n'], state['kind'], state['step']) == (3, 'l0', 303000)
            assert state['inference_only'] and state['inference_dtype'] == 'float64' and state['attention_backend'] == 'math'
            assert state['train'] == cfg['train']
            frozen = {k: v.clone() for k, v in model.state_dict().items()}
            for v in frozen.values(): finite(v)
            frames, labels = load_data('outputs/data/balls', 3, 'test')
            assert frames.shape[:2] == (1000, 100)
            plans = {}
            for h in [4, 6, 8]:
                guard(); t = task_start(3, 0, h)
                model_inputs = task_inputs(frames, labels['actions'], 0, t, h)
                began = time.perf_counter()
                actions = learned_actions(model, model_inputs, h, [CEM_SEED, 3, 0, h])
                latency = time.perf_counter()-began
                finite(actions); assert np.linalg.norm(actions, axis=-1).max() <= 3+1e-6
                plans[str(h)] = dict(start_frame=t, goal_frame=t+h, actions=actions.tolist(), latency_seconds=latency)
                print(json.dumps(dict(horizon=h, latency_seconds=latency)), flush=True)
            repeated = learned_actions(model, model_inputs, 8, [CEM_SEED, 3, 0, 8])
            np.testing.assert_allclose(repeated, plans['8']['actions'], rtol=1e-4, atol=1e-5)
            checks, arrays = candidate_check(model, model_inputs)
            np.savez_compressed(out/'candidate_diagnostic.npz', **arrays)
            assert all(torch.equal(v, frozen[k]) for k, v in model.state_dict().items())
            assert digest(checkpoint) == result['checkpoint_sha256']
            verify_files(Path.cwd(), data_hashes); verify_files(Path.cwd(), sw['repository_source_sha256']); guard()
            estimate = sum(v['latency_seconds'] for v in plans.values())*1000
            result.update(plans=plans, numerical_checks=checks, repeated_actions_max_abs=float(np.max(np.abs(repeated-np.array(plans['8']['actions'])))),
                          projected_full_model_seconds=estimate, cpu_full_run_eligible=estimate <= 6900,
                          weights_unchanged=True, cuda_initialized=False, numerical_checks_passed=True,
                          diagnostic_sha256=digest(out/'candidate_diagnostic.npz'), manifest_sha256=digest(out/'manifest.json'),
                          elapsed_seconds=time.monotonic()-started,
                          limitation='One fixed episode for runtime and numerical feasibility only; no success-rate estimate. Future CPU full evaluation requires its own preregistration and this actual review.')
            atomic_json(out/'result.json', result); print(json.dumps(result), flush=True)
    except BaseException as exc:
        result.update(status='technical_blocked', error=repr(exc), elapsed_seconds=time.monotonic()-started)
        atomic_json(out/'result.json', result); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--software', action='store_true')
    software() if p.parse_args().software else actual()
