"""B003: complete frozen B001 baseline planning on explicitly declared CPUs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, provenance, require_record, require_review, reviewed_artifact, verify_files

B002 = Path('outputs/gated/B002')
assert require_review(B002/'software/review.json')['status'] == 'passed'
verify_files(B002, json.loads((B002/'software/result.json').read_text())['source_sha256'])
spec = importlib.util.spec_from_file_location('b003_reviewed_cpu_guards', B002/'diagnose.py')
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
guard, finite = prior.guard, prior.finite
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from go2nav.data.balls import load as load_data
from go2nav.eval.balls_planning import TASK_SEED, CEM_SEED, cap, learned_actions, score, summarize, task_inputs, task_start
from go2nav.training.balls import load, to_pixels

BASE = Path('outputs/gated/B003')


def lineage(name):
    record, rh = require_record('B003'); assert name in record['protocol']['order']
    parent = Path('outputs/gated/A003_recovered')/name; checkpoint = parent/'calibrated.pt'
    evidence = reviewed_artifact(parent/'review.json', checkpoint, 'calibrated_sha256')
    audit = Path('outputs/gated/A015/finals')/name
    assert require_review(audit/'review.json')['status'] == 'passed'
    final = json.loads((audit/'result.json').read_text()); am = json.loads((audit/'manifest.json').read_text())
    assert final['step'] == 303000 and evidence['source_sha256'] == final['model_sha256']
    op = Path('outputs/gated/A016/runs')/name/'review.json'
    assert require_review(op)['status'] == 'passed'
    config = Path('outputs/balls/recovery_A014')/name/'config.json'
    assert digest(config) == am['snapshot_sha256']['config.json']
    cfg = json.loads(config.read_text()); n, kind, seed = cfg['n'], cfg['kind'], cfg['seed']
    assert seed == 0 and kind in ['l0', 'l1'] and name == f'{kind}_n{n}_s{seed}'
    probe = Path('outputs/gated/probes')/name
    pr = require_review(probe/'review.json'); assert pr['status'] in ['passed', 'measured_negative']
    presult = json.loads((probe/pr['result']).read_text())
    assert presult['checkpoint_sha256'] == digest(checkpoint) and presult['manifest_sha256'] == digest(probe/'manifest.json')
    pm = json.loads((probe/'manifest.json').read_text()); pre = Path('outputs/gated/C006')/f'pre_{name}.json'
    assert pm['inputs'][str(pre)] == digest(pre)
    bound = json.loads(pre.read_text())['data_sha256']
    files = [f'outputs/data/balls/n{n}/test/{f}' for f in ['frames.npy', 'labels.npz', 'meta.json']]
    data_hashes = {f: bound[f] for f in files}; verify_files(Path.cwd(), data_hashes)
    controls = Path(f'outputs/gated/B001/n{n}')
    assert require_review(controls/'review.json')['status'] == 'passed'
    cr = json.loads((controls/'result.json').read_text())
    assert cr['controls_passed'] and cr['n'] == n and cr['cem_seed'] == CEM_SEED and cr['task_seed'] == TASK_SEED
    assert cr['cases_sha256'] == digest(controls/'cases.jsonl')
    verify_files(Path.cwd(), json.loads((controls/'manifest.json').read_text())['inputs'])
    inputs = [checkpoint, parent/'review.json', audit/'review.json', config, op, probe/'review.json', probe/'manifest.json',
              pre, controls/'review.json', controls/'result.json', controls/'cases.jsonl'] + [Path(f) for f in files]
    return record, rh, checkpoint, inputs, data_hashes, cfg


def consultation(name, checkpoint, data_hashes, rh):
    p = BASE/f'pre_{name}.json'; value = json.loads(p.read_text())
    expected = dict(name=name, research_sha256=rh, b001_record_sha256=digest('docs/research/records/B001.json'),
                    checkpoint_sha256=digest(checkpoint), parent_review_sha256=digest(checkpoint.parent/'review.json'), data_sha256=data_hashes)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('B003 consultation identity/protocol/parent/data mismatch')
    if not value.get('consulted_at') or not value.get('sources') or not value.get('consequences'):
        raise ValueError('Missing substantive consultation')
    if any(not all(s.get(k) for k in ['url', 'version', 'sections', 'consequence']) for s in value['sources']):
        raise ValueError('Incomplete source consultation')
    return p


@torch.no_grad()
def numerical_check(model, inputs, n):
    z = model.encode(to_pixels(torch.from_numpy(inputs.history)).double()[None])
    goal = model.encode(to_pixels(torch.from_numpy(inputs.goal)).double()[None, None])[:, 0]
    past = torch.from_numpy(inputs.history_actions).double()[None]/3
    proposals = cap(2*np.random.default_rng([CEM_SEED, n, 0, 8]).standard_normal((100, 8, 2)).astype(np.float32))

    def calc(a):
        actions = torch.cat((past.expand(len(a), -1, -1), torch.from_numpy(a).double()/3), 1)
        pred = model.rollout(z.expand(len(a), -1, -1), actions, 8)[:, -1]
        cost = (pred-goal).square().sum(-1); finite(pred); finite(cost)
        return pred, cost

    pred, costs = calc(proposals); singles = [calc(proposals[i:i+1]) for i in range(8)]
    sp = torch.cat([v[0] for v in singles]); sc = torch.cat([v[1] for v in singles])
    rp, rc = calc(proposals[::-1].copy())
    pairs = [(pred[:8], sp), (costs[:8], sc), (pred, rp.flip(0)), (costs, rc.flip(0))]
    for a, b in pairs: torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    names = ['singleton_latents', 'singleton_costs', 'reversed_latents', 'reversed_costs']
    checks = {k: float((a-b).abs().max()) for k, (a, b) in zip(names, pairs)}
    arrays = dict(proposals=proposals, latents=pred.numpy(), costs=costs.numpy(),
                  singleton_latents=sp.numpy(), singleton_costs=sc.numpy())
    return checks, arrays


def cases(model, frames, labels, n, kind, research_id='B003'):
    for episode in range(len(frames)):
        guard()
        for h in [4, 6, 8]:
            t = task_start(n, episode, h, frames.shape[1]); seed = [CEM_SEED, n, episode, h]
            start, goal = labels['red'][episode, t], labels['red'][episode, t+h]
            began = time.perf_counter()
            actions = learned_actions(model, task_inputs(frames, labels['actions'], episode, t, h), h, seed)
            finite(actions)
            if np.linalg.norm(actions, axis=-1).max() > 3+1e-6: raise ValueError('Action cap violation')
            row = dict(research_id=research_id, n=n, episode=episode, group=episode//200, horizon=h,
                       start_frame=t, goal_frame=t+h, start_position=start.tolist(), goal_position=goal.tolist(),
                       arm=kind, seed=seed, actions=actions.tolist(), latency_s=time.perf_counter()-began,
                       **score(start, goal, actions, labels['white'][episode, t+1:t+h+1]))
            finite(row['endpoint_error']); yield row


def run(name):
    guard(); started = time.monotonic()
    record, rh, checkpoint, inputs, data_hashes, cfg = lineage(name)
    assert require_review(BASE/'software/review.json')['status'] == 'passed'
    sw = json.loads((BASE/'software/result.json').read_text()); assert sw['research_sha256'] == rh
    verify_files(BASE, sw['source_sha256']); verify_files(Path.cwd(), sw['repository_source_sha256'])
    pre = consultation(name, checkpoint, data_hashes, rh)
    out = Path('outputs/gated/planning')/name; out.mkdir(parents=True, exist_ok=False)
    n, kind, seed = cfg['n'], cfg['kind'], cfg['seed']
    manifest = provenance('B003', dict(name=name, n=n, kind=kind, seed=seed, device='cpu', dtype='float64',
                          b001_protocol=json.loads(Path('docs/research/records/B001.json').read_text())['protocol']), inputs+[pre])
    manifest.update(helper_sha256=sw['source_sha256'], software_review_sha256=digest(BASE/'software/review.json'))
    atomic_json(out/'manifest.json', manifest)
    result = dict(research_id='B003', n=n, kind=kind, seed=seed, device='cpu', status='awaiting_literature_review')
    rows = []
    try:
        with sdpa_kernel(SDPBackend.MATH):
            model, _, state = load(checkpoint, 'cpu')
            assert (state['n'], state['kind'], state['step']) == (n, kind, 303000)
            assert state['inference_only'] and state['inference_dtype'] == 'float64' and state['attention_backend'] == 'math'
            assert state['train'] == cfg['train']
            frozen = {k: v.clone() for k, v in model.state_dict().items()}
            for v in frozen.values(): finite(v)
            frames, labels = load_data('outputs/data/balls', n, 'test'); assert frames.shape[:2] == (1000, 100)
            t = task_start(n, 0, 8)
            numerical, arrays = numerical_check(model, task_inputs(frames, labels['actions'], 0, t, 8), n)
            np.savez_compressed(out/'numerics.npz', **arrays)
            atomic_json(out/'numerical_checks.json', numerical)
            with (out/'cases.jsonl').open('x') as f:
                for row in cases(model, frames, labels, n, kind):
                    f.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                    if row['horizon'] == 8:
                        f.flush()
                        atomic_json(out/'progress.json', dict(at=time.time(), episodes=row['episode']+1, cases=len(rows),
                            elapsed_seconds=time.monotonic()-started, last_horizon=8))
                        if row['episode'] % 25 == 0:
                            print(json.dumps(dict(episodes=row['episode']+1, total=1000)), flush=True)
            assert len(rows) == 3000
            assert all(torch.equal(v, frozen[k]) for k, v in model.state_dict().items())
            verify_files(Path.cwd(), data_hashes); verify_files(Path.cwd(), sw['repository_source_sha256']); guard()
            assert digest(checkpoint) == manifest['inputs'][str(checkpoint)]
            result.update(summarize(rows))
            result.update(task_seed=TASK_SEED, cem_seed=CEM_SEED, cases_sha256=digest(out/'cases.jsonl'),
                          checkpoint_sha256=digest(checkpoint), numerical_checks=numerical,
                          numerics_sha256=digest(out/'numerics.npz'), weights_unchanged=True, cuda_initialized=False,
                          parent_review_sha256=digest(checkpoint.parent/'review.json'), manifest_sha256=digest(out/'manifest.json'),
                          elapsed_seconds=time.monotonic()-started, complete_cases=3000,
                          controls_note='controls_passed in summarize concerns arms present in this output; matching B001 controls are separately reviewed inputs.')
            atomic_json(out/'result.json', result)
            print(json.dumps(dict(name=name, cases=3000, status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    except BaseException as exc:
        result.update(status='technical_blocked', error=repr(exc), completed_cases=len(rows), elapsed_seconds=time.monotonic()-started)
        atomic_json(out/'result.json', result); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); run(p.parse_args().run)
