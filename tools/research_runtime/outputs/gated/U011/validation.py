"""Full64 selected encoder/O0/O1 validation using the deployed inference loader."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from common import ORIGINAL as BASE, ORIGINAL_HELPERS as HELPERS, code_hashes, inputs, need, qualified, integration_hashes, progress
sys.path.append(str(BASE.resolve()))
from math_checks import pipeline_checks, planning_checks
from go2nav.eval.slots_checkpoint import windows
from go2nav.eval.slots_inference import IMPLEMENTATION_SOURCES, POLICY, load_full64
from go2nav.eval.slots_recovery import numerics
from go2nav.research import atomic_json, dedicated_device, digest, provenance, require_record, verify_files
from go2nav.training.balls import to_pixels


def gradient_names():
    return {key: set(json.loads(Path(path).read_text())['windows']['train'][0]['unmasked']['gradients'])
            for key, path in [('encoder', 'outputs/gated/C004/slots_n3_s0/result.json'),
                              ('predictor', 'outputs/gated/C004/o0_n3_s0/result.json')]}


def summarize(result):
    """Fail closed on incomplete coverage; preserve complete numerical negatives."""
    need(result.get('complete') and set(result['windows']) == {'train', 'val'}, 'Incomplete windows')
    kind = result['kind']
    names = gradient_names()
    passed = []
    for rows in result['windows'].values():
        need(len(rows) == 8, 'Expected eight paired batches')
        for row in rows:
            need(len(row['episodes']) == len(row['starts']) == 2, 'Expected two examples')
            keys = ['encoder']+([] if kind == 'slots' else ['predictor'])+(['masked'] if kind == 'o1' else [])
            for key in keys:
                value = row[key]
                outputs = {'slots', 'reconstruction', 'masks'} if key == 'encoder' else {'predicted'}
                need(set(value['outputs']) == set(value['grouping']) == outputs, 'Incomplete numerical outputs')
                need(set(value['gradients']) == names['encoder' if key == 'encoder' else 'predictor'], 'Incomplete gradient coverage')
                need(value['missing_gradients'] == {'cpu': [], 'cuda:0': []}, 'Missing gradients')
                local = [v['passed'] for field in ('outputs', 'gradients') for v in value[field].values()]
                for groups in value['grouping'].values():
                    need(set(groups) == {'repeat', 'reversed', 'separate'}, 'Incomplete grouping')
                    local += [v['passed'] for v in groups.values()]
                need(value['passed'] == all(local), 'Inconsistent numerical summary')
                passed += local
            if kind != 'slots':
                value = row['pipeline']
                need(set(value['checks']) == {'cpu', 'repeat', 'reversed', 'separate'}, 'Incomplete pipeline')
                local = []
                for group in value['checks'].values():
                    need(set(group) == {'history', 'goal', 'predicted', 'cost'}, 'Incomplete pipeline fields')
                    local += [v['passed'] for v in group.values()]
                need(value['passed'] == all(local), 'Inconsistent pipeline summary')
                passed += local
    need(set(result['planning']) == (set() if kind == 'slots' else {'train', 'val'}), 'Incomplete planning cohorts')
    for value in result['planning'].values():
        need(set(value['checks']) == {'cpu', 'repeat', 'reversed', 'separate'} and
             set(value['encoding']) == {'history', 'goal'}, 'Incomplete planning fields')
        local = [v['passed'] for v in value['encoding'].values()]
        for group in value['checks'].values():
            need(set(group) == {'rollout', 'cost'} and len(group['rollout']['by_step']) == 8, 'Incomplete rollout')
            local += [v['passed'] for v in group.values()]
        need(value['passed'] == all(local), 'Inconsistent planning summary')
        passed += local
    need(len(passed) == {'slots': 1104, 'o0': 2772, 'o1': 4164}[kind], 'Incorrect comparison count')
    return len(passed), all(passed)


def run(name):
    started = time.monotonic()
    integration = qualified()
    identity, bound = inputs(name)
    record, rh = require_record('C014')
    code = code_hashes()
    out = BASE/name
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path('outputs/gated/C001')/name/'model.pt'
    manifest = provenance('C014', identity, bound)
    manifest['helper_sha256'] = {p: digest(BASE/p) for p in HELPERS}
    manifest['computational_source_sha256'] = code
    manifest.update(integration=integration, integration_helper_sha256=integration_hashes(), budget_policy='U001', work_seconds=None)
    atomic_json(out/'manifest.json', manifest)
    result = dict(research_id='C014', research_sha256=rh, integration=integration, budget_policy='U001', work_seconds=None, **identity,
        status='awaiting_literature_review', selected_policy=POLICY,
        checkpoint_sha256=digest(checkpoint), implementation_sha256={p: code[p] for p in IMPLEMENTATION_SOURCES},
        inputs_sha256=bound, windows={}, planning={}, complete=False, checks_passed=False)
    def check():
        progress(out, name)
    try:
        dedicated_device(.5)
        torch.cuda.set_per_process_memory_fraction(.5)
        model, state = load_full64(checkpoint, 'cuda:0')
        initial = {k: v.detach().cpu().clone() for k, v in
                   (dict(model.named_parameters()) | dict(model.named_buffers())).items()}
        kind, n = identity['kind'], identity['n']
        encoder = model if kind == 'slots' else model.encoder
        cache = None if kind == 'slots' else Path(state['cache'])
        for split in ('train', 'val'):
            frames = np.load(f'outputs/data/balls/n{n}/{split}/frames.npy', mmap_mode='r')
            with np.load(f'outputs/data/balls/n{n}/{split}/labels.npz') as f:
                actions = f['actions']
            slots = None if cache is None else np.load(cache/f'{split}.npy', mmap_mode='r')
            need(slots is None or slots.shape == (*frames.shape[:2], n+2, 128), 'Cache shape differs')
            episodes, starts = windows(frames.shape, n, split)
            result['windows'][split] = []
            for j in range(0, len(episodes), 2):
                check()
                e, t = episodes[j:j+2, None], starts[j:j+2, None]
                raw = torch.from_numpy(np.array(frames[e, t+np.arange(3)]))
                pixels = to_pixels(raw).double()
                row = dict(episodes=e[:, 0].tolist(), starts=t[:, 0].tolist())
                row['encoder'] = numerics('rgb_slots', encoder, check, [pixels])
                if kind != 'slots':
                    states = torch.from_numpy(np.array(slots[e, t+np.arange(4)])).double()
                    commands = torch.from_numpy(np.array(actions[e, t+np.arange(3)]))
                    cached_actions = (commands/3).double()
                    mask = torch.zeros(2, n+2, dtype=torch.bool)
                    row['predictor'] = numerics('predictor', model.predictor, check, [states, cached_actions, mask])
                    if kind == 'o1':
                        mask[0, 1], mask[1, -2:] = True, True
                        row['history_mask'] = mask.tolist()
                        row['masked'] = numerics('predictor', model.predictor, check, [states, cached_actions, mask])
                    goals = torch.from_numpy(np.array(frames[e, t+3]))
                    row['pipeline'] = pipeline_checks(model, raw, goals, commands, check)
                result['windows'][split].append(row)
                atomic_json(out/'result.json', result)
            if kind != 'slots':
                result['planning'][split] = planning_checks(model, frames, actions, split,
                    int(episodes[0]), int(starts[0]), out/f'{split}.npz', check, n)
            atomic_json(out/'result.json', result)
            print(dict(name=name, split=split, paired_batches=8), flush=True)
        after = dict(model.named_parameters()) | dict(model.named_buffers())
        result['weights_unchanged'] = initial.keys() == after.keys() and all(torch.equal(v.cpu(), initial[k]) for k, v in after.items())
        need(result['weights_unchanged'], 'Parameters or buffers changed')
        verify_files(Path.cwd(), bound)
        verify_files(Path.cwd(), code)
        verify_files(BASE, manifest['helper_sha256'])
        verify_files(Path('outputs/gated/U011'), manifest['integration_helper_sha256'])
        result.update(complete=True, inputs_unchanged=True)
        count, passed = summarize(result)
        result.update(comparison_count=count, checks_passed=passed, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        result.update(status='technical_blocked', error=repr(exc), checks_passed=False)
        atomic_json(out/'result.json', result)
        raise
    atomic_json(out/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('windows', 'planning', 'inputs_sha256', 'implementation_sha256')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    with sdpa_kernel(SDPBackend.MATH):
        run(args.run)
