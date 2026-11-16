"""Independent NumPy inspection of C014 summaries and saved planning arrays."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import psutil
from common import ORIGINAL as BASE, case, need, reviewed, operation_metadata
from go2nav.research import atomic_json, digest, require_record, verify_files


def recompute(a, b, value):
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid saved array')
    delta = np.abs(a-b)
    close = np.isclose(a, b, rtol=1e-4, atol=1e-5)
    need(bool(close.all()) == value['passed'] and int((~close).sum()) == value['failed_elements'], 'Comparison verdict differs')
    need(math.isclose(float(delta.max()), value['max_abs'], rel_tol=1e-12, abs_tol=1e-18), 'Comparison maximum differs')
    need(math.isclose(float(np.sqrt(np.mean(delta.astype('float64')**2))), value['rms'], rel_tol=1e-12, abs_tol=1e-18), 'Comparison RMS differs')


def assignment_costs(matrix):
    """Enumerate all assignments independently, with one cost matrix in memory."""
    need(matrix.ndim == 3 and matrix.shape[-1] == matrix.shape[-2] and matrix.shape[-1] in (3, 5, 8), 'Invalid assignment shape')
    need(np.isfinite(matrix).all(), 'Nonfinite assignment matrix')
    slots = matrix.shape[-1]
    permutations = np.array(list(itertools.permutations(range(slots))))
    return np.array([value[np.arange(slots), permutations].sum(-1).min() for value in matrix])


def inspect_payload(r, out):
    identity = case(r['name'])
    need(all(r[k] == v for k, v in identity.items()), 'Wrong case identity')
    kind, n = identity['kind'], identity['n']
    need(r['complete'] and r['weights_unchanged'] and r['inputs_unchanged'], 'Incomplete or mutated result')
    need(r['selected_policy'] == 'encoder64_predictor64', 'Wrong precision policy')
    expected_gradients = {key: set(json.loads(Path(p).read_text())['windows']['train'][0]['unmasked']['gradients'])
        for key, p in [('encoder', 'outputs/gated/C004/slots_n3_s0/result.json'), ('predictor', 'outputs/gated/C004/o0_n3_s0/result.json')]}
    measured = dict(comparisons=0, independent_array_summaries=0, raw_arrays=0, exhaustive_assignment_costs=0,
                    maximum_assignment_difference=0., failures=[], maxima={})
    outcomes = []
    def reported(value, path):
        need(type(value['passed']) is bool and all(math.isfinite(value[k]) and value[k] >= 0 for k in ('max_abs', 'rms')), 'Invalid comparison summary')
        measured['comparisons'] += 1
        category = path.split('/')[0]
        measured['maxima'][category] = max(measured['maxima'].get(category, 0), value['max_abs'])
        if not value['passed']:
            measured['failures'].append(dict(path=path, **value))
        return value['passed']
    need(set(r['windows']) == {'train', 'val'}, 'Missing split')
    for split, rows in r['windows'].items():
        frames = np.load(f'outputs/data/balls/n{n}/{split}/frames.npy', mmap_mode='r')
        length = frames.shape[1]-3
        flat = np.random.default_rng((2026111516, n, ['train', 'val'].index(split))).choice(frames.shape[0]*length, 16, replace=False)
        episodes, starts = flat//length, flat % length
        need(len(rows) == 8, 'Incomplete pair coverage')
        for j, row in enumerate(rows):
            need(row['episodes'] == episodes[j*2:j*2+2].tolist() and row['starts'] == starts[j*2:j*2+2].tolist(), 'Sample identities differ')
            keys = ['encoder']+([] if kind == 'slots' else ['predictor'])+(['masked'] if kind == 'o1' else [])
            if kind == 'o1':
                mask = np.zeros((2, n+2), bool)
                mask[0, 1], mask[1, -2:] = True, True
                need(row['history_mask'] == mask.tolist(), 'History masks differ')
            for key in keys:
                value = row[key]
                names = {'slots', 'reconstruction', 'masks'} if key == 'encoder' else {'predicted'}
                need(set(value['outputs']) == set(value['grouping']) == names, 'Incomplete outputs')
                need(set(value['gradients']) == expected_gradients['encoder' if key == 'encoder' else 'predictor'], 'Gradient names differ')
                need(value['missing_gradients'] == {'cpu': [], 'cuda:0': []}, 'Missing gradients')
                local = []
                for field in ('outputs', 'gradients'):
                    for name, v in value[field].items():
                        local.append(reported(v, f'{key}/{split}/{j}/{field}/{name}'))
                for name, group in value['grouping'].items():
                    need(set(group) == {'repeat', 'reversed', 'separate'}, 'Missing grouping')
                    local += [reported(v, f'{key}/{split}/{j}/{name}/{mode}') for mode, v in group.items()]
                need(value['passed'] == all(local), 'Inconsistent numerical summary')
                outcomes += local
            if kind != 'slots':
                local = []
                need(set(row['pipeline']['checks']) == {'cpu', 'repeat', 'reversed', 'separate'}, 'Missing pipeline mode')
                for mode, group in row['pipeline']['checks'].items():
                    need(set(group) == {'history', 'goal', 'predicted', 'cost'}, 'Missing pipeline output')
                    local += [reported(v, f'pipeline/{split}/{j}/{mode}/{name}') for name, v in group.items()]
                need(row['pipeline']['passed'] == all(local), 'Inconsistent pipeline summary')
                outcomes += local
    need(set(r['planning']) == (set() if kind == 'slots' else {'train', 'val'}), 'Missing planning coverage')
    for split, v in r['planning'].items():
        first = r['windows'][split][0]
        need((v['episode'], v['start']) == (first['episodes'][0], first['starts'][0]), 'Planning sample differs')
        file = out/f'{split}.npz'
        need(digest(file) == v['arrays_sha256'], 'Planning arrays changed')
        with np.load(file, allow_pickle=False) as data:
            a = {k: data[k] for k in data.files}
        modes = ['cpu', 'batched', 'repeat', 'reversed', 'separate']
        fields = {'cpu_commands', 'gpu_commands', 'cpu_history', 'gpu_history', 'cpu_goal', 'gpu_goal'} | {f'{m}_{f}' for m in modes for f in ('rollout', 'cost')}
        need(set(a) == fields, 'Saved array inventory differs')
        measured['raw_arrays'] += len(a)
        for name, value in a.items():
            shape = (100, 10, 2) if name.endswith('commands') else (1, 3, n+2, 128) if name.endswith('history') else (1, n+2, 128) if name.endswith('goal') else (100, 8, n+2, 128) if name.endswith('rollout') else (100,)
            need(value.shape == shape and list(shape) == v['array_shapes'][name] and value.dtype == np.float64 and np.isfinite(value).all(), 'Wrong array shape/dtype or nonfinite')
        with np.load(f'outputs/data/balls/n{n}/{split}/labels.npz') as labels:
            actions = labels['actions']
        rng = np.random.default_rng((2026111516, n, ['train', 'val'].index(split), 100))
        future = (2*rng.standard_normal((100, 8, 2))).astype(np.float32)
        future *= np.minimum(1, 3/np.maximum(np.linalg.norm(future, axis=-1, keepdims=True), 1e-12))
        e, t = v['episode'], v['start']
        commands = np.concatenate((np.broadcast_to(actions[e, t:t+2], (100, 2, 2)), future), 1).astype(np.float64)/3
        need(np.array_equal(commands, a['cpu_commands']) and np.allclose(commands, a['gpu_commands'], rtol=np.finfo(np.float64).eps, atol=0), 'Physical command cohort differs')
        local = []
        need(set(v['checks']) == {'cpu', 'repeat', 'reversed', 'separate'}, 'Planning comparisons missing')
        for mode, group in v['checks'].items():
            need(set(group) == {'rollout', 'cost'}, 'Missing rollout or cost')
            for field, value in group.items():
                recompute(a[f'batched_{field}'], a[f'{mode}_{field}'], value)
                measured['independent_array_summaries'] += 1
                local.append(reported(value, f'planning/{split}/{mode}/{field}'))
                if field == 'rollout':
                    need(len(value['by_step']) == 8, 'Missing per-step comparisons')
                    for step in range(8):
                        recompute(a['batched_rollout'][:, step], a[f'{mode}_rollout'][:, step], value['by_step'][step])
                        measured['independent_array_summaries'] += 1
        need(set(v['encoding']) == {'history', 'goal'}, 'Missing encoding comparisons')
        for name, value in v['encoding'].items():
            recompute(a[f'gpu_{name}'], a[f'cpu_{name}'], value)
            measured['independent_array_summaries'] += 1
            local.append(reported(value, f'planning/{split}/encoding/{name}'))
        need(v['passed'] == all(local), 'Inconsistent planning summary')
        outcomes += local
        for mode in modes:
            goal = a['cpu_goal' if mode == 'cpu' else 'gpu_goal']
            predicted = a[f'{mode}_rollout'][:, -1]
            matrix = ((predicted[:, :, None, :]-goal[:, None, :, :])**2).sum(-1)
            costs = assignment_costs(matrix)
            need(np.allclose(costs, a[f'{mode}_cost'], rtol=1e-12, atol=1e-12), 'Independent assignment cost differs')
            measured['exhaustive_assignment_costs'] += len(costs)
            measured['maximum_assignment_difference'] = max(measured['maximum_assignment_difference'], float(np.abs(costs-a[f'{mode}_cost']).max()))
    expected = {'slots': 1104, 'o0': 2772, 'o1': 4164}[kind]
    need(measured['comparisons'] == r['comparison_count'] == expected and r['checks_passed'] == all(outcomes), 'Overall numerical outcome differs')
    need(measured['independent_array_summaries'] == (0 if kind == 'slots' else 84), 'Wrong saved-summary coverage')
    need(measured['raw_arrays'] == (0 if kind == 'slots' else 32), 'Wrong array coverage')
    need(measured['exhaustive_assignment_costs'] == (0 if kind == 'slots' else 1000), 'Wrong assignment coverage')
    measured['numerical_passed'] = all(outcomes)
    return measured


def run(name):
    out, op = BASE/name, BASE/'operations'/name
    r = json.loads((out/'result.json').read_text())
    o = json.loads((op/'result.json').read_text())
    m = json.loads((out/'manifest.json').read_text())
    om = json.loads((op/'manifest.json').read_text())
    operation_metadata(r, o, m, om, json.loads(Path('outputs/gated/gpu_intervals.json').read_text()))
    need(r['research_sha256'] == require_record('C014')[1], 'Record differs')
    need(o['returncode'] == 0 and o['child_terminal'] and not any(o[k] for k in ('timed_out', 'actions', 'errors')), 'Operation not normally terminal')
    try:
        need(psutil.Process(om['pid']).create_time() != om['creation_time'], 'Owned child still present')
    except psutil.NoSuchProcess:
        pass
    need(o['manifest_sha256'] == digest(op/'manifest.json'), 'Operation manifest changed')
    verify_files(out, o['child_artifact_sha256'])
    verify_files(Path.cwd(), m['inputs'])
    need(m['inputs'] == r['inputs_sha256'] == om['inputs'], 'Input inventories differ')
    verify_files(Path.cwd(), m['computational_source_sha256'])
    verify_files(BASE, m['helper_sha256'])
    reviewed(BASE/'software_numerics', 'C014')
    need(digest(BASE/'software_numerics/review.json') == om['software_review_sha256'], 'Software review changed')
    rows = json.loads(Path('outputs/gated/gpu_intervals.json').read_text())
    owned = [v for v in rows if v['id'] == om['outer_interval_id']]
    need(len(owned) == 1 and owned[0]['end'] is not None and owned[0]['research_id'] == 'C004' and owned[0]['supervisor'] == 'C014', 'Owned interval not closed')
    result = dict(research_id='C014', integration_research_id='U011', integration=r['integration'], name=name, status='awaiting_literature_review',
        result_sha256=digest(out/'result.json'), operational_result_sha256=digest(op/'result.json'),
        inspection_source_sha256=digest(__file__), owned_closed_interval=owned[0], **inspect_payload(r, out),
        limitation='Independent coverage/identity/finite-summary inspection and all retained planning-array comparisons and costs. Raw small-batch outputs and gradients were not saved, so those tensors are not independently recomputed. No physical performance claim.')
    need('torch' not in sys.modules, 'Inspector imported Torch')
    result['torch_imported'] = False
    need(not (out/'inspection.json').exists(), 'Inspection already exists')
    atomic_json(out/'inspection.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    run(p.parse_args().run)
