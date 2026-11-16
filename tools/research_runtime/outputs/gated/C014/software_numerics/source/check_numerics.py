"""C014 focused CPU numerical/lineage/supervision software verification."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import psutil
import torch
from common import BASE, HELPERS, admit, case, close, code_hashes, inherited, inputs, training_inputs
from inspect_result import assignment_costs, inspect_payload, recompute
from math_checks import commands_for, metric, pipeline_checks
from validation import gradient_names, summarize
from go2nav.eval.slots_checkpoint import windows
from go2nav.eval.slots_inference import load_full64
from go2nav.eval.slots_recovery import evaluate
from go2nav.models.balls_slots import ObjectMaskedPredictor
from go2nav.research import atomic_json, digest, require_record, verify_files


def refuse(call):
    try:
        call()
    except (ValueError, KeyError, FileNotFoundError, FloatingPointError):
        return
    raise AssertionError('Expected refusal')


def fixture(kind):
    names = gradient_names()
    def numerical(key):
        outputs = ('slots', 'reconstruction', 'masks') if key == 'encoder' else ('predicted',)
        value = dict(passed=True, max_abs=0., rms=0.)
        return dict(outputs={p: value.copy() for p in outputs}, gradients={p: value.copy() for p in names[key]},
            missing_gradients={'cpu': [], 'cuda:0': []}, grouping={p: {m: value.copy() for m in ('repeat', 'reversed', 'separate')} for p in outputs}, passed=True)
    row = dict(episodes=[0, 1], starts=[0, 1], encoder=numerical('encoder'))
    planning = {}
    if kind != 'slots':
        row['predictor'] = numerical('predictor')
        row['pipeline'] = dict(passed=True, checks={m: {p: dict(passed=True) for p in ('history', 'goal', 'predicted', 'cost')} for m in ('cpu', 'repeat', 'reversed', 'separate')})
        planning = {s: dict(passed=True, encoding={k: dict(passed=True) for k in ('history', 'goal')},
            checks={m: dict(cost=dict(passed=True), rollout=dict(passed=True, by_step=[dict(passed=True) for _ in range(8)])) for m in ('cpu', 'repeat', 'reversed', 'separate')}) for s in ('train', 'val')}
    if kind == 'o1':
        row['masked'] = numerical('predictor')
    return dict(kind=kind, complete=True, windows={s: [copy.deepcopy(row) for _ in range(8)] for s in ('train', 'val')}, planning=planning)


def main():
    started = time.monotonic()
    record, rh = require_record('C014')
    checks = []
    torch.set_num_threads(2)
    torch.manual_seed(2026111516)
    for kind, expected in [('slots', 1104), ('o0', 2772), ('o1', 4164)]:
        good = fixture(kind)
        assert summarize(good) == (expected, True)
        bad = copy.deepcopy(good)
        bad['windows']['train'].pop()
        refuse(lambda: summarize(bad))
        bad = copy.deepcopy(good)
        bad['windows']['train'][0]['encoder']['gradients'].pop(next(iter(gradient_names()['encoder'])))
        refuse(lambda: summarize(bad))
    bad = fixture('o1')
    bad['windows']['val'][0].pop('masked')
    refuse(lambda: summarize(bad))
    bad = fixture('o0')
    bad['planning']['train']['checks']['cpu']['rollout']['passed'] = False
    bad['planning']['train']['passed'] = False
    assert summarize(bad) == (2772, False)
    checks.append('Encoder1104/O02772/O14164 exact coverage; omitted pairs, gradients, O1masked checks refuse; complete numerical negative retained')
    allowed = record['protocol']['encoder_cases']+record['protocol']['predictor_cases']
    assert len(set(allowed)) == 13
    for name in allowed:
        assert case(name)['name'] == name
    for name in ('o0_n1_s0', 'o1_n6_s1', 'slots_n3_s3', '../o1_n3_s0', 'l0_n3_s0'):
        refuse(lambda: case(name))
    checks.append('All13 approved numerical identities route explicitly; unplanned cases refuse')
    acts = np.arange(80, dtype=np.float32).reshape(2, 20, 2)/10
    for n in (1, 3, 6):
        for split in ('train', 'val'):
            command = commands_for(acts, 0, 1, split, torch.float64, n=n)
            assert command.shape == (100, 10, 2) and command.dtype == torch.float64
            assert torch.equal(command[0, :2], torch.from_numpy(acts[0, 1:3]).double()/3)
            assert torch.linalg.vector_norm(command[:, 2:], dim=-1).max() <= 1.000001
            assert torch.equal(command, commands_for(acts, 0, 1, split, torch.float64, n=n))
    assert not torch.equal(commands_for(acts, 0, 1, 'train', torch.float64, n=1), commands_for(acts, 0, 1, 'train', torch.float64, n=6))
    checks.append('Count/split-specific fixed candidate streams, physical cap and deployed float64 action normalization')
    model, _ = load_full64('outputs/gated/C001/o0_n3_s0/model.pt')
    for split in ('train', 'val'):
        frames = np.load(f'outputs/data/balls/n3/{split}/frames.npy', mmap_mode='r')
        e, t = windows(frames.shape, 3, split)
        e, t = e[:2, None], t[:2, None]
        raw = torch.from_numpy(np.array(frames[e, t+np.arange(3)]))
        goal = torch.from_numpy(np.array(frames[e, t+3]))
        with np.load(f'outputs/data/balls/n3/{split}/labels.npz') as f:
            commands = torch.from_numpy(np.array(f['actions'][e, t+np.arange(3)]))
        assert raw.shape == (2, 3, 64, 64, 3) and goal.shape == (2, 1, 64, 64, 3)
        assert pipeline_checks(model, raw, goal, commands, lambda: None)['passed']
    for n in (1, 3, 6):
        predictor = ObjectMaskedPredictor().double().eval()
        mask = torch.zeros(2, n+2, dtype=torch.bool)
        mask[0, 1], mask[1, -2:] = True, True
        data = [torch.randn(2, 4, n+2, 128, dtype=torch.float64), torch.randn(2, 3, 2, dtype=torch.float64), mask]
        loss, value = evaluate('predictor', predictor, data)
        loss.backward()
        assert value['predicted'].shape == (2, 4, n+2, 128)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
    checks.append('Actual indexed CPU train/val full pipeline; O1 fixed masks at all3counts produce finite gradients for all83parameters')
    rng = np.random.default_rng(44)
    from scipy.optimize import linear_sum_assignment
    for size in (3, 5, 8):
        matrices = rng.normal(size=(3, size, size))
        brute = assignment_costs(matrices)
        scipy = np.array([m[linear_sum_assignment(m)].sum() for m in matrices])
        assert np.allclose(brute, scipy, atol=1e-12, rtol=1e-12)
    x, y = torch.zeros(2, 8, 3), torch.zeros(2, 8, 3)
    y[0, 7, 0] = 2e-5
    v = metric(x, y)
    assert not v['passed']
    recompute(x.numpy(), y.numpy(), v)
    wrong = dict(v, passed=True)
    refuse(lambda: recompute(x.numpy(), y.numpy(), wrong))
    refuse(lambda: metric(x, torch.full_like(x, float('nan'))))
    checks.append('Independent3/5/8slot exhaustive assignment matches SciPy; raw-array comparison detects false verdict and nonfinite tensors')
    # Exercise the independent inspector on preserved real C013 arrays, expressed
    # as a temporary C014-shaped O0 payload. This is a software fixture only.
    prior = Path('outputs/gated/C013/retry_001/o0_n3_s0')
    old = json.loads((prior/'result.json').read_text())
    payload = dict(old['policies']['encoder64_predictor64'], name='o0_n3_s0', kind='o0', n=3, seed=0,
                   inputs_unchanged=True, selected_policy='encoder64_predictor64', comparison_count=2772)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for split in ('train', 'val'):
            (d/f'{split}.npz').symlink_to((prior/f'encoder64_predictor64_{split}.npz').resolve())
        measured = inspect_payload(payload, d)
        assert measured['comparisons'] == 2772 and measured['independent_array_summaries'] == 84 and measured['exhaustive_assignment_costs'] == 1000
        bad = copy.deepcopy(payload)
        bad['windows']['train'][0]['starts'][0] += 1
        refuse(lambda: inspect_payload(bad, d))
        bad = copy.deepcopy(payload)
        bad['planning']['train']['arrays_sha256'] = '0'*64
        refuse(lambda: inspect_payload(bad, d))
    checks.append('Independent inspector reconstructs preserved C013full64 actual84array summaries/1000assignment costs and refuses sample/array corruption')
    ledger_before = digest('outputs/gated/gpu_intervals.json')
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        ledger = d/'ledger.json'
        atomic_json(ledger, [])
        row, _, work = admit(ledger, d/'out')
        assert work == 1755 and row['research_id'] == 'C004' and row['supervisor'] == 'C014'
        refuse(lambda: admit(ledger, d/'other'))
        class Live:
            def poll(self): return None
        refuse(lambda: close(ledger, row['id'], Live()))
        refuse(lambda: close(ledger, 'wrong', None))
        close(ledger, row['id'], None)
        refuse(lambda: close(ledger, row['id'], None))
        for exhausted in [[dict(research_id='C004', start=0, end=8*3600)], [dict(research_id='D001', start=0, end=48*3600)]]:
            atomic_json(ledger, exhausted)
            refuse(lambda: admit(ledger, d/'exhausted'))
    base = inherited()
    for seconds, deadline, expected in [(0, 2, False), (10, .1, True)]:
        argv = [sys.executable, '-c', f'import time; time.sleep({seconds})']
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc = psutil.Process(child.pid)
        signals = []
        timed = base.wait_owned(child, proc, proc.create_time(), argv, Path.cwd(), time.monotonic()+deadline, signals, poll_seconds=.02, term_seconds=.5, kill_seconds=.5)
        assert timed == expected and child.poll() is not None and bool(signals) == expected
    assert digest('outputs/gated/gpu_intervals.json') == ledger_before
    checks.append('Existing reserve/global accounting, open/exhausted/live/wrong/double-closure refusals and actual ownedCPU normal/deadline cleanup')
    identity, bound = inputs('slots_n3_s0')
    for name in ('o0_n3_s0', 'o1_n3_s0'):
        _, training = training_inputs(name)
        assert any(p.endswith('/checkpoint.pt') for p in training)
    refuse(lambda: inputs('o1_n3_s0'))
    assert not torch.cuda.is_initialized()
    checks.append('Actual encoder and bothpredictor terminal/data/cache/source bindings verify; missing actual full64encoder admission refuses, no CUDA')
    return dict(research_id='C014', research_sha256=rh, status='awaiting_literature_review', checks=checks,
        check_groups=len(checks), helper_sha256={p: digest(BASE/p) for p in HELPERS}, repository_source_sha256=code_hashes(),
        elapsed_seconds=time.monotonic()-started, cuda_initialized=False, actual_encoder_input_count=len(bound),
        preimplementation_sha256=digest(BASE/'pre_numerical_implementation.json'),
        limitation='CPU software verification and preserved C013 numerical-array fixtures; no new selected-checkpoint CPU/GPU scientific validation or C009v2 evaluation admission.')


if __name__ == '__main__':
    out = BASE/'software_numerics'
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in HELPERS:
        shutil.copyfile(BASE/p, out/'source'/p)
    try:
        result = main()
    except BaseException as exc:
        atomic_json(out/'result.json', dict(research_id='C014', status='technical_blocked', error=repr(exc), helper_sha256={p: digest(BASE/p) for p in HELPERS}))
        raise
    atomic_json(out/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'repository_source_sha256'}))
