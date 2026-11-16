"""CPU software fixtures only: C014 precision boundary and evidence refusals."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch import nn
from go2nav.eval import slots_inference as inference
from go2nav.eval.balls_binding import measure
from go2nav.eval.balls_planning import Inputs, learned_actions
from go2nav.eval.balls_probes import features
from go2nav.eval.slots_checkpoint import windows
from go2nav.models.balls_slots import SlotWorld
from go2nav.research import atomic_json, digest, require_record, verify_files
from go2nav.training.balls import to_pixels
from go2nav.training.balls_slots import load_world, load_encoder

BASE = Path('outputs/gated/C014')


def refuse(call):
    try:
        call()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError('Invalid evidence was accepted')


def fixture_review(directory, result, sources):
    # Deliberately synthetic, ephemeral review fixtures; never scientific evidence.
    atomic_json(directory/'result.json', result)
    atomic_json(directory/'review.json', dict(research_id='C014', result='result.json',
        result_sha256=digest(directory/'result.json'), reviewed_at='synthetic_fixture',
        sources=sources, measurements='synthetic software fixture', uncertainty='synthetic',
        explanations='synthetic', next_step='synthetic', status='passed'))
    return directory/'review.json'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, 5, dtype=torch.float64))
        self.calls = []
    def forward(self, pixels):
        assert pixels.dtype == torch.float64
        self.calls.append(tuple(pixels.shape))
        value = pixels.mean((-3, -2, -1))[..., None, None]*self.scale
        return value.expand(-1, -1, 5, -1).clone()


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4, dtype=torch.float64))
    def rollout(self, history, actions, steps):
        assert history.dtype == actions.dtype == torch.float64
        z, values = history[:, -1:], []
        for i in range(steps):
            z = z+actions[:, i+2:i+3].sum(-1)[..., None, None]*self.scale
            values.append(z)
        return torch.cat(values, 1)


def run():
    started = time.monotonic()
    record, rh = require_record('C014')
    torch.set_num_threads(2)
    torch.manual_seed(2026111516)
    sources = {p: digest(p) for p in inference.IMPLEMENTATION_SOURCES}
    checkpoint = Path('outputs/gated/C001/o0_n3_s0/model.pt')
    encoder_path = Path('outputs/gated/C001/slots_n3_s0/model.pt')
    before = {str(p): digest(p) for p in (checkpoint, encoder_path)}
    checks = []
    model, state = inference.load_full64(checkpoint)
    encoder, encoder_state = inference.load_full64(encoder_path)
    original = load_world(checkpoint, 'cpu')
    for loaded, raw in ((model, original), (encoder, load_encoder(encoder_path, 'cpu')[0])):
        a = dict(loaded.named_parameters()) | dict(loaded.named_buffers())
        b = dict(raw.named_parameters()) | dict(raw.named_buffers())
        assert a.keys() == b.keys()
        for name, tensor in a.items():
            assert torch.equal(tensor, b[name].to(tensor.dtype))
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float64 and torch.isfinite(tensor).all()
        assert not loaded.training
    checks.append('Actual O0/encoder strict loads and exact promotion of every parameter and buffer, including nonpersistent grids')
    policy_path = Path('outputs/gated/C013/policy.py')
    assert digest(policy_path) == 'a7477ccaa240e9ac8fba897a888a34eebc89bf2dc16d847cd10a37093c0a4cac'
    spec = importlib.util.spec_from_file_location('c013_policy_reference', policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    reference = policy.load_policy(checkpoint, inference.POLICY)
    samples = {}
    for split in ('train', 'val'):
        frames_path = Path(f'outputs/data/balls/n3/{split}/frames.npy')
        labels_path = frames_path.with_name('labels.npz')
        frames = np.load(frames_path, mmap_mode='r')
        with np.load(labels_path) as labels:
            actions = labels['actions']
        episodes, starts = windows(frames.shape, 3, split)
        e, t = episodes[:2, None], starts[:2, None]
        pixels = to_pixels(torch.from_numpy(np.array(frames[e, t+np.arange(3)]))).double()
        with torch.no_grad():
            actual, expected = model.encode(pixels), reference.encode(pixels)
            assert torch.equal(actual, expected)
            assert torch.equal(encoder(pixels), expected)
            a = torch.from_numpy(np.array(actions[e, t+np.arange(3)])).double()/3
            assert torch.equal(model.rollout(actual, a, 1), reference.rollout(expected, a, 1))
            masks = encoder.decode(actual[:, -1:])[1]
            assert masks.dtype == torch.float64 and torch.isfinite(masks).all()
        # The actual deployed probe call's history starts two frames before t.
        f, g = (features(m, frames, actions, episodes[:2], starts[:2]+2) for m in (model, reference))
        assert f.keys() == g.keys() and all(torch.equal(f[k], g[k]) for k in f)
        samples[split] = dict(episodes=episodes[:2].tolist(), starts=starts[:2].tolist(),
            frames_sha256=digest(frames_path), labels_sha256=digest(labels_path))
    checks.append('Actual indexed train/validation histories, rollout, decoder and four probe feature variants exactly match C013 full64')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        good = dict(research_id='C014', research_sha256=rh, status='awaiting_literature_review',
            selected_policy=inference.POLICY, complete=True, checks_passed=True,
            weights_unchanged=True, inputs_unchanged=True, checkpoint_sha256=digest(checkpoint),
            kind='o0', n=3, seed=0, name='o0_n3_s0', comparison_count=2772,
            implementation_sha256=sources, inputs_sha256={str(checkpoint.resolve()): digest(checkpoint)})
        review = fixture_review(directory, good, record['sources'])
        assert inference.require_numerical_review(checkpoint, review) == good
        admitted = inference.load_reviewed_world(checkpoint, review, 'cpu')
        assert all(p.dtype == torch.float64 for p in admitted.parameters())
        changes = dict(selected_policy='encoder32_predictor64', complete=False, checks_passed=False,
            weights_unchanged=False, inputs_unchanged=False, checkpoint_sha256='0'*64,
            kind='o1', n=1, seed=1, name='o0_n3_s1', comparison_count=2771,
            implementation_sha256={}, inputs_sha256={}, research_id='C004', research_sha256='0'*64,
            status='technical_blocked')
        for key, value in changes.items():
            bad = copy.deepcopy(good)
            bad[key] = value
            fixture_review(directory, bad, record['sources'])
            refuse(lambda: inference.require_numerical_review(checkpoint, review))
        bad = copy.deepcopy(good)
        bad['implementation_sha256'][inference.IMPLEMENTATION_SOURCES[0]] = '0'*64
        fixture_review(directory, bad, record['sources'])
        refuse(lambda: inference.require_numerical_review(checkpoint, review))
        fixture_review(directory, good, record['sources'])
        saved_review = json.loads(review.read_text())
        saved_review['status'] = 'technical_blocked'
        atomic_json(review, saved_review)
        refuse(lambda: inference.require_numerical_review(checkpoint, review))
        fixture_review(directory, good, record['sources'])
        (directory/'result.json').write_text('{}')
        refuse(lambda: inference.require_numerical_review(checkpoint, review))
        refuse(lambda: inference.require_numerical_review(checkpoint, directory/'missing.json'))
        refuse(lambda: inference.require_numerical_review(checkpoint, Path('outputs/gated/C013/retry_001/o0_n3_s0/review.json')))
        fixture_review(directory, good, record['sources'])
        refuse(lambda: inference.load_reviewed_encoder(checkpoint, review, 'cpu'))
        enc_good = dict(good, kind='slots', name='slots_n3_s0', comparison_count=1104,
            checkpoint_sha256=digest(encoder_path), inputs_sha256={str(encoder_path.resolve()): digest(encoder_path)})
        fixture_review(directory, enc_good, record['sources'])
        checked_encoder, _ = inference.load_reviewed_encoder(encoder_path, review, 'cpu')
        assert next(checked_encoder.parameters()).dtype == torch.float64
        refuse(lambda: inference.load_reviewed_world(encoder_path, review, 'cpu'))
    checks.append('Synthetic admission fixtures: world/encoder pass; 23 invalid, stale, wrong-policy, changed-source and wrong-kind cases refuse')
    toy = SlotWorld(Encoder(), Predictor()).eval()
    rng = np.random.default_rng(2026111516)
    frames = rng.integers(0, 256, (2, 12, 8, 8, 3), dtype=np.uint8)
    actions = rng.normal(size=(2, 12, 2)).astype(np.float32)
    f = features(toy, frames, actions, np.array([0, 1]), np.array([3, 4]))
    assert all(x.dtype == torch.float64 and torch.isfinite(x).all() for x in f.values())
    inputs = Inputs(frames[0, :3], actions[0, :2], frames[0, 6])
    first = learned_actions(toy, inputs, 4, 2026111504)
    assert np.array_equal(first, learned_actions(toy, inputs, 4, 2026111504))
    assert first.shape == (4, 2) and np.linalg.norm(first, axis=-1).max() <= 3.00001
    assert (1, 3, 3, 8, 8) in toy.encoder.calls and (1, 1, 3, 8, 8) in toy.encoder.calls
    checks.append('Original full100candidate/20iteration CEM and probe interfaces use float64; independent one-frame goal, finite deterministic capped plans')
    verify_files(Path.cwd(), before)
    verify_files(Path.cwd(), sources)
    assert not torch.cuda.is_initialized()
    return dict(research_id='C014', research_sha256=rh, status='awaiting_literature_review',
        scope='loader_and_evaluator_boundary_only', checks=checks, check_groups=len(checks),
        implementation_sha256=sources, helper_sha256=digest(__file__), input_sha256=before,
        samples=samples, elapsed_seconds=time.monotonic()-started, cuda_initialized=False,
        limitation='CPU software/equivalence fixtures only. No C014 selected-checkpoint GPU validation, actual admission, numerical supervisor or versioned C009 route is qualified by this result.')


if __name__ == '__main__':
    out = BASE/'software_loader'
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = run()
    except BaseException as exc:
        atomic_json(out/'result.json', dict(research_id='C014', status='technical_blocked', error=repr(exc), helper_sha256=digest(__file__)))
        raise
    atomic_json(out/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('implementation_sha256', 'samples')}))
