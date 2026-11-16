"""Scoped C006 CPU helper checks, with no scientific cases or GPU work."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, reviewed_artifact, verify_files
from analysis import sample, targets, features, fit_linear, apply_linear, residual, finite, guard, np, torch

base = Path('outputs/gated/C006')
record, rh = require_record('C006')
out = base/'software'
out.mkdir(exist_ok=False)
started = time.monotonic()
checks = {}


def rejects(call, exception):
    try:
        call()
    except exception:
        return
    raise AssertionError('Expected rejection')


try:
    for name in ['analysis.py', 'run_one.py', 'check_probe.py']:
        ast.parse((base/name).read_text())
    guard()
    for n in [1, 3, 6]:
        for split in ['train', 'test']:
            e, t = sample(1000, 100, n, split)
            rng = np.random.default_rng([2026111511, n, split == 'test'])
            expected_e = rng.integers(1000, size=4096) if split == 'train' else np.repeat(np.arange(1000), 5)
            expected_t = rng.integers(2, 96, size=len(expected_e))
            assert np.array_equal(e, expected_e) and np.array_equal(t, expected_t)
            assert t.min() >= 2 and t.max()+4 < 100
            if split == 'test':
                assert np.array_equal(np.bincount(e), np.full(1000, 5))
    rejects(lambda: sample(100, 100, 3, 'val'), ValueError)
    checks['exact_C001_sampling_all_counts_and_bounds'] = True

    red = np.array([[2., 3.], [4., 5.]])
    white = np.array([[[4., 2.], [1., 3.], [1., 1.]], [[3., 4.], [2., 1.], [3., 2.]]])
    np.testing.assert_array_equal(targets(red, white), targets(red, white[:, [2, 0, 1]]))
    np.testing.assert_array_equal(targets(red, white)[0], [2, 3, 1, 1, 1, 3, 4, 2])
    checks['white_lexicographic_sort_and_permutation_invariance'] = True

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.observations = []
            self.actions = []

        def encode(self, pixels):
            self.observations.append(pixels.clone())
            return pixels.mean((-2, -1))[..., :2]

        def rollout(self, z, a, steps):
            self.actions.append(a.clone())
            # Order-sensitive output exposes accidental future-action permutations.
            weights = torch.arange(1, 7, dtype=a.dtype).reshape(1, 6, 1)
            end = z[:, -1]+(a*weights).sum(1)
            return end[:, None].expand(-1, steps, -1)

    frames = np.zeros((2, 12, 4, 4, 3), dtype=np.uint8)
    for ep in range(2):
        for t in range(12):
            frames[ep, t] = ep*100+t
    actions = np.arange(48, dtype=np.float64).reshape(2, 12, 2)
    episodes, starts = np.array([0, 1, 0]), np.array([3, 6, 5])
    model = RecordingModel()
    values = features(model, frames, actions, episodes, starts, batch=32)
    factual, reversed_actions = model.actions
    expected = torch.from_numpy(actions[episodes[:, None], starts[:, None]+np.arange(-2, 4)])/3
    torch.testing.assert_close(factual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reversed_actions[:, :2], factual[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(reversed_actions[:, 2:], factual[:, 2:].flip(1), rtol=0, atol=0)
    assert not torch.equal(values['rollout'], values['shuffled'])
    assert torch.equal(values['current'], values['no_rollout'])
    altered = frames.copy()
    # This separate single case alters every future image but retains its history.
    altered[0, 4:] = 255
    before = features(RecordingModel(), frames, actions, episodes[:1], starts[:1])
    after = features(RecordingModel(), altered, actions, episodes[:1], starts[:1])
    assert all(torch.equal(before[k], after[k]) for k in before)
    checks['past_actions_preserved_future_reversed_no_future_image_leakage'] = True
    singles = features(RecordingModel(), frames, actions, episodes, starts, batch=1)
    reverse = features(RecordingModel(), frames, actions, episodes[::-1].copy(), starts[::-1].copy(), batch=3)
    for key, value in values.items():
        torch.testing.assert_close(value, singles[key], rtol=0, atol=0)
        torch.testing.assert_close(value, reverse[key].flip(0), rtol=0, atol=0)
    checks['feature_batch_and_permutation_consistency_fake_model'] = True

    rng = np.random.default_rng(42)
    x = rng.normal(size=(257, 8)); x[:, -1] = 3  # Constant column exercises std floor.
    y = x@rng.normal(size=(8, 4))+rng.normal(size=4)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    decoder = fit_linear(xt, yt)
    mean, std, w = decoder
    design = np.column_stack(((x-mean.numpy())/std.numpy(), np.ones(len(x))))
    penalty = np.diag([.001]*8+[0.])
    augmented = np.vstack((design/np.sqrt(len(x)), np.sqrt(penalty)))
    rhs = np.vstack((y/np.sqrt(len(x)), np.zeros((9, 4))))
    reference = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    np.testing.assert_allclose(w.numpy(), reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(apply_linear(xt, decoder).numpy(), design@reference, rtol=1e-10, atol=1e-10)
    ridge_residual = residual(xt, yt, decoder)
    assert ridge_residual <= 1e-8
    checks['ridge_augmented_lstsq_constant_column_and_residual'] = True

    for value in [float('nan'), np.array([np.inf]), torch.tensor([float('nan')]), {'nested': [float('inf')]}]:
        rejects(lambda: finite(value), ValueError)
    saved = os.environ.pop('HIP_VISIBLE_DEVICES')
    try:
        rejects(guard, RuntimeError)
    finally:
        os.environ['HIP_VISIBLE_DEVICES'] = saved
    guard()
    checks['nonfinite_and_visible_GPU_environment_rejected'] = True

    with tempfile.TemporaryDirectory(prefix='c006_bindings_') as tmp:
        root = Path(tmp)
        artifact = root/'artifact'; artifact.write_bytes(b'original')
        hashes = {'artifact': digest(artifact)}
        verify_files(root, hashes)
        result = root/'result.json'
        atomic_json(result, dict(checkpoint_sha256=digest(artifact)))
        # Explicitly synthetic fixture: it is never a scientific approval.
        review = dict(research_id='synthetic_fixture', result='result.json', result_sha256=digest(result),
            reviewed_at='fixture', sources=[dict(url='fixture', version='fixture', sections='fixture', consequence='fixture')],
            measurements='fixture', uncertainty='fixture', explanations='fixture', next_step='fixture', status='passed')
        atomic_json(root/'review.json', review)
        reviewed_artifact(root/'review.json', artifact)
        artifact.write_bytes(b'changed')
        rejects(lambda: verify_files(root, hashes), ValueError)
        rejects(lambda: reviewed_artifact(root/'review.json', artifact), ValueError)
        result.write_text('{}')
        rejects(lambda: require_review(root/'review.json'), ValueError)
    checks['source_data_artifact_and_stale_parent_bindings_reject_tamper'] = True

    actual_parents = {}
    for name in record['protocol']['order']:
        parent = Path('outputs/gated/A003_recovered')/name
        evidence = reviewed_artifact(parent/'review.json', parent/'calibrated.pt', 'calibrated_sha256')
        audit = Path('outputs/gated/A015/finals')/name
        assert require_review(audit/'review.json')['status'] == 'passed'
        assert require_review(Path('outputs/gated/A016/runs')/name/'review.json')['status'] == 'passed'
        final = json.loads((audit/'result.json').read_text())
        assert evidence['source_sha256'] == final['model_sha256'] and final['step'] == 303000
        config = Path('outputs/balls/recovery_A014')/name/'config.json'
        assert digest(config) == json.loads((audit/'manifest.json').read_text())['snapshot_sha256']['config.json']
        actual_parents[name] = digest(parent/'review.json')
    checks['six_actual_final_artifact_lineages_bound_without_inference'] = True
    guard()
    result = dict(research_id='C006', research_sha256=rh, status='awaiting_literature_review', checks=checks,
        source_sha256={n: digest(base/n) for n in ['analysis.py', 'run_one.py', 'check_probe.py']},
        repository_source_sha256={str(p): digest(p) for p in sorted(Path('go2nav').rglob('*.py'))},
        a016_supervision_sha256=digest('outputs/gated/A016/supervision.py'), actual_parent_reviews=actual_parents,
        ridge_relative_residual=ridge_residual, cuda_initialized=False, scientific_cases_executed=0,
        elapsed_seconds=time.monotonic()-started,
        limitations='Scoped helpers and actual parent hashes. Real-model numerics, data checks, resource supervision and scientific scores require each actual case. Owned-child termination reuses unchanged reviewed A016 source.')
    atomic_json(out/'result.json', result)
    print(json.dumps(dict(checks=checks, ridge_relative_residual=ridge_residual, cuda_initialized=False)))
except BaseException as exc:
    atomic_json(out/'result.json', dict(research_id='C006', research_sha256=rh, status='technical_blocked', checks=checks, error=repr(exc)))
    raise
