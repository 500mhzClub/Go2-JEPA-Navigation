"""F004 CPU inference/artifact fixtures; no actual selected-model numeric claim."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from f004_forecast_evidence import BASE, ROOT, implementation_sources, reviewed, selected_checkpoint
from f004_software import FlatRasters, fixture, small_model
from go2nav.data.paired_forecasts import INPUTS, extract
from go2nav.eval.paired_forecast_inference import (FLAVORS, FRAMES, NUMERIC_POLICY, InputView, load_model,
    numerical_comparison, predict, read_artifact, validate_keys, write_artifact)
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, require_record


def data_fixture(n=5):
    data, _ = extract(*fixture())
    data = {k: np.repeat(v, n, axis=0) for k, v in data.items()}
    data['episode'][:] = np.arange(n)
    data['history_pose'][..., 0] += np.arange(n)[:, None]*3
    data['history_pose'][..., 1] += np.arange(n)[:, None]*2
    data['history_pose'][..., 2] = np.arange(n)[:, None]*.3
    data['history_velocity'][:, :, 0] = [1.2, -.7]
    return data


def arrays_fixture(n=2):
    return {f'{flavor}_{frame}': np.zeros((n, 7, 16, 6, 2), np.float32)
            for flavor in FLAVORS for frame in FRAMES}


class Checks(unittest.TestCase):
    def test_independent_world_cv_oracle_and_partial_batch(self):
        data = data_fixture()
        model = ObjectJEPA(ObjectConfig(dim=8, depth=1, heads=1, head_dim=8, mlp=16,
                                       hidden=12, dropout=.1, residual=True))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        events = []
        output, keys = predict(model, InputView(data), FlatRasters(), 32, progress=lambda a, b: events.append((a, b)))
        self.assertEqual(events, [(32, 35), (35, 35)])
        self.assertFalse(model.training)
        self.assertEqual(model.config.dropout, .1)
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        for scene in range(5):
            origin, yaw = data['history_pose'][scene, -1, :2].astype(float), float(data['history_pose'][scene, -1, 2])
            c, s = np.cos(yaw), np.sin(yaw)
            for future in range(7):
                for step in range(16):
                    world = data['history_xy'][scene, -1, 0].astype(float)+.4*(step+1)*data['history_velocity'][scene, -1, 0].astype(float)
                    relative = world-origin
                    body = np.array([c*relative[0]+s*relative[1], -s*relative[0]+c*relative[1]])
                    np.testing.assert_allclose(output['world'][scene, future, step, 0], world, atol=2e-5, rtol=1e-5)
                    np.testing.assert_allclose(output['body'][scene, future, step, 0], body, atol=2e-5, rtol=1e-5)
        validate_keys(keys)
        self.assertFalse(torch.cuda.is_initialized())

    def test_labels_absent_real_model_grouping_and_exact_case_order(self):
        data = data_fixture()
        model = small_model()
        view = InputView(data)
        one, keys = predict(model, view, FlatRasters(), 1)
        observed_only = {k: data[k] for k in (*INPUTS, 'episode', 't0')}
        grouped, grouped_keys = predict(model, InputView(observed_only), FlatRasters(), 32)
        self.assertEqual(set(view.take(np.array([0]), np.array([6]))), set(INPUTS))
        np.testing.assert_array_equal(keys, grouped_keys)
        np.testing.assert_array_equal(keys[:, 0], np.repeat(np.arange(5), 7))
        np.testing.assert_array_equal(keys[:, 2], np.tile(np.arange(7), 5))
        for frame in FRAMES:
            np.testing.assert_allclose(one[frame], grouped[frame], atol=1e-4, rtol=1e-4)
        poisoned = dict(data, target=np.full_like(data['target'], np.nan), valid=~data['valid'],
                        waiting=~data['waiting'], walker=np.full_like(data['walker'], -1), kind=data['kind']+99)
        after, _ = predict(model, InputView(poisoned), FlatRasters(), 32)
        for frame in FRAMES:
            np.testing.assert_array_equal(grouped[frame], after[frame])

    def test_every_numerical_coordinate_and_fixed_threshold(self):
        arrays = arrays_fixture()
        keys = InputView(data_fixture(2)).keys()
        result = numerical_comparison(arrays, keys)
        self.assertTrue(result['passed']); self.assertEqual(len(result['comparisons']), 12)
        self.assertEqual(result['policy']['atol'], 1e-4)
        # Last step, factual future and an unmatched slot must still be checked.
        arrays['gpu32_body'][0, 6, 15, 5, 1] = np.float32(1e-4)
        self.assertTrue(numerical_comparison(arrays, keys)['passed'])
        arrays['gpu32_body'][0, 6, 15, 5, 1] = np.nextafter(np.float32(1e-4), np.float32(np.inf))
        result = numerical_comparison(arrays, keys)
        self.assertFalse(result['passed'])
        row = result['comparisons']['gpu32_vs_cpu1/body']
        self.assertEqual(row['first_failed_index'], [0, 6, 15, 5, 1])
        self.assertEqual(row['failed_by_step'], [0]*15+[1])
        self.assertEqual(row['failed_coordinates'], 1)
        self.assertTrue(result['comparisons']['gpu32_vs_cpu1/world']['passed'])
        for value in (np.full_like(arrays['cpu1_world'], np.nan), arrays['cpu1_world'].astype(np.float64),
                      arrays['cpu1_world'][:1]):
            with self.assertRaises(ValueError): numerical_comparison(dict(arrays, cpu1_world=value), keys)
        arrays.pop('gpu1_world')
        with self.assertRaises(ValueError): numerical_comparison(arrays, keys)

    def test_artifact_roundtrip_stale_binding_and_overwrite(self):
        keys = InputView(data_fixture(2)).keys()
        binding = dict(model='synthetic', checkpoint='fixture', source='fixture', data='fixture', numeric_review='fixture')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'artifact'
            arrays = arrays_fixture()
            written = write_artifact(out, arrays, keys, binding, 'numerical')
            loaded, restored = read_artifact(out, binding, keys, 'numerical')
            self.assertEqual(written, loaded)
            for name in arrays: np.testing.assert_array_equal(arrays[name], restored[name])
            with self.assertRaises(FileExistsError): write_artifact(out, arrays, keys, binding, 'numerical')
            for field in binding:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    read_artifact(out, dict(binding, **{field: 'stale'}), keys, 'numerical')
            bad = copy.deepcopy(written); bad['numerical']['comparisons']['gpu32_vs_cpu1/body']['max_absolute_error'] = 1.
            atomic_json(out/'result.json', bad)
            with self.assertRaisesRegex(ValueError, 'summary'): read_artifact(out, binding, keys, 'numerical')
            atomic_json(out/'result.json', written)
            with (out/'cpu1_body.npy').open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): read_artifact(out, binding, keys, 'numerical')
            prediction = Path(folder)/'prediction'
            write_artifact(prediction, {'world': arrays['cpu1_world']}, keys, binding, 'prediction')
            read_artifact(prediction, binding, keys, 'prediction')

    def test_changed_keys_rejected_even_with_recomputed_file_digest(self):
        keys = InputView(data_fixture(2)).keys()
        array = arrays_fixture()['cpu1_world']
        for corrupt in (keys[::-1], keys[:7], np.concatenate((keys[:7], keys[:7])),
                        np.concatenate((keys[7:], keys[:7])), keys.astype(np.int32)):
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)/'artifact'
                result = write_artifact(out, {'world': array}, keys, {'fixture': True}, 'prediction')
                with (out/'keys.npy').open('wb') as f: np.save(f, corrupt, allow_pickle=False)
                result['files']['keys.npy'] = digest(out/'keys.npy'); atomic_json(out/'result.json', result)
                with self.assertRaises(ValueError): read_artifact(out, {'fixture': True}, keys, 'prediction')

    def test_bound_checkpoint_loading_and_final_selection(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            payload = dict(model=model.state_dict(), config=model.config.to_dict(), step=20000, binding={'fixture': True})
            torch.save(payload, path)
            expected = dict(name='paired', sha256=digest(path), config=payload['config'], step=20000,
                            training_binding={'fixture': True})
            restored = load_model(path, expected, 'cpu')
            self.assertFalse(restored.training)
            for key, value in model.state_dict().items(): torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
            for key, value in (('sha256', 'stale'), ('config', {}), ('step', 1000), ('training_binding', {}), ('name', 'unknown')):
                with self.subTest(field=key), self.assertRaises(ValueError): load_model(path, dict(expected, **{key: value}), 'cpu')
            payload['model']['head.2.weight'] = payload['model']['head.2.weight'].double()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                load_model(path, dict(expected, sha256=digest(path)), 'cpu')

    def test_actual_frozen_m5_header_and_review_target_refusals(self):
        record, rh = require_record('F004')
        additional = json.loads((BASE/'additional_inputs.json').read_text())
        bound = {}
        header = selected_checkpoint('m5', bound, record, rh, additional)
        self.assertEqual(header['sha256'], 'caf37b6ec182207bbb900fc4bf53fd1e8199e0765371a7771dfd0b5d508e171e')
        self.assertEqual(header['config']['dropout'], .1)
        self.assertTrue(header['config']['residual'])
        # These review fixtures are ephemeral software inputs, never scientific attestations.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder = Path(folder)
            atomic_json(folder/'result.json', {'fixture': True})
            review = dict(research_id='fixture', result='result.json', result_sha256=digest(folder/'result.json'),
                          reviewed_at='fixture', status='passed', sources=[dict(url='fixture', version='fixture',
                          sections='fixture', consequence='fixture')], measurements={'fixture': True},
                          uncertainty=['fixture'], explanations=['fixture'], next_step='fixture')
            atomic_json(folder/'review.json', review)
            reviewed({}, folder/'review.json', folder/'result.json')
            with self.assertRaises(ValueError): reviewed({}, folder/'review.json', folder/'other.json')
            atomic_json(folder/'review.json', dict(review, status='technical_blocked'))
            with self.assertRaises(ValueError): reviewed({}, folder/'review.json', folder/'result.json')
        self.assertFalse(torch.cuda.is_initialized())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    _, rh = require_record('F004')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    atomic_json(out/'result.json', dict(research_id='F004', phase='inference_software', record_sha256=rh,
        created_at=datetime.now(timezone.utc).isoformat(), status='passed' if result.wasSuccessful() else 'technical_blocked',
        checks=result.testsRun, failures=[dict(test=str(t), traceback=e) for t, e in result.errors+result.failures],
        source_sha256=implementation_sources(), preimplementation_sha256=digest(BASE/'inference_preimplementation.json'),
        limits='CPU synthetic inference, numerical-comparator and provenance fixtures, plus frozen M5 header inspection only. No actual scientific forecast or CPU/GPU selected-checkpoint check. Serial scientific supervisor still required.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
