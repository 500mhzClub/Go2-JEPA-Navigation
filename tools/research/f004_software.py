"""Declared F004 CPU software checks. This is not a scientific training run."""
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
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from go2nav.data.paired_forecasts import Cache, FUTURE, SCHEMA, extract, initial_match
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, require_record, verify_files
from go2nav.training.paired_forecasts import (Sampler, counts, features, forecast, objective, rate,
                                             restore_checkpoint, save_checkpoint, targets)

SOURCES = ['go2nav/data/paired_forecasts.py', 'go2nav/training/paired_forecasts.py',
           'go2nav/models/object_jepa.py', 'go2nav/models/lewm.py', 'go2nav/training/object_jepa.py',
           'go2nav/planning/wall_raster.py', 'go2nav/research.py', 'tools/research/f004_software.py']


def fixture():
    b = dict(t0=np.full(6, 8, np.int32), contact=np.full(6, -1, np.int16),
             requested=np.zeros((6, 72, 3), np.float32), pose=np.zeros((6, 73, 3), np.float32),
             xy=np.full((6, 73, 6, 2), np.nan, np.float32), seen=np.zeros((6, 73, 6), bool),
             id=np.full((6, 73, 6), -1, np.int32), velocity=np.zeros((6, 73, 6, 2), np.float32),
             walkers_xy=np.full((6, 73, 4, 2), np.nan, np.float32),
             waiting=np.zeros((6, 73, 4), bool), in_sight=np.zeros((6, 73, 4), bool))
    b['xy'][:, :, 0] = (1, 0)
    b['xy'][:, :, 1] = (2, 2)  # Unmatched observed slot must remain in model input.
    b['seen'][:, :, :2] = True
    b['id'][:, :, 0], b['id'][:, :, 1] = 4, 9
    b['walkers_xy'][:, :, 0] = (1, 0)
    b['in_sight'][:, :, 0] = True
    for a in range(6):
        b['walkers_xy'][a, 9:, 0, 0] += (a+1)*.01*np.arange(1, 65)
        b['pose'][a, 9:, 0] = a*.01*np.arange(1, 65)
        b['requested'][a, 8:, 0] = a*.1
    b['contact'][2] = 16
    b['walkers_xy'][2, 24:] = np.nan
    parent = dict(pose=np.zeros((90, 3), np.float32), requested=np.full((90, 3), .07, np.float32),
                  walkers_xy=np.full((90, 4, 2), np.nan, np.float32))
    parent['walkers_xy'][:, 0] = (1, 0)
    parent['walkers_xy'][9:, 0, 0] += .2*np.arange(1, 82)
    meta = dict(index=0, maze=0, kind='reactive')
    return b, parent, meta


class FlatRasters:
    index = {0: 0}

    def lookup(self, layout, points):
        return torch.ones(points.shape[:-1], device=points.device)


def flat(array, futures=(0, 1)):
    return {k: (v[np.zeros(len(futures), int), futures] if k in ('target', 'valid', 'contact', 'future_pose', 'future_commands')
             else np.repeat(v, len(futures), axis=0)) for k, v in array.items()}


def small_model():
    torch.manual_seed(17)
    model = ObjectJEPA(ObjectConfig(dim=16, depth=1, heads=1, head_dim=16, mlp=32, hidden=24, dropout=0, residual=True))
    # Exercise nontrivial learned dynamics and action-conditioning, rather than a zero residual head.
    torch.nn.init.normal_(model.head[-1].weight, std=.03)
    for layer in model.predictor.layers:
        torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=.03)
    return model


class Checks(unittest.TestCase):
    def test_extract_identity_time_and_contact(self):
        b, parent, meta = fixture()
        out, coverage = extract(b, parent, meta)
        self.assertEqual(coverage['retained'], 1)
        np.testing.assert_array_equal(out['walker'], [[0, -1, -1, -1, -1, -1]])
        self.assertTrue(out['history_seen'][0, :, 1].all())
        np.testing.assert_array_equal(out['target'][0, 1, :, 0], b['walkers_xy'][1, 8+FUTURE, 0])
        np.testing.assert_array_equal(out['target'][0, 6, :, 0], parent['walkers_xy'][8+FUTURE, 0])
        self.assertTrue(out['valid'][0, 2, :3, 0].all())
        self.assertFalse(out['valid'][0, 2, 3:, 0].any())  # Contact at endpoint16 is censored.
        self.assertFalse(out['valid'][..., 1:].any())
        np.testing.assert_array_equal(out['future_commands'][0, 6], parent['requested'][8:72])
        b['seen'][:, 9:] = False
        b['id'][:, 9:] = -1
        again, _ = extract(b, parent, meta)
        np.testing.assert_array_equal(again['valid'], out['valid'])
        _, coverage = extract(b, {k: v[:72] for k, v in parent.items()}, meta)
        self.assertEqual(coverage['parent_future_missing'], 1)

    def test_matching_ties_visibility_and_false_positives(self):
        b, _, _ = fixture()
        def match():
            return initial_match(b['xy'][:, 8], b['seen'][:, 8], b['id'][:, 8],
                                 b['walkers_xy'][:, 8], b['in_sight'][:, 8], b['pose'][:, 8])[0]
        self.assertTrue((match()[:, 0] == 0).all())
        b['xy'][:, 8, 1] = (1, 0)
        self.assertTrue((match() == -1).all())
        b['xy'][:, 8, 1] = (2, 2)
        b['in_sight'][:, 8, 0] = False
        self.assertTrue((match() == -1).all())

    def test_feature_label_separation_and_slot_permutation(self):
        out, _ = extract(*fixture())
        raw = flat(out)
        rasters = FlatRasters()
        order = np.array([[2, 1, 5, 0, 3, 4]]*2)
        inputs = features(raw, rasters, 'cpu', order)
        altered = copy.deepcopy(raw)
        for k in ('target', 'valid', 'walker', 'waiting', 'kind', 'contact'):
            altered[k] = np.zeros_like(altered[k])
        other = features(altered, rasters, 'cpu', order)
        for k in inputs:
            torch.testing.assert_close(inputs[k], other[k], rtol=0, atol=0)
        self.assertTrue((inputs['objects'][:, :, 1, 3] == 1).all())  # False-positive track retained.
        target, valid = targets(raw, inputs, order)
        self.assertTrue(valid[:, :, 3].all())
        self.assertFalse(valid[:, :, [0, 1, 2, 4, 5]].any())
        np.testing.assert_allclose(target[:, :, 3], raw['target'][:, :, 0])

    def test_future_suffix_causality_and_identical_actions(self):
        out, _ = extract(*fixture())
        raw = flat(out, (0, 0))
        rasters, model = FlatRasters(), small_model().eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            inputs = features(raw, rasters, 'cpu')
            prediction = forecast(model, inputs, rasters)
            torch.testing.assert_close(prediction[0], prediction[1], rtol=0, atol=0)
            changed = copy.deepcopy(raw)
            changed['future_commands'][:, 16:] += .5
            changed['future_pose'][:, 4:] += .7
            later = forecast(model, features(changed, rasters, 'cpu'), rasters)
            torch.testing.assert_close(prediction[:, :4], later[:, :4], rtol=0, atol=0)
            # The changed tape really influences a later forecast with nonzero learned parameters.
            self.assertGreater(float((prediction[:, 4:]-later[:, 4:]).abs().max()), 1e-7)

    def test_objective_effect_and_microbatch_gradients(self):
        y = torch.zeros(4, 16, 6, 2)
        y[1, :, :, 0] = 2
        y[3, :, :, 0] = 1
        valid = torch.ones(4, 16, 6, dtype=torch.bool)
        valid[2:, 8:] = False
        p = y.clone().requires_grad_()
        loss, _, response = objective(p, y, valid, 'paired')
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        p = torch.zeros_like(y, requires_grad=True)
        full, trajectory, response = objective(p, y, valid, 'paired')
        self.assertGreater(float(response), 0)
        full.backward()
        split = torch.zeros_like(y, requires_grad=True)
        total = 0.
        for k in (0, 2):
            value, _, _ = objective(split[k:k+2], y[k:k+2], valid[k:k+2], 'paired', counts(valid))
            value.backward()
            total += float(value.detach())
        self.assertAlmostEqual(float(full.detach()), total, places=6)
        torch.testing.assert_close(p.grad, split.grad, rtol=0, atol=0)
        self.assertAlmostEqual(float(objective(p, y, valid, 'branch')[0].detach()), float(trajectory))

    def test_sampler_arm_matching_and_restart(self):
        a, b, c = [Sampler(123, 0, arm) for arm in ('factual', 'branch', 'paired')]
        for _ in range(3):
            aa, bb, cc = [s.draw(17) for s in (a, b, c)]
            for x, y in zip(bb, cc):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(aa[0], bb[0])
            np.testing.assert_array_equal(aa[2], bb[2])
            np.testing.assert_array_equal(bb[2][::2], bb[2][1::2])
            self.assertTrue((bb[1][::2] != bb[1][1::2]).all())
            self.assertTrue((aa[1] == 6).all())
        state = b.state_dict()
        expected = b.draw(9)
        b.load_state_dict(state)
        for x, y in zip(expected, b.draw(9)):
            np.testing.assert_array_equal(x, y)
        with self.assertRaises(ValueError):
            Sampler(123, 0, 'invalid')

    def test_optimizer_rng_scheduler_resume_and_binding(self):
        model = small_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sampler = Sampler(9, 0, 'paired')
        binding = dict(record='fixed', source='fixed', data='fixed', arm='paired')
        def update(m, opt, s, step):
            s.draw(3)
            opt.zero_grad(set_to_none=True)
            opt.param_groups[0]['lr'] = 3e-4*rate(step)
            value = sum((p*torch.rand_like(p)).sum() for p in m.parameters())
            value.backward()
            opt.step()
        update(model, optimizer, sampler, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            save_checkpoint(path, model, optimizer, sampler, 1, binding)
            update(model, optimizer, sampler, 1)
            expected_sample = sampler.draw(4)
            restored = small_model()
            opt = torch.optim.AdamW(restored.parameters(), lr=3e-4)
            sample = Sampler(9, 0, 'paired')
            self.assertEqual(restore_checkpoint(path, restored, opt, sample, binding), 1)
            update(restored, opt, sample, 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            for x, y in zip(expected_sample, sample.draw(4)):
                np.testing.assert_array_equal(x, y)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, restored, opt, sample, dict(binding, data='changed'))
        self.assertEqual(rate(19999), 0)
        self.assertEqual(rate(499), 1)

    def test_cache_schema_and_frozen_bytes(self):
        out, _ = extract(*fixture())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            for k, v in out.items():
                np.save(root/'train'/(k+'.npy'), v)
            cache = Cache(root, 'train')
            raw = cache.take([0, 0], [6, 2])
            np.testing.assert_array_equal(raw['target'][0], out['target'][0, 6])
            with self.assertRaises(ValueError):
                cache.take([0], [-1])
            path = root/'train/t0.npy'
            binding = {'train/t0.npy': digest(path)}
            verify_files(root, binding)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_files(root, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    _, rh = require_record('F004')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    atomic_json(out/'result.json', dict(research_id='F004', phase='scientific_primitives_software',
                status='passed' if result.wasSuccessful() else 'technical_blocked',
                created_at=datetime.now(timezone.utc).isoformat(), record_sha256=rh,
                source_sha256={p: digest(ROOT/p) for p in SOURCES}, checks=result.testsRun,
                failures=[dict(test=str(t), traceback=e) for t, e in result.errors+result.failures],
                scope='Synthetic data/features/loss/sampling/recovery and cache schema. CPU optimizer fixture is a unit check, not scientific training. Actual export independent verification and GPU supervisor/numerics are separate gates.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
