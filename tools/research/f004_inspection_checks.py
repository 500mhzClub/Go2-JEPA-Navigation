"""CPU software fixtures for terminal F004 training inspection, including corruptions."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from f004_inspect_training import (BASE, ROOT, SOURCES, equal_tree, inspect_checkpoint, inspect_logs,
                                   inspect_operation, learning_rate, original_child_absent, replay_sampler)
from f004_software import FlatRasters, fixture, flat
from f004_train import update
from go2nav.data.paired_forecasts import extract
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, require_record, verify_files
from go2nav.training.paired_forecasts import Sampler, save_checkpoint


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        record, rh = require_record('F004')
        cls.settings = dict(record['protocol']['training'], steps=3, warmup_steps=1, pairs_per_batch=2,
                            microbatch_pairs=1, log_every=1, validation_every=1, validation_points=2)
        torch.manual_seed(0)
        cls.model = ObjectJEPA(ObjectConfig(dim=8, depth=1, heads=1, head_dim=8, mlp=16,
                                           hidden=12, dropout=0., residual=True))
        cls.initial = {k: v.clone() for k, v in cls.model.state_dict().items()}
        cls.initial_rng = torch.get_rng_state().clone()
        cls.binding = dict(record_sha256=rh, arm='paired', seed=0, inputs_sha256={'fixture': 'synthetic'},
                           source_sha256={'fixture': 'synthetic'}, model=cls.model.config.to_dict(), settings=cls.settings)
        cls.sampler_rng = replay_sampler(3, cls.settings, [0, 1, 2, 3])
        optimizer = torch.optim.AdamW(cls.model.parameters(), lr=cls.settings['lr'],
                                     weight_decay=cls.settings['weight_decay'])
        sampler = Sampler(3, 0, 'paired')
        data, _ = extract(*fixture())
        cls.checkpoints, cls.logs = [], []
        with tempfile.TemporaryDirectory() as folder, sdpa_kernel(SDPBackend.MATH):
            path = Path(folder)/'checkpoint.pt'
            for step in range(4):
                if step:
                    points, futures, order = sampler.draw(cls.settings['pairs_per_batch'])
                    raw = flat(data, tuple(futures))
                    values = update(cls.model, optimizer, raw, order, FlatRasters(), 'paired', cls.settings, step-1)
                    cls.logs.append(dict(step=step, elapsed_seconds=float(step), **values))
                save_checkpoint(path, cls.model, optimizer, sampler, step, cls.binding)
                cls.checkpoints.append(torch.load(path, map_location='cpu', weights_only=True))

    def check(self, checkpoint, step=3, require_cuda=False):
        return inspect_checkpoint(checkpoint, self.model, self.initial, self.initial_rng, self.binding,
                                  step, 3, self.sampler_rng[step], require_cuda=require_cuda)

    def test_complete_real_optimizer_and_rng_fixture(self):
        for step, checkpoint in enumerate(self.checkpoints):
            result = self.check(checkpoint, step)
            self.assertEqual(result['batchnorm_forwards'], step*32)
        self.assertEqual(self.check(self.checkpoints[-1])['lr'], 0.)
        self.assertTrue(any(not torch.equal(v, self.checkpoints[-1]['model'][k]) for k, v in self.initial.items()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_identity_scheduler_sampler_and_rng_corruptions(self):
        changes = dict(step=2, binding={}, config={}, scheduler={},
                       sampler=dict(n=3, arm='paired', rng=self.sampler_rng[2]),
                       torch_rng=torch.zeros_like(self.initial_rng), cuda_rng=[torch.ones(2)])
        for key, value in changes.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.check(dict(self.checkpoints[-1], **{key: value}))
        with self.assertRaisesRegex(ValueError, 'device RNG'):
            self.check(self.checkpoints[-1], require_cuda=True)
        gpu_header = dict(self.checkpoints[-1], cuda_rng=[torch.zeros(16, dtype=torch.uint8)])
        self.check(gpu_header, require_cuda=True)  # Saved byte state shape only; not GPU RNG/restart validity.

    def test_model_and_optimizer_corruptions(self):
        def tensor_nan(c): c['model']['head.2.weight'][0, 0] = float('nan')
        def tensor_dtype(c): c['model']['head.2.weight'] = c['model']['head.2.weight'].double()
        def tensor_shape(c): c['model']['head.2.weight'] = c['model']['head.2.weight'][:1]
        def missing_tensor(c): del c['model']['head.2.bias']
        def masking_changed(c): c['model']['mask_token'][0] += 1
        def bn_changed(c):
            name = next(k for k in c['model'] if k.endswith('num_batches_tracked'))
            c['model'][name] -= 1
        def optimizer_missing(c): c['optimizer']['state'].pop(next(iter(c['optimizer']['state'])))
        def optimizer_step(c): next(iter(c['optimizer']['state'].values()))['step'] -= 1
        def optimizer_negative(c): next(iter(c['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
        def optimizer_shape(c): next(iter(c['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
        def optimizer_nan(c): next(iter(c['optimizer']['state'].values()))['exp_avg'].fill_(float('inf'))
        def optimizer_lr(c): c['optimizer']['param_groups'][0]['lr'] = 1e-9
        for mutation in (tensor_nan, tensor_dtype, tensor_shape, missing_tensor, masking_changed, bn_changed,
                         optimizer_missing, optimizer_step, optimizer_negative, optimizer_shape, optimizer_nan, optimizer_lr):
            bad = copy.deepcopy(self.checkpoints[-1]); mutation(bad)
            with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError): self.check(bad)
        bad = copy.deepcopy(self.checkpoints[0]); bad['model']['head.0.weight'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Initialization'): self.check(bad, 0)

    def test_log_completeness_finiteness_and_monitoring_only(self):
        validation = [dict(step=s, points=2, counts=[20, 18, 10], endpoint_error_m=[.1, .2, .3],
                           selection='monitoring_only_final_step_selected') for s in (1, 2, 3)]
        result = inspect_logs(self.logs, validation, self.settings, 4)
        self.assertEqual(result['training_rows'], 3)
        for logs in (self.logs[:-1], self.logs[::-1], self.logs+[self.logs[-1]]):
            with self.assertRaises(ValueError): inspect_logs(logs, validation, self.settings, 4)
        for field, value in (('loss', float('nan')), ('trajectory_targets', 0), ('elapsed_seconds', 0), ('lr', 99)):
            bad = copy.deepcopy(self.logs); bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect_logs(bad, validation, self.settings, 4)
        for field, value in (('selection', 'best'), ('counts', [20, 18, 9]), ('points', 3),
                             ('endpoint_error_m', [0, -1, 2])):
            bad = copy.deepcopy(validation); bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect_logs(self.logs, bad, self.settings, 4)

    def test_owned_process_and_terminal_operation_integrity(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], cwd=ROOT)
        created = psutil.Process(child.pid).create_time()
        identity = dict(pid=child.pid, creation_time=created, cwd=str(ROOT),
                        argv=[sys.executable, str(ROOT/'tools/research/f004_train.py'), '--admission',
                              str(ROOT/'outputs/test/operations/admission.json')])
        try:
            with self.assertRaisesRegex(ValueError, 'remains present'): original_child_absent(identity)
            self.assertEqual(original_child_absent(dict(identity, creation_time=created-1)), 'pid_reused_original_absent')
        finally:
            child.terminate(); child.wait(timeout=5)
        self.assertEqual(original_child_absent(identity), 'pid_absent')
        interval = dict(id='fixture', start=created, end=None, arm='paired')
        admission = dict(arm='paired', seed=0, record_sha256='fixture', interval=interval,
                         cwd=str(ROOT), science_out=str(ROOT/'outputs/test/science'), supervisor_created=created-1)
        operation = dict(run_identity=dict(arm='paired', seed=0), record_sha256='fixture',
                         status='awaiting_literature_review', returncode=0, child_terminal=True,
                         inputs_unchanged=True, stalled=False, actions=[], errors=[], interval_id='fixture', elapsed_seconds=1.)
        closed = dict(interval, end=created+1, closure='F004 supervisor: owned child terminal confirmed')
        inspect_operation(operation, admission, identity, [closed])
        for field, value in (('returncode', -15), ('child_terminal', False), ('inputs_unchanged', False),
                             ('stalled', True), ('actions', ['TERM']), ('errors', ['failure']), ('interval_id', 'other')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect_operation(dict(operation, **{field: value}), admission, identity, [closed])
        for rows in ([], [closed, closed], [interval], [dict(closed, arm='branch')]):
            with self.assertRaises(ValueError): inspect_operation(operation, admission, identity, rows)

    def test_exact_nested_state_and_hash_binding(self):
        checkpoint = self.checkpoints[-1]
        self.assertTrue(equal_tree(checkpoint, copy.deepcopy(checkpoint)))
        bad = copy.deepcopy(checkpoint); bad['optimizer']['param_groups'][0]['params'].reverse()
        self.assertFalse(equal_tree(checkpoint, bad))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'artifact'; path.write_text('recorded')
            hashes = {'artifact': digest(path)}
            verify_files(folder, hashes)
            path.write_text('changed')
            with self.assertRaises(ValueError): verify_files(folder, hashes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    _, rh = require_record('F004')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    files = (*SOURCES, 'tools/research/f004_software.py', 'tools/research/f004_train.py',
             'tools/research/f004_runtime.py', 'go2nav/research.py', 'go2nav/data/paired_forecasts.py',
             'go2nav/training/paired_forecasts.py', 'go2nav/models/object_jepa.py', 'go2nav/models/lewm.py')
    atomic_json(out/'result.json', dict(research_id='F004', phase='inspection_software', record_sha256=rh,
                created_at=datetime.now(timezone.utc).isoformat(), status='passed' if result.wasSuccessful() else 'technical_blocked',
                source_sha256={p: digest(ROOT/p) for p in files},
                preimplementation_sha256=digest(BASE/'inspection_preimplementation.json'), checks=result.testsRun,
                failures=[dict(test=str(t), traceback=e) for t, e in result.errors+result.failures],
                limits='Tiny synthetic CPU optimizer fixture and terminal-integrity checks. No scientific checkpoint inspected or forecast scored.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
