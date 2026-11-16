"""CPU fixtures for the F004 evaluator's serial execution and scoring boundaries."""
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import psutil
import torch

import f004_evaluation as execution
from f004_evaluation import (BASE, ROOT, admit, child_admission, compute, execution_sources,
                             score, stage_evidence, validate_pre)
from f004_inference_checks import data_fixture
from f004_runtime import close_interval, supervision
from go2nav.eval.paired_forecast_inference import NUMERIC_POLICY, InputView, read_artifact
from go2nav.research import atomic_json, digest, require_record


def pre_fixture(identity, binding, bound=None):
    return dict(run_identity=identity, binding=binding, inputs_sha256=bound or {}, source_sha256={},
                source_commit='fixture', consulted_at='fixture', consequences=['fixture'],
                sources=[dict(url='fixture', version='fixture', sections='fixture', consequence='fixture')])


class Checks(unittest.TestCase):
    def test_exact_preconsultation_and_invalid_stage(self):
        identity, binding = dict(model='m5', phase='numerical'), {'fixture': True}
        pre = pre_fixture(identity, binding)
        validate_pre(pre, identity, binding, {}, {}, 'fixture')
        for key, value in (('run_identity', {}), ('binding', {}), ('inputs_sha256', {'missing': 'input'}),
                           ('source_sha256', {'changed': 'source'}), ('source_commit', 'changed'),
                           ('sources', []), ('consequences', []), ('consulted_at', None)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_pre(dict(pre, **{key: value}), identity, binding, {}, {}, 'fixture')
        for name, phase in (('unknown', 'numerical'), ('m5', 'training')):
            with self.assertRaises(ValueError): stage_evidence(name, phase)

    def test_evaluation_interval_serial_elapsed_only_and_terminal_close(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder)/'ledger.json'
            atomic_json(ledger, [dict(id='historical', start=0, end=100*3600)])
            identity = dict(model='m5', phase='numerical')
            interval = admit(ledger, Path(folder)/'out', identity)
            self.assertEqual(interval['model'], 'm5'); self.assertNotIn('arm', interval)
            self.assertNotIn('gpu_hours_limit', interval)
            with self.assertRaises(ValueError): admit(ledger, Path(folder)/'other', identity)
            class Live:
                def poll(self): return None
            with self.assertRaises(ValueError): close_interval(ledger, interval, Live())
            close_interval(ledger, interval, None)
            self.assertIsNotNone(json.loads(ledger.read_text())[-1]['end'])

    def test_parent_and_ledger_child_admission(self):
        identity, binding = dict(model='m5', phase='numerical'), {'fixture': True}
        parent = psutil.Process()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            interval = dict(id='fixture', start=0, end=None)
            ledger = folder/'ledger.json'; atomic_json(ledger, [interval])
            admission = dict(cwd=str(ROOT), run_identity=identity, binding=binding, inputs_sha256={},
                             source_sha256={}, supervisor_pid=parent.pid, supervisor_created=parent.create_time(),
                             ledger=str(ledger), interval=interval)
            path = folder/'admission.json'; atomic_json(path, admission)
            with patch.object(execution, 'stage_evidence', return_value=(binding, {})), \
                 patch.object(execution, 'execution_sources', return_value={}), \
                 patch.object(execution.os, 'getppid', return_value=parent.pid):
                self.assertEqual(child_admission(path), admission)
                for key, value in (('supervisor_created', parent.create_time()-1), ('supervisor_pid', -1),
                                   ('binding', {}), ('source_sha256', {'changed': 'source'}), ('cwd', '/wrong')):
                    atomic_json(path, dict(admission, **{key: value}))
                    with self.subTest(field=key), self.assertRaises(ValueError): child_admission(path)
                atomic_json(path, admission); atomic_json(ledger, [dict(interval, end=1)])
                with self.assertRaises(ValueError): child_admission(path)

    def test_real_owned_progress_terminal_and_stall(self):
        common = supervision()
        _, _, owned = common.inherited()
        for moving in (True, False):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'progress'
                script = ('import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\nfor i in range(5):\n p.write_text(str(i));time.sleep(.08)\n'
                          if moving else 'import time;time.sleep(30)')
                argv = [sys.executable, '-c', script, str(path)]
                child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process, actions = psutil.Process(child.pid), []
                created = process.create_time()
                try:
                    with self.assertRaises(RuntimeError): owned.stop_owned(child, process, created-1, argv, ROOT, actions, .3, .3)
                    stalled = common.wait_progress(child, process, created, argv, ROOT, [path], actions,
                                                  stall_seconds=.25, poll_seconds=.02, term_seconds=.3, kill_seconds=.3)
                    self.assertEqual(stalled, not moving)
                    self.assertEqual(bool(actions), not moving)
                    self.assertIsNotNone(child.poll())
                finally:
                    if child.poll() is None: owned.stop_owned(child, process, created, argv, ROOT, actions, .3, .3)

    def dispatch(self, folder, phase='numerical', failure=None):
        view = InputView(data_fixture(64))
        operation = folder/'operations'; operation.mkdir()
        binding = dict(model=dict(name='m5', path='synthetic'), numerical_policy=copy.deepcopy(NUMERIC_POLICY))
        if phase == 'prediction':
            binding.update(numerical_result_sha256='fixture', numerical_review_sha256='fixture',
                           prediction_policy='GPU_float32_math_batch32')
        admission = dict(run_identity=dict(model='m5', phase=phase), binding=binding,
                         science_out=str(folder/'science'), operation_out=str(operation))
        calls = []
        def fake_model(path, expected, device): return str(device)
        def fake_predict(model, selected, rasters, batch, scenes=None, progress=None):
            flavor = ('cpu' if model == 'cpu' else 'gpu')+str(batch)
            calls.append(flavor)
            if failure == 'raise' and flavor == 'gpu32': raise RuntimeError('fixture later-flavor failure')
            keys = selected.keys(scenes)
            arrays = {frame: np.zeros((len(keys)//7, 7, 16, 6, 2), np.float32) for frame in ('body', 'world')}
            if failure == 'numerical' and flavor == 'gpu32': arrays['body'][0, 0, -1, -1, -1] = 1.
            progress(len(keys), len(keys))
            return arrays, keys
        with patch.object(execution, 'load_model', side_effect=fake_model), \
             patch.object(execution, 'predict', side_effect=fake_predict), contextlib.redirect_stdout(io.StringIO()):
            result = compute(admission, view, lambda device: 'synthetic_raster', lambda: torch.device('cuda:0'))
        return admission, view, result, calls

    def test_fixed_dispatch_complete_artifacts_and_numeric_failure(self):
        for phase, failure in (('numerical', None), ('prediction', None), ('numerical', 'numerical')):
            with tempfile.TemporaryDirectory() as folder:
                admission, view, result, calls = self.dispatch(Path(folder), phase, failure)
                self.assertEqual(calls, ['cpu1', 'cpu32', 'gpu1', 'gpu32'] if phase == 'numerical' else ['gpu32'])
                verified, _ = read_artifact(admission['science_out'], admission['binding'], view.keys(), phase)
                self.assertEqual(result, verified)
                self.assertEqual(result['status'], 'technical_blocked' if failure else 'awaiting_literature_review')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'later-flavor'): self.dispatch(folder, failure='raise')
            self.assertEqual(len(list((folder/'operations/completed_flavors').glob('*.npy'))), 6)
            self.assertFalse((folder/'science/result.json').exists())
        bad = dict(run_identity=dict(model='m5', phase='numerical'),
                   binding=dict(model=dict(name='m5'), numerical_policy=NUMERIC_POLICY))
        with self.assertRaisesRegex(ValueError, 'CPU fallback'):
            compute(bad, None, None, lambda: torch.device('cpu'))
        bad['run_identity']['phase'] = 'prediction'
        with self.assertRaisesRegex(ValueError, 'numerical admission'): compute(bad, None, None)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_scoring_uses_all_models_and_no_automatic_review(self):
        data = data_fixture(4)
        predictions = {name: data['target'].copy() for name in ('factual', 'branch', 'paired', 'm5')}
        _, rh = require_record('F004')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); (folder/'data').mkdir(); atomic_json(folder/'data/result.json', {'fixture': True})
            binding = dict(record_sha256=rh, models=['factual', 'branch', 'paired', 'm5'], data_result_sha256=digest(folder/'data/result.json'))
            pre = pre_fixture(dict(model='all', phase='scoring'), binding)
            pre_path = folder/'pre.json'; atomic_json(pre_path, pre)
            fake_git = lambda args, **kwargs: '' if 'status' in args else 'fixture\n'
            with patch.object(execution, 'BASE', folder), patch.object(execution, 'scoring_inputs', return_value=(data, predictions, {})), \
                 patch.object(execution, 'require_execution_software'), patch.object(execution, 'execution_sources', return_value={}), \
                 patch.object(execution.subprocess, 'check_output', side_effect=fake_git), contextlib.redirect_stdout(io.StringIO()):
                result = score(pre_path)
                self.assertEqual(result['status'], 'awaiting_literature_review')
                self.assertEqual(result['report']['models']['paired']['6.4']['trajectory']['factual']['fde']['all']['mean'], 0.)
                self.assertFalse(result['report']['engineering_bars']['passed'])
                self.assertFalse((folder/'evaluation/scoring/review.json').exists())
                with self.assertRaises(ValueError): score(pre_path)
        self.assertFalse(torch.cuda.is_initialized())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    _, rh = require_record('F004')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    atomic_json(out/'result.json', dict(research_id='F004', phase='evaluation_execution_software', record_sha256=rh,
        created_at=datetime.now(timezone.utc).isoformat(), status='passed' if result.wasSuccessful() else 'technical_blocked',
        checks=result.testsRun, failures=[dict(test=str(t), traceback=e) for t, e in result.errors+result.failures],
        source_sha256=execution_sources(), preimplementation_sha256=digest(BASE/'evaluation_execution_preimplementation.json'),
        limits='CPU fixtures of admission, real subprocess ownership, mocked-device stage dispatch and actual synthetic metric scoring. No actual GPU model numeric agreement or physical forecast result.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
