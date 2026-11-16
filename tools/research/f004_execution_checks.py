"""CPU checks of F004 admission, actual subprocess supervision and update plumbing."""
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
from unittest.mock import patch

import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from f004_runtime import (BASE, ROOT, admit, close_interval, fresh_paths, gpu_device, gpu_idle,
                         inventory, sources, supervision, validate_pre)
from f004_software import FlatRasters, fixture, flat, small_model
from f004_train import child_admission, update, validate
from go2nav.data.paired_forecasts import extract
from go2nav.research import atomic_json, digest, require_record, verify_files


class Checks(unittest.TestCase):
    def test_gpu_and_fresh_output_refusals(self):
        with patch('torch.cuda.is_available',return_value=False), self.assertRaisesRegex(ValueError,'CPU fallback disabled'):
            gpu_device()
        gpu_idle([dict(process_list=[dict(process_info='No running processes detected')])])
        for report in ([],[dict(process_list=[dict(process_info=dict(pid=123))])]):
            with self.assertRaises(ValueError):
                gpu_idle(report)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                fresh_paths(folder)
            fresh_paths(Path(folder)/'fresh')

    def test_preconsultation_complete_exact_bindings(self):
        _,rh = require_record('F004')
        bound,code = {'result':'fixed'},{'source':'fixed'}
        pre = dict(arm='factual',seed=0,record_sha256=rh,inputs_sha256=bound,source_sha256=code,
                   source_commit='commit',consulted_at='now',consequences='reviewed',
                   sources=[dict(url='primary',version='1',sections='method',consequence='decision')])
        validate_pre(pre,'factual',bound,'commit',code)
        changes = dict(arm='paired',seed=1,record_sha256='stale',inputs_sha256={},source_sha256={},
                       source_commit='other',sources=[],consulted_at=None,consequences=None)
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_pre(dict(pre,**{key:value}),'factual',bound,'commit',code)

    def test_ledger_is_serial_terminal_only_and_has_no_hour_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder)/'ledger.json'
            atomic_json(ledger,[dict(id='prior',research_id='old',start=0,end=72*3600)])
            row = admit(ledger,Path(folder)/'run','factual')
            self.assertIsNone(row['end'])
            with self.assertRaises(ValueError):
                admit(ledger,Path(folder)/'other','branch')
            class Live:
                def poll(self): return None
            with self.assertRaises(ValueError):
                close_interval(ledger,row,Live())
            close_interval(ledger,row,None)
            self.assertIsNotNone(json.loads(ledger.read_text())[-1]['end'])
            with self.assertRaises(ValueError):
                close_interval(ledger,row,None)

    def test_changed_files_invalid_arm_and_child_protocol(self):
        with self.assertRaises(ValueError):
            inventory('unplanned')
        record,rh = require_record('F004')
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'source'
            p.write_text('original')
            hashes = {'source':digest(p)}
            p.write_text('changed')
            with self.assertRaises(ValueError):
                verify_files(folder,hashes)
            p = Path(folder)/'admission.json'
            admission = dict(record_sha256=rh,settings=record['protocol']['training'],
                             model=record['protocol']['model'],arm='factual',seed=0)
            for changed in (dict(admission,record_sha256='stale'),
                            dict(admission,settings=dict(admission['settings'],steps=10)),
                            dict(admission,model=dict(admission['model'],dim=8)),
                            dict(admission,arm='other'),dict(admission,seed=1)):
                atomic_json(p,changed)
                with self.assertRaises(ValueError):
                    child_admission(p)

    def test_real_owned_clean_exit_and_stall_termination(self):
        common = supervision()
        _,_,owned = common.inherited()
        for delay,stall,expected in ((.1,1.,False),(10.,.15,True)):
            argv = [sys.executable,'-c',f'import time; time.sleep({delay})']
            child = subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process,actions = psutil.Process(child.pid),[]
            created = process.create_time()
            try:
                result = common.wait_progress(child,process,created,argv,ROOT,[],actions,
                                              stall_seconds=stall,poll_seconds=.02,term_seconds=.3,kill_seconds=.3)
                self.assertEqual(result,expected)
                self.assertIsNotNone(child.poll())
                self.assertEqual(bool(actions),expected)
            finally:
                if child.poll() is None:
                    owned.stop_owned(child,process,created,argv,ROOT,actions,.3,.3)

    def test_real_progress_and_wrong_identity_refusal(self):
        common = supervision()
        _,_,owned = common.inherited()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress'
            script = 'import pathlib,time,sys\np=pathlib.Path(sys.argv[1])\nfor i in range(7):\n p.write_text(str(i));time.sleep(.08)\n'
            argv = [sys.executable,'-c',script,str(path)]
            child = subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process,actions = psutil.Process(child.pid),[]
            created = process.create_time()
            try:
                with self.assertRaises(RuntimeError):
                    owned.stop_owned(child,process,created-1,argv,ROOT,actions,.3,.3)
                self.assertEqual(actions,[])
                self.assertIsNone(child.poll())
                result = common.wait_progress(child,process,created,argv,ROOT,[path],actions,
                                              stall_seconds=.3,poll_seconds=.02,term_seconds=.3,kill_seconds=.3)
                self.assertFalse(result)
                self.assertEqual(child.returncode,0)
                self.assertEqual(path.read_text(),'6')
            finally:
                if child.poll() is None:
                    owned.stop_owned(child,process,created,argv,ROOT,actions,.3,.3)

    def test_eight_microbatch_update_and_monitoring_do_not_select_weights(self):
        record,_ = require_record('F004')
        settings = dict(record['protocol']['training'])
        out,_ = extract(*fixture())
        # One tiny CPU unit update with eight microbatches; no scientific data or training fallback.
        raw = flat(out,(0,1)*8)
        order = np.tile(np.arange(6),(16,1))
        model = small_model()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
        settings['microbatch_pairs'] = 1
        with sdpa_kernel(SDPBackend.MATH):
            values = update(model,optimizer,raw,order,FlatRasters(),'paired',settings,0)
        self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(model.training)
        class TinyCache:
            n = 1
            def take(self,points,futures): return flat(out,tuple(futures))
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with sdpa_kernel(SDPBackend.MATH):
            result = validate(model,TinyCache(),FlatRasters(),settings)
        self.assertTrue(model.training)
        self.assertEqual(result['selection'],'monitoring_only_final_step_selected')
        for k,v in before.items():
            torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    _,rh = require_record('F004')
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    atomic_json(out/'result.json',dict(research_id='F004',phase='execution_software',
                status='passed' if result.wasSuccessful() else 'technical_blocked',
                created_at=datetime.now(timezone.utc).isoformat(),record_sha256=rh,source_sha256=sources(),
                checks=result.testsRun,failures=[dict(test=str(t),traceback=e) for t,e in result.errors+result.failures],
                preimplementation_sha256=digest(BASE/'execution_preimplementation.json'),
                limits='CPU admission/supervision and synthetic update checks. No scientific training, actual GPU numerical admission or forecast evaluation.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
