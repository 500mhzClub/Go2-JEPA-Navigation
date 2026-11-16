"""Launch one F004 arm with immutable evidence and an owned progress watchdog."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from f004_runtime import (ARMS, BASE, ROOT, admit, close_interval, fresh_paths, gpu_idle,
                         inventory, need, sources, supervision, validate_pre)
from go2nav.research import atomic_json, digest, require_record, verify_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--preconsultation',required=True)
    args = ap.parse_args()
    need(Path.cwd().resolve() == ROOT.resolve(), 'Run from the intended source checkout')
    record,rh = require_record('F004')
    _,uh = require_record('U001')
    bound,code = inventory(args.arm),sources()
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True), 'Source checkout is dirty')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    pre = json.loads(Path(args.preconsultation).read_text())
    validate_pre(pre,args.arm,bound,commit,code)
    out = BASE/'runs'/f'{args.arm}_s0'
    fresh_paths(out)
    common = supervision()
    _,_,owned = common.inherited()
    report = subprocess.run(['amd-smi','process','--json'],capture_output=True,text=True,check=True,timeout=10)
    gpu_idle(json.loads(report.stdout))
    ledger = ROOT/'outputs/gated/gpu_intervals.json'
    operation = out/'operations'
    # Ledger admission is serialized and refuses every existing live interval.
    interval = admit(ledger,out,args.arm)
    child = process = created = None
    actions,errors = [],[]
    stalled = False
    began = time.monotonic()
    supervisor = psutil.Process()
    science = out/'science'
    admission = dict(research_id='F004',record_sha256=rh,budget_policy='U001',authorization_sha256=uh,
                     arm=args.arm,seed=0,settings=record['protocol']['training'],model=record['protocol']['model'],
                     source_commit=commit,source_sha256=code,inputs_sha256=bound,preconsultation=pre,
                     preconsultation_sha256=digest(args.preconsultation),supervisor_pid=supervisor.pid,
                     supervisor_created=supervisor.create_time(),cwd=str(ROOT),science_out=str(science),
                     ledger=str(ledger),interval=interval,stall_seconds=1800,gpu_hours_limit=None,
                     admitted_at=datetime.now(timezone.utc).isoformat())
    argv = [sys.executable,str(ROOT/'tools/research/f004_train.py'),'--admission',str(operation/'admission.json')]
    env = dict(os.environ)
    for key in ('PYTHONPATH','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_ENABLE_SDMA'):
        env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_TUNABLEOP_ENABLED='0')
    try:
        operation.mkdir(parents=True,exist_ok=False)
        atomic_json(operation/'admission.json',admission)
        (operation/'process_before.json').write_text(report.stdout)
        with (operation/'training.log').open('xb') as log:
            child = subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            process = psutil.Process(child.pid)
            created = process.create_time()
            atomic_json(operation/'child.json',dict(pid=child.pid,creation_time=created,argv=argv,cwd=str(ROOT)))
            stalled = common.wait_progress(child,process,created,argv,ROOT,
                [science/'progress.json',operation/'training.log'],actions,
                heartbeat=lambda quiet: atomic_json(operation/'monitor_latest.json',dict(at=time.time(),
                    pid=child.pid,creation_time=created,seconds_without_progress=quiet)))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                owned.stop_owned(child,process,created,argv,ROOT,actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            close_interval(ledger,interval,child)
        except BaseException as exc:
            errors.append(repr(exc))
    terminal = child is None or child.poll() is not None
    code_return = None if child is None else child.returncode
    unchanged = False
    artifacts = {}
    if terminal:
        try:
            verify_files(ROOT,bound)
            verify_files(ROOT,code)
            need(inventory(args.arm) == bound, 'Terminal evidence inventory changed')
            unchanged = True
            artifacts = {str(p.relative_to(science)):digest(p) for p in science.rglob('*') if p.is_file()}
        except BaseException as exc:
            errors.append(repr(exc))
    result_path = science/'result.json'
    raw = json.loads(result_path.read_text()) if result_path.exists() else {}
    okay = (code_return == 0 and terminal and not errors and not stalled and unchanged
            and raw.get('status') == 'awaiting_literature_review'
            and raw.get('final_step') == record['protocol']['training']['steps'])
    atomic_json(operation/'result.json',dict(research_id='F004',phase='operations',
                status='awaiting_literature_review' if okay else 'technical_blocked',
                run_identity=dict(arm=args.arm,seed=0),record_sha256=rh,returncode=code_return,
                child_terminal=terminal,stalled=stalled,actions=actions,errors=errors,inputs_unchanged=unchanged,
                interval_id=interval['id'],admission_sha256=digest(operation/'admission.json'),
                elapsed_seconds=time.monotonic()-began,child_artifact_sha256=artifacts,
                next_step='Inspect actual checkpoint/logs, consult literature and review training/operations separately. No automatic successor.'))
    print(json.dumps(dict(arm=args.arm,status='awaiting_literature_review' if okay else 'technical_blocked',
                         returncode=code_return,errors=errors)),flush=True)
    if not okay:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
