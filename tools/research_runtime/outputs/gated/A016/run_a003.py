"""Run one unchanged final A003 case after all six reviewed completions. No auto-review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

validation = Path.cwd().resolve()
sys.path.insert(0, str(validation))
from go2nav.research import atomic_json, digest, require_record, require_review, require_training_slot, verify_files
from supervision import admit, close_intervals, remaining_seconds, stop_owned, wait_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    # This must refuse the live trainer before any output, ledger mutation or GPU tool.
    require_training_slot()
    record, record_hash = require_record('A016')
    base = validation/'outputs/gated/A016'
    assert require_review(base/'software/review.json')['status'] == 'passed'
    software = json.loads((base/'software/result.json').read_text())
    verify_files(base, software['source_sha256'])
    gated = validation.parent/'Go2-JEPA-Navigation-gated'
    pin = '161fe8b89592516690fcbb8916fd14224b87dfe6'
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=gated, text=True).strip() == pin
    assert subprocess.check_output(['git','status','--porcelain','--untracked-files=all'], cwd=gated, text=True) == ''
    a003 = json.loads((gated/'docs/research/records/A003.json').read_text())
    names = a003['protocol']['models']
    assert args.run in names
    plan = json.loads((validation/'outputs/gated/A014/continuation_plan.json').read_text())
    require_review(validation/'outputs/gated/A014/preparation/review.json')
    prepared = json.loads((validation/'outputs/gated/A014/preparation/result.json').read_text())
    assert prepared['plan_sha256'] == digest(validation/'outputs/gated/A014/continuation_plan.json')
    dependencies, checkpoints = {}, {}
    for command in plan['recovery_commands']:
        name = command['name']
        run = validation/'outputs/gated/A014/runs'/name
        audit = validation/'outputs/gated/A015/finals'/name
        completion_review = require_review(run/'review.json')
        assert completion_review['status'] == require_review(audit/'review.json')['status'] == 'passed'
        assert (validation/completion_review['final_audit_review']).resolve() == (audit/'review.json').resolve()
        assert completion_review['final_audit_review_sha256'] == digest(audit/'review.json')
        execution = json.loads((run/'execution.json').read_text())
        assert execution['status'] == 'exited_awaiting_literature_review' and execution['returncode'] == 0
        try:
            p = psutil.Process(execution['pid'])
            assert p.create_time() != execution['creation_time'] or p.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        terminal = json.loads((run/'result.json').read_text())
        assert terminal['execution_sha256'] == digest(run/'execution.json')
        result = json.loads((audit/'result.json').read_text())
        manifest = json.loads((audit/'manifest.json').read_text())
        assert result['status'] == 'passed' and result['step'] == 303000
        assert result['manifest_sha256'] == digest(audit/'manifest.json')
        assert manifest['execution_sha256'] == digest(run/'execution.json')
        assert manifest['continuation_result_sha256'] == digest(run/'result.json')
        original = Path(command['cwd'])
        assert (original/'outputs/data/balls').resolve() == (validation/'outputs/data/balls').resolve()
        verify_files(original, plan['original_source_sha256'])
        verify_files(original, command['data_sha256'])
        root = original/command['argv'][command['argv'].index('--out')+1]
        verify_files(root, manifest['snapshot_sha256'])
        checkpoints[name] = root/'model.pt'
        assert result['model_sha256'] == digest(checkpoints[name])
        dependencies[name] = dict(a014_review_sha256=digest(run/'review.json'),
                                  a015_review_sha256=digest(audit/'review.json'), model_sha256=result['model_sha256'])
    assert set(checkpoints) == set(names)
    previous = names[:names.index(args.run)]
    for name in previous:
        assert require_review(validation/'outputs/gated/A003_recovered'/name/'review.json')['status'] == 'passed'
        assert require_review(base/'runs'/name/'review.json')['status'] == 'passed'
    child_out = validation/'outputs/gated/A003_recovered'/args.run
    out = base/'runs'/args.run
    assert not child_out.exists() and not out.exists()
    gpu = subprocess.run(['amd-smi','process','--json'], capture_output=True, text=True, timeout=10, check=True)
    assert not any(isinstance(p['process_info'], dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'process_before.json').write_text(gpu.stdout)
    command = plan['recovery_commands'][0]
    argv = [command['argv'][0], '-m','go2nav.eval.balls_inference','--checkpoint',str(checkpoints[args.run]),
            '--data',str((validation/'outputs/data/balls').resolve()),'--out',str(child_out),'--research','A003']
    env = dict(os.environ)
    for key in command['env_unset']:
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    ledger = (validation/'outputs/gated/gpu_intervals.json').resolve()
    env['GO2NAV_GPU_LEDGER'] = str(ledger)
    launch_mono = time.monotonic()
    outer, remaining, work_seconds = admit(ledger, out)
    manifest = dict(research_id='A016', research_sha256=record_hash, name=args.run, source_commit=pin,
                    source_sha256=software['source_sha256'], software_review_sha256=digest(base/'software/review.json'),
                    dependencies=dependencies, argv=argv, cwd=str(gated), outer_interval_id=outer['id'],
                    admitted_at=outer['start'], remaining_at_admission=remaining, work_seconds=work_seconds,
                    cleanup_reserve_seconds=45, status='running')
    child = process = None
    actions, errors, closed = [], [], []
    timed_out = False
    identity = None
    try:
        atomic_json(out/'manifest.json', manifest)
        with (out/'evaluation.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=gated, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            identity = process.create_time()
            manifest.update(pid=child.pid, creation_time=identity)
            atomic_json(out/'manifest.json', manifest)
            timed_out = wait_owned(child, process, identity, argv, gated, launch_mono+work_seconds, actions,
                remaining=lambda: remaining_seconds(json.loads(ledger.read_text()), time.time()),
                heartbeat=lambda seconds: atomic_json(out/'monitor_latest.json',
                    dict(at=time.time(),pid=child.pid,creation_time=identity,remaining_work_seconds=seconds)))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or identity is None:
                    raise RuntimeError('Child identity unavailable; cannot safely signal')
                stop_owned(child, process, identity, argv, gated, actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            closed = close_intervals(ledger, outer['id'], child, child_out, gated)
        except BaseException as exc:
            errors.append(repr(exc))
    terminal = child is None or child.poll() is not None
    code = child.returncode if child is not None else None
    status = 'technical_blocked' if errors or code not in (0, None) else 'awaiting_literature_review'
    if timed_out and terminal and not errors:
        status = 'budget_incomplete'
    if child is None:
        status = 'technical_blocked'
    manifest.update(status=status, finished_at=time.time(), elapsed_monotonic_seconds=time.monotonic()-launch_mono)
    atomic_json(out/'manifest.json', manifest)
    raw = {n:digest(child_out/n) for n in ['result.json','manifest.json','calibrated.pt'] if (child_out/n).exists()}
    result = dict(research_id='A016', status=status, name=args.run, returncode=code, timed_out=timed_out,
                  child_terminal=terminal, actions=actions, errors=errors, closed_interval_ids=closed,
                  elapsed_seconds=manifest['elapsed_monotonic_seconds'], manifest_sha256=digest(out/'manifest.json'),
                  raw_a003_sha256=raw, next_step='Literature review of operational and actual A003 numerical result; no automatic successor.')
    atomic_json(out/'result.json', result)
    print(json.dumps(result), flush=True)
    if status != 'awaiting_literature_review':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
