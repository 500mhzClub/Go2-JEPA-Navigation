"""Bound one unchanged C001 stage. Results require manual literature review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from go2nav.research import atomic_json, digest, require_record, require_review, require_training_slot, verify_files
from supervision import admit, close_intervals, consultation, remaining_seconds, stop_owned, wait_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['slots', 'extract', 'o0', 'o1'])
    parser.add_argument('--n', required=True, type=int, choices=[1, 3, 6])
    parser.add_argument('--seed', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('--python', required=True)
    parser.add_argument('--data', default='outputs/data/balls')
    parser.add_argument('--preconsultation', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--cache')
    parser.add_argument('--parent-review')
    args = parser.parse_args()
    require_training_slot()
    _, record_hash = require_record('C005')
    _, science_hash = require_record('C001')
    base = ROOT/'outputs/gated/C005/v2'
    assert require_review(base/'software/review.json')['status'] == 'passed'
    software = json.loads((base/'software/result.json').read_text())
    verify_files(base, software['source_sha256'])
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    name = f'{args.stage}_n{args.n}_s{args.seed}'
    pre = consultation(args.preconsultation, name, science_hash)
    verification = ROOT/'outputs/gated/C003/recovery/review.json'
    assert require_review(verification)['status'] == 'passed'
    verified = json.loads((verification.parent/'manifest.json').read_text())['source_sha256']
    bound_sources = [f'go2nav/models/{n}.py' for n in ['balls_slots', 'slots', 'lewm', 'norm']]
    bound_sources += ['go2nav/training/balls_slots.py', 'go2nav/research.py']
    verify_files(ROOT, {f: verified[f] for f in bound_sources})
    dependencies = {}
    for model in ['l0_n1_s0', 'l0_n3_s0', 'l0_n6_s0', 'l1_n1_s0', 'l1_n3_s0', 'l1_n6_s0']:
        for folder in ['A003_recovered', 'A016/runs']:
            p = ROOT/'outputs/gated'/folder/model/'review.json'
            assert require_review(p)['status'] == 'passed'
            dependencies[str(p)] = digest(p)
    child_out = ROOT/'outputs/gated/C001'/name
    out = base/'runs'/name
    assert not child_out.exists() and not out.exists()
    gpu = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, timeout=10, check=True)
    assert not any(isinstance(p['process_info'], dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'process_before.json').write_text(gpu.stdout)
    argv = [args.python, '-m', 'go2nav.training.balls_slots', args.stage, '--data', args.data,
            '--n', str(args.n), '--seed', str(args.seed), '--research', 'C001', '--out', str(child_out),
            '--verification-review', str(verification)]
    for option in ['checkpoint', 'cache', 'parent_review']:
        if getattr(args, option):
            argv += ['--'+option.replace('_', '-'), getattr(args, option)]
    env = dict(os.environ)
    for key in ['PYTHONPATH', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA']:
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    ledger = (ROOT/'outputs/gated/gpu_intervals.json').resolve()
    env['GO2NAV_GPU_LEDGER'] = str(ledger)
    launch_mono = time.monotonic()
    outer, remaining, work = admit(ledger, out)
    manifest = dict(research_id='C005', research_sha256=record_hash, name=name, argv=argv, cwd=str(ROOT),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=software['source_sha256'], software_review_sha256=digest(base/'software/review.json'),
                    verification_review_sha256=digest(verification), dependencies=dependencies,
                    preconsultation=pre, preconsultation_sha256=digest(args.preconsultation),
                    outer_interval_id=outer['id'], admitted_at=outer['start'], remaining_at_admission=remaining,
                    work_seconds=work, cleanup_reserve_seconds=45, status='running')
    child = process = identity = None
    actions, errors, closed = [], [], []
    timed_out = False
    try:
        atomic_json(out/'manifest.json', manifest)
        with (out/'training.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            identity = process.create_time()
            manifest.update(pid=child.pid, creation_time=identity)
            atomic_json(out/'manifest.json', manifest)
            timed_out = wait_owned(child, process, identity, argv, ROOT, launch_mono+work, actions,
                remaining=lambda: remaining_seconds(json.loads(ledger.read_text()), time.time()),
                heartbeat=lambda seconds: atomic_json(out/'monitor_latest.json',
                    dict(at=time.time(), pid=child.pid, creation_time=identity, remaining_work_seconds=seconds)))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or identity is None:
                    raise RuntimeError('Child identity unavailable; cannot safely signal')
                stop_owned(child, process, identity, argv, ROOT, actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            closed = close_intervals(ledger, outer['id'], child, child_out, ROOT)
        except BaseException as exc:
            errors.append(repr(exc))
    terminal = child is None or child.poll() is not None
    code = child.returncode if child is not None else None
    raw_path = child_out/'result.json'
    raw_status = json.loads(raw_path.read_text()).get('status') if raw_path.exists() else None
    status = 'awaiting_literature_review'
    if errors or code != 0 or raw_status not in ['awaiting_literature_review', 'budget_incomplete']:
        status = 'technical_blocked'
    if terminal and not errors and (timed_out or (code == 0 and raw_status == 'budget_incomplete')):
        status = 'budget_incomplete'
    manifest.update(status=status, finished_at=time.time(), elapsed_monotonic_seconds=time.monotonic()-launch_mono)
    atomic_json(out/'manifest.json', manifest)
    artifacts = {n:digest(child_out/n) for n in ['result.json', 'manifest.json', 'checkpoint.pt', 'model.pt', 'index.json']
                 if (child_out/n).exists()}
    result = dict(research_id='C005', name=name, status=status, returncode=code, child_terminal=terminal,
                  timed_out=timed_out, actions=actions, errors=errors, raw_status=raw_status,
                  closed_interval_ids=closed, elapsed_seconds=manifest['elapsed_monotonic_seconds'],
                  manifest_sha256=digest(out/'manifest.json'), child_artifact_sha256=artifacts,
                  next_step='Inspect actual child artifacts, consult primary sources, review C001 and C005 separately; no automatic successor.')
    atomic_json(out/'result.json', result)
    print(json.dumps(result), flush=True)
    if status != 'awaiting_literature_review':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
