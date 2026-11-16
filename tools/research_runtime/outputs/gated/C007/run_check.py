"""Run one unchanged C004 check within its existing resource allowances."""
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
from go2nav.research import atomic_json, digest, require_record, require_review, require_training_slot, reviewed_artifact, verify_files
from supervision import admit, arguments, case_paths, check_parent, close_intervals, consultation, remaining_seconds, stop_owned, wait_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--preconsultation', required=True)
    args = parser.parse_args()
    paths = case_paths(ROOT, args.run)
    require_training_slot()
    _, record_hash = require_record('C007')
    _, science_hash = require_record('C004')
    base = ROOT/'outputs/gated/C007'
    assert require_review(base/'software/review.json')['status'] == 'passed'
    software = json.loads((base/'software/result.json').read_text())
    assert software['research_sha256'] == record_hash
    assert software['c004_record_sha256'] == science_hash
    verify_files(base, software['source_sha256'])
    verify_files(ROOT, software['repository_source_sha256'])
    assert digest(ROOT/'outputs/gated/A016/supervision.py') == software['a016_supervision_sha256']
    for parent in ['C003/recovery', 'C004/verification']:
        assert require_review(ROOT/'outputs/gated'/parent/'review.json')['status'] == 'passed'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    science = reviewed_artifact(paths['parent'], paths['checkpoint'])
    check_parent(science, paths)
    operations = [ROOT/'outputs/gated/C005'/version/'runs'/args.run/'review.json' for version in ['', 'v2']]
    operations = [p for p in operations if p.exists()]
    assert len(operations) == 1, 'Exactly one actual parent operational review is required'
    op_review = require_review(operations[0])
    assert op_review['status'] == 'passed'
    op = json.loads((operations[0].parent/op_review['result']).read_text())
    assert op['name'] == args.run and op['child_terminal'] and op['returncode'] == 0
    assert op['child_artifact_sha256']['model.pt'] == digest(paths['checkpoint'])
    dependencies = {str(paths['parent']): digest(paths['parent']), str(operations[0]): digest(operations[0])}
    if paths['kind'] != 'slots':
        er = require_review(paths['encoder_review'])
        assert er['status'] == 'passed'
        evidence = json.loads((paths['encoder_review'].parent/er['result']).read_text())
        assert (evidence['research_id'], evidence['kind'], evidence['n'], evidence['seed']) == ('C004', 'slots', paths['n'], paths['seed'])
        assert evidence['checks_passed']
        dependencies[str(paths['encoder_review'])] = digest(paths['encoder_review'])
    pre = consultation(args.preconsultation, paths, ROOT, science_hash, record_hash)
    child_out, out = paths['science'], paths['operational']
    assert not child_out.exists() and not out.exists()
    gpu = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, timeout=10, check=True)
    assert not any(isinstance(p['process_info'], dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'process_before.json').write_text(gpu.stdout)
    argv = arguments(args.python, paths, ROOT)
    env = dict(os.environ)
    for key in ['PYTHONPATH', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA']:
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    ledger = (ROOT/'outputs/gated/gpu_intervals.json').resolve()
    env['GO2NAV_GPU_LEDGER'] = str(ledger)
    start = time.monotonic()
    outer, remaining, work = admit(ledger, out)
    manifest = dict(research_id='C007', research_sha256=record_hash, c004_record_sha256=science_hash,
                    name=args.run, argv=argv, cwd=str(ROOT), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=software['source_sha256'], software_review_sha256=digest(base/'software/review.json'),
                    dependencies=dependencies, preconsultation=pre, preconsultation_sha256=digest(args.preconsultation),
                    outer_interval_id=outer['id'], admitted_at=outer['start'], remaining_at_admission=remaining,
                    work_seconds=work, per_call_seconds=1800, cleanup_reserve_seconds=45, status='running')
    child = process = identity = None
    actions, errors, closed = [], [], []
    timed_out = False
    try:
        atomic_json(out/'manifest.json', manifest)
        with (out/'validation.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            identity = process.create_time()
            manifest.update(pid=child.pid, creation_time=identity)
            atomic_json(out/'manifest.json', manifest)
            timed_out = wait_owned(child, process, identity, argv, ROOT, start+work, actions,
                remaining=lambda: remaining_seconds(json.loads(ledger.read_text()), time.time()),
                heartbeat=lambda seconds: atomic_json(out/'monitor_latest.json',
                    dict(at=time.time(), pid=child.pid, creation_time=identity, remaining_work_seconds=seconds)))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if process is None or identity is None:
                    raise RuntimeError('Owned child identity unavailable')
                stop_owned(child, process, identity, argv, ROOT, actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            closed = close_intervals(ledger, outer['id'], child, child_out, ROOT)
        except BaseException as exc:
            errors.append(repr(exc))
    terminal = child is None or child.poll() is not None
    code = None if child is None else child.returncode
    raw = child_out/'result.json'
    raw_status = json.loads(raw.read_text()).get('status') if raw.exists() else None
    status = 'awaiting_literature_review'
    if errors or code != 0 or raw_status not in ['awaiting_literature_review', 'budget_incomplete']:
        status = 'technical_blocked'
    if terminal and not errors and (timed_out or (code == 0 and raw_status == 'budget_incomplete')):
        status = 'budget_incomplete'
    manifest.update(status=status, finished_at=time.time(), elapsed_seconds=time.monotonic()-start)
    atomic_json(out/'manifest.json', manifest)
    result = dict(research_id='C007', name=args.run, status=status, returncode=code, child_terminal=terminal,
                  timed_out=timed_out, actions=actions, errors=errors, raw_status=raw_status,
                  closed_interval_ids=closed, elapsed_seconds=manifest['elapsed_seconds'],
                  manifest_sha256=digest(out/'manifest.json'),
                  child_artifact_sha256={f: digest(child_out/f) for f in ['result.json', 'manifest.json'] if (child_out/f).exists()},
                  next_step='Inspect and literature-review actual C004 and C007 results separately; no automatic successor.')
    atomic_json(out/'result.json', result)
    print(json.dumps(result), flush=True)
    if status != 'awaiting_literature_review':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
