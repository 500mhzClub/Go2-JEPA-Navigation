"""One C014 numerical case under U001; no automatic review or successor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
sys.path.insert(0, str(Path.cwd()))
from common import ORIGINAL as BASE, admit, close, inherited, inputs, need, reviewed, qualified, integration_hashes, extra_consultation, arguments, wait_progress, progress_paths
from go2nav.research import atomic_json, digest, require_record, require_training_slot, verify_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--preconsultation', required=True)
    args = p.parse_args()
    integration = qualified()
    require_training_slot()
    _, rh = require_record('C014')
    software, _ = reviewed(BASE/'software_numerics', 'C014')
    need(software['research_sha256'] == rh, 'Numerical software record differs')
    verify_files(BASE, software['helper_sha256'])
    verify_files(Path.cwd(), software['repository_source_sha256'])
    identity, bound = inputs(args.run)
    pre = json.loads(Path(args.preconsultation).read_text())
    need(pre.get('name') == args.run and pre.get('research_sha256') == rh and
         pre.get('inputs_sha256') == bound and pre.get('software_review_sha256') == digest(BASE/'software_numerics/review.json'),
         'Wrong or stale consultation')
    extra_consultation(pre, integration)
    need(pre.get('consulted_at') and pre.get('consequences') and pre.get('sources'), 'Missing consultation')
    need(all(all(s.get(k) for k in ('url', 'version', 'sections', 'consequence')) for s in pre['sources']), 'Incomplete sources')
    need(not subprocess.check_output(['git', 'status', '--porcelain'], text=True), 'Dirty numerical runtime')
    out, science = BASE/'operations'/args.run, BASE/args.run
    need(not out.exists() and not science.exists(), 'Output already exists')
    gpu = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, check=True, timeout=10)
    need(not any(isinstance(p['process_info'], dict) for g in json.loads(gpu.stdout) for p in g['process_list']), 'GPU occupied')
    out.mkdir(parents=True, exist_ok=False)
    (out/'process_before.json').write_text(gpu.stdout)
    ledger = Path('outputs/gated/gpu_intervals.json').resolve()
    row, remaining, work = admit(ledger, out)
    argv = arguments(args.python, args.run)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    manifest = dict(research_id='C014', research_sha256=rh, integration=integration, integration_helper_sha256=integration_hashes(), budget_policy='U001', stall_seconds=1800, cleanup_reserve_seconds=45, **identity, argv=argv, cwd=str(Path.cwd()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        helper_sha256=software['helper_sha256'], software_review_sha256=digest(BASE/'software_numerics/review.json'),
        inputs=bound, preconsultation=pre, preconsultation_sha256=digest(args.preconsultation),
        outer_interval_id=row['id'], remaining=remaining, work_seconds=work)
    child = proc = creation = None
    actions, errors, timed = [], [], False
    started = time.monotonic()
    base = inherited()
    try:
        atomic_json(out/'manifest.json', manifest)
        with (out/'validation.log').open('xb') as log:
            child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            proc = psutil.Process(child.pid)
            creation = proc.create_time()
            manifest.update(pid=child.pid, creation_time=creation)
            atomic_json(out/'manifest.json', manifest)
            timed = wait_progress(child, proc, creation, argv, Path.cwd(), progress_paths(science, out), actions,
                heartbeat=lambda quiet: atomic_json(out/'monitor_latest.json', dict(at=time.time(), pid=child.pid, no_output_seconds=quiet, budget_policy='U001')))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                base.stop_owned(child, proc, creation, argv, Path.cwd(), actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            close(ledger, row['id'], child)
        except BaseException as exc:
            errors.append(repr(exc))
    raw = json.loads((science/'result.json').read_text()) if (science/'result.json').exists() else {}
    code = None if child is None else child.returncode
    status = 'awaiting_literature_review' if code == 0 and not errors and not timed and raw.get('complete') else 'technical_blocked'
    result = dict(research_id='C014', integration_research_id='U011', budget_policy='U001', timeout_type='no_output_progress' if timed else None, name=args.run, status=status, returncode=code,
        child_terminal=child is None or child.poll() is not None, timed_out=timed, actions=actions, errors=errors,
        elapsed_seconds=time.monotonic()-started, manifest_sha256=digest(out/'manifest.json'),
        child_artifact_sha256={str(p.relative_to(science)): digest(p) for p in sorted(science.glob('*')) if p.is_file()},
        limitation='Operation only. Complete numerical failures remain failures; no automatic review or successor.')
    atomic_json(out/'result.json', result)
    print(json.dumps(result), flush=True)
    if status != 'awaiting_literature_review':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
