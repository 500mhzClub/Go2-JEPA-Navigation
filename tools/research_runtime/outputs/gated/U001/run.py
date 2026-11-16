"""Run a full original planning arm; time accounting does not impose a budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from common import BASE, HELPERS, admit, arguments, inherited, inputs, route, wait_progress
from go2nav.research import atomic_json, digest, require_record, require_review, require_training_slot, verify_files

def main():
    parser = argparse.ArgumentParser()
    for key in ['run', 'python', 'preconsultation']: parser.add_argument('--'+key, required=True)
    a = parser.parse_args(); p = route(a.run); require_training_slot()
    _, uh = require_record('U001'); _, rh = require_record('C014'); _, sh = require_record('B001')
    sw = json.loads((BASE/'software/result.json').read_text())
    assert require_review(BASE/'software/review.json')['status'] == 'passed' and sw['research_sha256'] == uh
    verify_files(BASE, sw['source_sha256']); verify_files(Path.cwd(), sw['repository_source_sha256'])
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    bound = {str(f): digest(f) for f in inputs(p)}; pre = json.loads(Path(a.preconsultation).read_text())
    assert pre['name'] == a.run and pre['research_sha256'] == uh and pre['inputs_sha256'] == bound
    assert pre['software_review_sha256'] == digest(BASE/'software/review.json')
    assert pre.get('sources') and pre.get('consulted_at') and pre.get('consequences')
    out, science = p['operations'], p['science']; assert not out.exists() and not science.exists()
    gpu = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, check=True, timeout=10)
    assert not any(isinstance(item['process_info'], dict) for g in json.loads(gpu.stdout) for item in g['process_list'])
    argv = arguments(a.python, p); env = dict(os.environ)
    for key in ['PYTHONPATH', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA']: env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    ledger = Path('outputs/gated/gpu_intervals.json').resolve(); env['GO2NAV_GPU_LEDGER'] = str(ledger)
    out.mkdir(parents=True); (out/'process_before.json').write_text(gpu.stdout)
    began = time.monotonic(); outer = admit(ledger, out, 'B001'); _, inherited_supervision, owned = inherited()
    manifest = dict(research_id='C009', budget_policy='U001', authorization_sha256=uh, integration_research_id='C014', integration_research_sha256=rh,
        scientific_research_id='B001', scientific_research_sha256=sh, stage='planning', name=a.run, argv=argv, cwd=str(Path.cwd()),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), software_review_sha256=digest(BASE/'software/review.json'),
        source_sha256=sw['source_sha256'], preconsultation=pre, preconsultation_sha256=digest(a.preconsultation),
        outer_interval_id=outer['id'], admitted_at=outer['start'], work_seconds=None, stall_seconds=1800, status='running')
    child = process = created = None; actions = []; errors = []; closed = []; stalled = False
    try:
        atomic_json(out/'manifest.json', manifest)
        with (out/'evaluation.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid); created = process.create_time(); manifest.update(pid=child.pid, creation_time=created)
            atomic_json(out/'manifest.json', manifest)
            stalled = wait_progress(child, process, created, argv, Path.cwd(), [science/'cases.jsonl', out/'evaluation.log'], actions,
                heartbeat=lambda quiet: atomic_json(out/'monitor_latest.json', dict(at=time.time(), pid=child.pid, creation_time=created, seconds_without_output=quiet, budget_policy='U001')))
    except BaseException as exc: errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try: owned.stop_owned(child, process, created, argv, Path.cwd(), actions)
            except BaseException as exc: errors.append(repr(exc))
        try: closed = inherited_supervision.close_intervals(ledger, outer['id'], child, science, Path.cwd(), 'B001')
        except BaseException as exc: errors.append(repr(exc))
    terminal = child is None or child.poll() is not None; code = None if child is None else child.returncode
    artifacts = {}; raw_status = None; unchanged = False
    if terminal:
        try:
            if (science/'result.json').exists(): raw_status = json.loads((science/'result.json').read_text())['status']
            artifacts = {str(f.relative_to(science)): digest(f) for f in science.rglob('*') if f.is_file()}
            verify_files(Path.cwd(), bound); verify_files(BASE, sw['source_sha256']); unchanged = True
        except BaseException as exc: errors.append(repr(exc))
    status = 'awaiting_literature_review' if code == 0 and not errors and not stalled and raw_status == 'awaiting_literature_review' else 'technical_blocked'
    manifest.update(status=status, finished_at=time.time(), elapsed_seconds=time.monotonic()-began); atomic_json(out/'manifest.json', manifest)
    result = dict(research_id='C009', budget_policy='U001', scientific_research_id='B001', stage='planning', name=a.run, status=status,
        returncode=code, child_terminal=terminal, timed_out=stalled, timeout_type='no_output_progress' if stalled else None, actions=actions, errors=errors,
        closed_interval_ids=closed, raw_status=raw_status, inputs_unchanged=unchanged, child_artifact_sha256=artifacts,
        elapsed_seconds=manifest['elapsed_seconds'], manifest_sha256=digest(out/'manifest.json'), next_step='Independently inspect and literature-review the complete result before the paired successor.')
    atomic_json(out/'result.json', result); print(json.dumps(result), flush=True)
    if status != 'awaiting_literature_review': raise SystemExit(1)

if __name__ == '__main__': main()
