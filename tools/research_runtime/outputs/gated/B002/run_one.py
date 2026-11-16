"""Run the B002 CPU preflight with a bounded, identity-checked child."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files


def main():
    base = Path('outputs/gated/B002'); _, rh = require_record('B002')
    assert require_review(base/'software/review.json')['status'] == 'passed'
    sw = json.loads((base/'software/result.json').read_text()); assert sw['research_sha256'] == rh
    verify_files(base, sw['source_sha256']); verify_files(Path.cwd(), sw['repository_source_sha256'])
    path = Path('outputs/gated/A016/supervision.py')
    assert require_review(path.parent/'software/review.json')['status'] == 'passed'
    assert digest(path) == sw['a016_supervision_sha256'] == '7cfa40983615bcf263b847c900822c48bc420f2c1d975f4d80ce8034086a7939'
    spec = importlib.util.spec_from_file_location('b002_owned_child', path)
    owned = importlib.util.module_from_spec(spec); spec.loader.exec_module(owned)
    assert not (base/'preflight_l0_n3_s0').exists()
    out = base/'operations'; out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path('outputs/gated/A014/continuation_plan.json').read_text())
    argv = [plan['recovery_commands'][0]['argv'][0], str(base/'diagnose.py')]
    env = dict(os.environ); env.pop('PYTHONPATH', None)
    env.update(HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    ledger = Path('outputs/gated/gpu_intervals.json'); before = digest(ledger)
    began = time.monotonic(); actions, errors = [], []; peak = 0; reason = None
    with (out/'preflight.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process = psutil.Process(child.pid); identity = process.create_time()
        atomic_json(out/'execution.json', dict(research_id='B002', pid=child.pid, creation_time=identity, argv=argv,
                    cwd=str(Path.cwd()), started_at=time.time(), research_sha256=rh, software_review_sha256=digest(base/'software/review.json')))
        try:
            while child.poll() is None:
                try: peak = max(peak, process.memory_info().rss)
                except psutil.NoSuchProcess: pass
                if time.monotonic()-began >= 165: reason = 'CPU180second_cap_including15second_cleanup'
                if peak > 16*2**30: reason = 'CPU16GiB_RSS_cap'
                if reason: break
                try: child.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            if child.poll() is None:
                try: owned.stop_owned(child, process, identity, argv, Path.cwd(), actions, 10, 5)
                except BaseException as exc: errors.append(repr(exc))
    terminal = child.poll() is not None
    status = 'awaiting_literature_review' if child.returncode == 0 and not errors else 'technical_blocked'
    if reason and terminal and not errors: status = 'budget_incomplete'
    result = dict(research_id='B002', status=status, returncode=child.returncode, child_terminal=terminal,
                  elapsed_seconds=time.monotonic()-began, peak_rss_gb=peak/2**30, stop_reason=reason, signals=actions, errors=errors,
                  execution_sha256=digest(out/'execution.json'), gpu_ledger_unchanged=digest(ledger)==before,
                  no_B002_gpu_intervals=not any(r['research_id']=='B002' for r in json.loads(ledger.read_text())))
    atomic_json(out/'result.json', result); print(json.dumps(result), flush=True)
    if child.returncode != 0 or errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
