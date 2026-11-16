"""B003 one-at-a-time bounded CPU execution; never auto-review or advance."""
import argparse
import fcntl
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
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); name = p.parse_args().run
    base = Path('outputs/gated/B003'); record, rh = require_record('B003'); order = record['protocol']['order']; assert name in order
    with (base/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert require_review(base/'software/review.json')['status'] == 'passed'
        sw = json.loads((base/'software/result.json').read_text()); assert sw['research_sha256'] == rh
        verify_files(base, sw['source_sha256']); verify_files(Path.cwd(), sw['repository_source_sha256'])
        assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        elapsed_previous = 0
        for prev in order[:order.index(name)]:
            assert require_review(Path('outputs/gated/planning')/prev/'review.json')['status'] in ['passed', 'measured_negative']
            assert require_review(base/'operations'/prev/'review.json')['status'] == 'passed'
            elapsed_previous += json.loads((base/'operations'/prev/'result.json').read_text())['elapsed_seconds']
        assert elapsed_previous+7200 <= 43200
        pre = base/f'pre_{name}.json'; consultation = json.loads(pre.read_text())
        assert consultation['name'] == name and consultation['research_sha256'] == rh and consultation['sources'] and consultation['consequences']
        path = Path('outputs/gated/A016/supervision.py')
        assert require_review(path.parent/'software/review.json')['status'] == 'passed'
        assert digest(path) == sw['a016_supervision_sha256'] == '7cfa40983615bcf263b847c900822c48bc420f2c1d975f4d80ce8034086a7939'
        spec = importlib.util.spec_from_file_location('b003_owned_child', path); owned = importlib.util.module_from_spec(spec); spec.loader.exec_module(owned)
        science = Path('outputs/gated/planning')/name; assert not science.exists()
        out = base/'operations'/name; out.mkdir(parents=True, exist_ok=False)
        plan = json.loads(Path('outputs/gated/A014/continuation_plan.json').read_text())
        argv = [plan['recovery_commands'][0]['argv'][0], str(base/'planning.py'), '--run', name]
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        env.update(HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        ledger = Path('outputs/gated/gpu_intervals.json'); before = digest(ledger)
        began = time.monotonic(); actions, errors = [], []; peak = 0; reason = None
        with (out/'planning.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid); identity = process.create_time()
            atomic_json(out/'execution.json', dict(research_id='B003', name=name, pid=child.pid, creation_time=identity, argv=argv,
                cwd=str(Path.cwd()), started_at=time.time(), research_sha256=rh, software_review_sha256=digest(base/'software/review.json'),
                preconsultation_sha256=digest(pre), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
            try:
                while child.poll() is None:
                    try: peak = max(peak, process.memory_info().rss)
                    except psutil.NoSuchProcess: pass
                    elapsed = time.monotonic()-began
                    if elapsed >= 7185: reason = 'CPU7200second_cap_including15second_cleanup'
                    if peak > 16*2**30: reason = 'CPU16GiB_RSS_cap'
                    atomic_json(out/'monitor_latest.json', dict(at=time.time(), pid=child.pid, creation_time=identity,
                        elapsed_seconds=elapsed, peak_rss_gb=peak/2**30, stop_reason=reason))
                    if reason: break
                    try: child.wait(timeout=min(2, max(.01, 7185-elapsed)))
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
        result = dict(research_id='B003', name=name, status=status, returncode=child.returncode, child_terminal=terminal,
            elapsed_seconds=time.monotonic()-began, peak_rss_gb=peak/2**30, stop_reason=reason, signals=actions, errors=errors,
            execution_sha256=digest(out/'execution.json'), gpu_ledger_unchanged=digest(ledger)==before,
            no_B003_gpu_intervals=not any(r['research_id']=='B003' for r in json.loads(ledger.read_text())),
            child_artifact_sha256={f:digest(science/f) for f in ['result.json','manifest.json','cases.jsonl'] if (science/f).exists()},
            note='Concurrent GPU work may legitimately change its ledger; this CPU case creates no interval. No successor or review is automatic.')
        atomic_json(out/'result.json', result); print(json.dumps(result), flush=True)
        if child.returncode != 0 or errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
