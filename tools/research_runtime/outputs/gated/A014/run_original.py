"""A014: one original continuation; external progress watchdog; no auto-review."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from a014_monitor import observed_step

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

base = Path('outputs/gated/A014')
_, rh = require_record('A014')
plan = json.loads((base/'continuation_plan.json').read_text())
prep = require_review(base/'preparation/review.json')
software = require_review(base/'software/review.json')
assert prep['status'] == software['status'] == 'passed'
prep_result = json.loads((base/'preparation/result.json').read_text())
assert prep_result['plan_sha256'] == digest(base/'continuation_plan.json')
software_result = json.loads((base/'software/result.json').read_text())
for name, expected in software_result['source_sha256'].items():
    assert digest(base/name) == expected, name
assert require_review(plan['postboot_review'])['status'] == 'passed'
index = int(sys.argv[1])
assert 0 <= index < len(plan['recovery_commands'])
command = plan['recovery_commands'][index]
if index:
    previous_name = plan['recovery_commands'][index-1]['name']
    assert require_review(base/'runs'/previous_name/'review.json')['status'] == 'passed'
for process in psutil.process_iter(['pid', 'cmdline']):
    assert not any(a.startswith('go2nav.training.') for a in (process.info['cmdline'] or [])), process.info
original = Path(command['cwd'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=original, text=True).strip() == plan['original_source_commit']
verify_files(original, plan['original_source_sha256'])
verify_files(original, command['data_sha256'])
training_out = original/command['argv'][command['argv'].index('--out')+1]
verify_files(training_out, command['input_sha256'])
config = json.loads((training_out/'config.json').read_text())
assert config['total_steps'] == 303000
env = dict(os.environ)
for key in command['env_unset']:
    env.pop(key, None)
env.update(command['env_set'])
out = base/'runs'/command['name']
out.mkdir(parents=True, exist_ok=False)
gpu = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, timeout=10, check=True)
(out/'process_before.json').write_text(gpu.stdout)
assert not any(isinstance(p['process_info'], dict) for g in json.loads(gpu.stdout) for p in g['process_list'])
started = time.time()
last_step = command['start_step']
last_progress_at = started
actions = []
stop_reason = None
metrics = training_out/'metrics.jsonl'
with (training_out/'recovery_train.log').open('xb') as log:
    child = subprocess.Popen(command['argv'], cwd=original, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process = psutil.Process(child.pid)
    identity = process.create_time()
    execution = dict(research_id='A014', research_sha256=rh, name=command['name'], pid=child.pid,
                     creation_time=identity, supervisor_pid=os.getpid(), started_at=started,
                     start_step=last_step, total_steps=303000, status='running', command=command,
                     boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                     original_source_commit=plan['original_source_commit'],
                     launcher_sha256=digest(__file__), preparation_review_sha256=digest(base/'preparation/review.json'),
                     software_review_sha256=digest(base/'software/review.json'),
                     postboot_review_sha256=digest(plan['postboot_review']),
                     accounting='Original logical continuation, excluded from additional48-hour ledger')
    atomic_json(out/'execution.json', execution)
    print(json.dumps(dict(name=command['name'], pid=child.pid, creation_time=identity, start_step=last_step)), flush=True)
    while child.poll() is None:
        error = None
        try:
            step = observed_step(metrics, started, last_step)
            if step > last_step:
                last_step, last_progress_at = step, time.time()
        except (OSError, ValueError, TypeError) as exc:
            error = repr(exc)
        now = time.time()
        atomic_json(out/'monitor_latest.json', dict(at=now, pid=child.pid, creation_time=identity,
                    last_step=last_step, last_progress_at=last_progress_at, elapsed_seconds=now-started,
                    no_progress_seconds=now-last_progress_at, observation_error=error))
        if now-started >= 24*3600:
            stop_reason = 'declared24hour_wall_limit'
        elif error is None and now-last_progress_at >= 600:
            stop_reason = 'declared600second_no_verified_step_progress_limit'
        if stop_reason:
            assert process.is_running() and process.create_time() == identity
            assert process.cwd() == str(original) and process.cmdline() == command['argv']
            atomic_json(out/'stop_observation.json', dict(reason=stop_reason, at=now, last_step=last_step,
                        checkpoint_sha256=digest(training_out/'checkpoint.pt'), metrics_sha256=digest(metrics)))
            process.terminate(); actions.append(dict(signal='SIGTERM', at=time.time()))
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                assert process.is_running() and process.create_time() == identity
                process.kill(); actions.append(dict(signal='SIGKILL', at=time.time()))
                child.wait(timeout=10)
            break
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    returncode = child.wait()
execution.update(status='exited_awaiting_literature_review', returncode=returncode,
                 finished_at=time.time(), elapsed_seconds=time.time()-started, stop_reason=stop_reason,
                 signals=actions, last_monitored_step=last_step)
atomic_json(out/'execution.json', execution)
atomic_json(out/'result.json', dict(research_id='A014', status='awaiting_literature_review',
            name=command['name'], returncode=returncode, elapsed_seconds=execution['elapsed_seconds'],
            execution_sha256=digest(out/'execution.json'), stop_reason=stop_reason,
            final_artifact_verification='pending', next_step='Verify final/full303000 state and review before successor'))
print(json.dumps(dict(name=command['name'], returncode=returncode, review_required=True)), flush=True)
