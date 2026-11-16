"""Run exactly one approved original continuation; stop at its result review gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

base = Path('outputs/gated/A010')
plan = json.loads((base/'continuation_plan.json').read_text())
require_record('A010')
assert plan['cleanup_authorization'] == 'user_approved'
assert require_review(base/'cleanup/review.json')['status'] == 'passed'
index = int(sys.argv[1])
command = plan['recovery_commands'][index]
assert require_review(command['preparation_review'])['status'] == 'passed'
if index:
    previous = plan['recovery_commands'][index-1]['name']
    assert require_review(base/'runs'/previous/'review.json')['status'] == 'passed'
for process in psutil.process_iter(['pid', 'cmdline']):
    assert not any(arg.startswith('go2nav.training.') for arg in (process.info['cmdline'] or [])), process.info
original = Path(command['cwd'])
manifest = json.loads((base/'preparation/manifest.json').read_text())
verify_files(original, manifest['original_source_sha256'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=original, text=True).strip() == '7dfc83722da0a9fb805ccc879fe7e5459709a2fd'
verify_files(original, command['data_sha256'])
training_out = original/command['argv'][command['argv'].index('--out')+1]
assert digest(training_out/'checkpoint.pt') == command['checkpoint_sha256']
prepared = json.loads((Path(command['preparation_review']).parent/'result.json').read_text())['runs'][command['name']]
start_step = prepared['step']
env = dict(os.environ)
for key in command['env_unset']:
    env.pop(key, None)
env.update(command['env_set'])
out = base/'runs'/command['name']
out.mkdir(parents=True, exist_ok=False)
started = time.time()
with (training_out/'recovery_train.log').open('xb') as log:
    child = subprocess.Popen(command['argv'], cwd=original, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process = psutil.Process(child.pid)
    execution = dict(research_id='A010', name=command['name'], pid=child.pid,
                     creation_time=process.create_time(), supervisor_pid=os.getpid(),
                     started_at=started, start_step=start_step, total_steps=303000,
                     status='running', command=command, checkpoint_sha256=command['checkpoint_sha256'],
                     original_source_commit='7dfc83722da0a9fb805ccc879fe7e5459709a2fd',
                     launcher_sha256=digest(__file__), cleanup_review_sha256=digest(base/'cleanup/review.json'),
                     accounting='Continuation of original logical experiment; excluded from additional48-hour ledger')
    atomic_json(out/'execution.json', execution)
    print(json.dumps(dict(name=command['name'], pid=child.pid, creation_time=process.create_time(),
                          start_step=start_step)), flush=True)
    returncode = child.wait()
execution.update(status='exited_awaiting_literature_review', returncode=returncode,
                 finished_at=time.time(), elapsed_seconds=time.time()-started)
atomic_json(out/'execution.json', execution)
atomic_json(out/'result.json', dict(research_id='A010', status='awaiting_literature_review',
            name=command['name'], returncode=returncode, elapsed_seconds=execution['elapsed_seconds'],
            execution_sha256=digest(out/'execution.json'), final_artifact_verification='pending',
            next_step='Verify final step/config/finite state and literature-review before any successor'))
print(json.dumps(dict(name=command['name'], returncode=returncode, review_required=True)), flush=True)
