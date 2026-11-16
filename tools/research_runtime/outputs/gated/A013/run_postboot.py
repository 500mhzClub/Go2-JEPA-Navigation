"""CPU-only A013 supervisor; closes budget interval only after child exit."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, device_hours, digest, require_record, require_review, verify_files

case = 'postboot_primitive'
assert len(sys.argv) == 1
record, rh = require_record('A013')
base = Path('outputs/gated/A013')
handoff = json.loads((base/'handoff.json').read_text())
if Path('/proc/sys/kernel/random/boot_id').read_text().strip() == handoff['boot_id']:
    raise SystemExit('Refusing GPU probe: boot ID unchanged; awaiting user-coordinated reboot.')
assert require_review(base/'preparation/review.json')['status'] == 'passed'
parent_review = Path('outputs/gated/A012/sdma_disabled/review.json')
assert require_review(parent_review)['status'] == 'technical_blocked'
for filename, expected in handoff['artifact_sha256'].items():
    assert digest(filename) == expected, filename
verify_files(handoff['original_checkout'], handoff['original_source_sha256'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=handoff['original_checkout'], text=True).strip() == handoff['original_source_commit']
assert digest('outputs/gated/A012/probe.py') == handoff['probe_sha256']
assert digest('outputs/gated/A010/continuation_plan.json') == handoff['original_plan_sha256']
for p in psutil.process_iter(['pid', 'cmdline']):
    args = p.info['cmdline'] or []
    assert not any(a.startswith('go2nav.training.') for a in args), p.info
out = base/case
out.mkdir(exist_ok=False)
command = json.loads(Path('outputs/gated/A010/continuation_plan.json').read_text())['recovery_commands'][0]
env = dict(os.environ)
for key in command['env_unset']:
    env.pop(key, None)
env.update(command['env_set'])
argv = [command['argv'][0], str(Path('outputs/gated/A012/probe.py').resolve())]
for name, cmd in [('process_before', ['amd-smi', 'process', '--json']),
                  ('metric_before', ['amd-smi', 'metric', '--json']),
                  ('kernel_before', ['journalctl', '-k', '--since', '2 minutes ago', '--no-pager'])]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    (out/(name+'.txt')).write_text(r.stdout+r.stderr)
if any(isinstance(v['process_info'], dict) for gpu in json.loads((out/'process_before.txt').read_text()) for v in gpu['process_list']):
    raise RuntimeError('Another compute process is active')
ledger = Path('outputs/gated/gpu_intervals.json')
run_id = uuid.uuid4().hex
def interval(end=None):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if end is None:
            assert not any(r['end'] is None for r in rows)
            assert device_hours(rows) < 48-.02
            rows.append(dict(id=run_id, research_id='A013', out=str(out), start=time.time(), end=None))
        else:
            next(r for r in rows if r['id'] == run_id)['end'] = end
        atomic_json(ledger, rows)

manifest = dict(research_id='A013', research_sha256=rh, case=case, argv=argv,
                cwd=command['cwd'], env_unset=command['env_unset'],
                env_set=command['env_set'],
                source_sha256={p.name:digest(p) for p in [Path('outputs/gated/A012/probe.py'), Path(__file__)]},
                parent_review_sha256=digest(parent_review), run_id=run_id)
atomic_json(out/'manifest.json', manifest)
started = time.time()
interval()
child = None
timed_out = False
signals = []
try:
    with (out/'probe.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=command['cwd'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest.update(pid=child.pid, creation_time=psutil.Process(child.pid).create_time(), started_at=started)
        atomic_json(out/'manifest.json', manifest)
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate(); signals.append('SIGTERM')
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); signals.append('SIGKILL'); child.wait(timeout=5)
finally:
    if child is None or child.poll() is not None:
        interval(time.time())
    else:
        atomic_json(out/'unresolved_child.json', dict(pid=child.pid, ledger_left_open=True))
elapsed = time.time()-started
events = [json.loads(line) for line in (out/'probe.log').read_text().splitlines() if line.startswith('{')]
result = dict(research_id='A013', status='awaiting_literature_review', case=case,
              manifest_sha256=digest(out/'manifest.json'), log_sha256=digest(out/'probe.log'),
              returncode=child.returncode, timed_out=timed_out, signals=signals,
              elapsed_seconds=elapsed, last_stage=events[-1]['stage'] if events else None,
              completed=child.returncode==0 and events[-1]['stage']=='passed', events=events)
atomic_json(out/'result.json', result)
for name, cmd in [('process_after', ['amd-smi', 'process', '--json']),
                  ('kernel_after', ['journalctl', '-k', '--since', '2 minutes ago', '--no-pager'])]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    (out/(name+'.txt')).write_text(r.stdout+r.stderr)
print(json.dumps(result, indent=2), flush=True)
