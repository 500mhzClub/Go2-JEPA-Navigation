"""Wait for preserved trainers, check final artifacts, run only A003, then stop for review.

This process never signals, resumes or edits a trainer. Its state file reports
waiting/running/technical_blocked/awaiting_literature_review explicitly.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from go2nav.research import (REPO, atomic_json, provenance, require_record,
                            require_training_slot, verify_files)


def verify_final(path, config):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('step') != config['total_steps'] or state.get('inference_only'):
        raise ValueError(f'{path}: expected an original completed checkpoint')
    for key, expected in (('kind', config['kind']), ('n', config['n']),
                          ('train', config['train']), ('config', config['model'])):
        if json.dumps(state.get(key), sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f'{path}: final checkpoint disagrees with original {key}')


def run(runs, data, out, pids, research_id='A003'):
    record, _ = require_record(research_id)
    if research_id != 'A003':
        raise ValueError('This queue is only the preregistered A003 final-checkpoint suite')
    runs, data, out = Path(runs).resolve(), Path(data).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = {name: runs/name/'config.json' for name in record['protocol']['models']}
    manifest = provenance(research_id, dict(stage='wait_for_final_checkpoints', runs=str(runs), data=str(data)),
                          configs.values())
    processes = [psutil.Process(pid) for pid in pids]
    if len(processes) != 6 or len({p.pid for p in processes}) != 6:
        raise ValueError('Capture all six distinct original trainer processes')
    for p in processes:
        if 'go2nav.training.balls' not in p.cmdline():
            raise ValueError(f'{p.pid} is not an original balls trainer')
    manifest['processes'] = [dict(pid=p.pid, created_at=p.create_time(), command=p.cmdline()) for p in processes]
    atomic_json(out/'manifest.json', manifest)
    state = dict(research_id=research_id, pid=os.getpid(), completed=[], status='waiting')

    def update(status, **values):
        state.update(status=status, updated_at=time.time(), **values)
        atomic_json(out/'state.json', state)

    try:
        deadline = time.monotonic()+24*3600
        while True:
            active = []
            for p in processes:
                try:
                    if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                        active.append(p.pid)
                except psutil.NoSuchProcess:
                    pass
            update('waiting', active_pids=active)
            if not active:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Original trainers still active after 24h; left untouched')
            time.sleep(30)
        require_training_slot()
        verify_files(REPO, manifest['source_sha256'])
        verify_files(REPO, manifest['inputs'])
        for name, path in configs.items():
            verify_final(runs/name/'model.pt', json.loads(path.read_text()))
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
        for name in configs:
            verify_files(REPO, manifest['source_sha256'])
            require_training_slot()
            update('running', current=name)
            command = [sys.executable, '-m', 'go2nav.eval.balls_inference', '--checkpoint', str(runs/name/'model.pt'),
                       '--data', str(data), '--out', str(out/name), '--research', research_id]
            with (out/f'{name}.log').open('x') as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(name)
        update('awaiting_literature_review', current=None,
               next_step='Revisit the literature and review each final numerical result before dependent experiments')
    except BaseException as exc:
        update('technical_blocked', error=repr(exc))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'data', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--pids', nargs=6, type=int, required=True)
    a = p.parse_args()
    run(a.runs, a.data, a.out, a.pids)


if __name__ == '__main__':
    main()
