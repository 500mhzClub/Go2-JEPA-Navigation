"""CPU-only supervision/accounting for an unchanged A003 child."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
import uuid

import psutil
from go2nav.research import atomic_json


def union_seconds(rows, now):
    intervals = sorted((r['start'], now if r['end'] is None else r['end']) for r in rows)
    total, end = 0., None
    for start, stop in intervals:
        if stop < start:
            raise ValueError('Ledger clock reversal')
        total += max(0., stop-max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def remaining_seconds(rows, now):
    return dict(global_seconds=48*3600-union_seconds(rows, now),
                a_seconds=4*3600-union_seconds([r for r in rows if r['research_id'].startswith('A')], now))


def admit(ledger, out, reserve=45):
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        if any(r['end'] is None for r in rows):
            raise ValueError('Another GPU interval remains open')
        now = time.time()
        remaining = remaining_seconds(rows, now)
        work_seconds = min(remaining.values())-reserve
        if work_seconds <= 0:
            raise ValueError('Insufficient remaining A/global budget including cleanup reserve')
        row = dict(id=uuid.uuid4().hex, research_id='A016', out=str(out), start=now, end=None)
        rows.append(row)
        atomic_json(ledger, rows)
    return row, remaining, work_seconds


def identity_matches(process, identity, argv, cwd):
    try:
        return (process.is_running() and process.create_time() == identity
                and process.cmdline() == argv and Path(process.cwd()).resolve() == Path(cwd).resolve())
    except psutil.NoSuchProcess:
        return False


def stop_owned(child, process, identity, argv, cwd, actions, term_seconds=30, kill_seconds=10):
    for name, method, timeout in [('SIGTERM', process.terminate, term_seconds),
                                  ('SIGKILL', process.kill, kill_seconds)]:
        if child.poll() is not None:
            return
        if not identity_matches(process, identity, argv, cwd):
            if child.poll() is not None:
                return
            raise RuntimeError('Refusing signal: owned child identity mismatch')
        try:
            method()
        except psutil.NoSuchProcess:
            if child.poll() is not None:
                return
            raise
        actions.append(dict(signal=name, at=time.time()))
        try:
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError('Owned child termination unconfirmed; leave intervals open')


def wait_owned(child, process, identity, argv, cwd, deadline, actions,
               remaining=None, heartbeat=None, reserve=45, poll_seconds=30,
               term_seconds=30, kill_seconds=10):
    while child.poll() is None:
        available = deadline-time.monotonic()
        if remaining is not None:
            available = min(available, min(remaining().values())-reserve)
        if heartbeat is not None:
            heartbeat(available)
        if available <= 0:
            stop_owned(child, process, identity, argv, cwd, actions, term_seconds, kill_seconds)
            return True
        try:
            child.wait(timeout=min(poll_seconds, available))
        except subprocess.TimeoutExpired:
            pass
    return False


def close_intervals(ledger, outer_id, child, child_out, child_cwd):
    if child is not None and child.poll() is None:
        raise RuntimeError('Refusing ledger closure: child remains live')
    child_out = Path(child_out).resolve()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(ledger.read_text())
        outer = [r for r in rows if r['id'] == outer_id]
        if len(outer) != 1 or outer[0]['research_id'] != 'A016' or outer[0]['end'] is not None:
            raise ValueError('Outer interval identity/state mismatch')
        outer = outer[0]
        inner = [r for r in rows if r['research_id'] == 'A003'
                 and (Path(child_cwd)/r['out']).resolve() == child_out]
        if len(inner) > 1 or any(r['start'] < outer['start'] for r in inner):
            raise ValueError('Ambiguous or historical child interval; refuse closure')
        manifest_path = child_out/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if len(inner) != 1 or manifest['run_id'] != inner[0]['id']:
                raise ValueError('Child manifest/interval mismatch')
        now = time.time()
        changed = []
        for row in [outer]+inner:
            if row['end'] is None:
                if now < row['start']:
                    raise ValueError('Ledger clock reversal')
                row.update(end=now, closure='A016: owned child terminal confirmed')
                changed.append(row['id'])
        atomic_json(ledger, rows)
    return changed
