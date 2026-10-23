"""Run a cohort of missions (an experiment's plan) in parallel, read each, and tabulate the results.

    python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml [--cohort NAME] [--workers 4] [--out outputs/cohorts]
    python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml --retry ASSIGNMENT [...]
    python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml --resume
    python -m go2nav.eval.cohort --summarise outputs/cohorts/rexit2_ref [...]
    python -m go2nav.eval.cohort --rebuild outputs/cohorts/rexit2_ref

An experiment config lists cohorts; each cohort is a set of arms on registered maze episodes with shared options
(recovery, dynamics, decoder fit). Every mission runs in its own process (`go2nav.eval.mission`), then the episode
reader (`go2nav.eval.episode`). Missions start only while the disk keeps the 12-GiB reserve (plus 2 GiB) and memory
allows; C3 missions are limited to `--c3-lanes` at a time and C5-C8 missions to `--lewm-lanes` (GPU memory; the
lanes are `go2nav.controllers.registry.GPU_LANES`). Each row reports the outcome, contacts, wall
clearance, hold rates and failure taxonomy, plus how often each recovery mechanism intervened and how often the
registered pose jumped beyond the robot's physical limits, so recovery cannot hide weak prediction.

Folder: <out>/<cohort>/{config.json, result.json, summary.md, logs/}, missions in <out>/<cohort>/runs/<assignment>.
`--retry` reruns listed missions of a finished cohort (e.g. ones stopped by an infrastructure fault); each previous
attempt is kept beside the new one as runs/<assignment>.attempt<N> (and its log), and the tables are rewritten.
`--resume` continues an interrupted experiment: finished cohorts are skipped; in an interrupted one (a folder without
result.json) the finished missions are kept (and evaluated if they were not), unfinished ones are set aside as attempts
and rerun with the missions not yet started.
`--rebuild` re-reads every mission of a cohort folder and rewrites result.json and summary.md.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

import yaml

from go2nav.controllers.registry import GPU_LANES, MEMORY_GIB

GIB = 1024**3
RESERVE_GIB, RESERVE_MARGIN_GIB = 12, 2
MEMORY_MARGIN_GIB = 8
POSE_JUMP_M, POSE_JUMP_RAD = .05, .15   # the Go2 moves at most 3 cm and 0.05 rad in 100 ms


def load_experiment(path):
    return yaml.safe_load(Path(path).read_text())


def jobs(cohort):
    """[(arm, set, maze, episode, assignment)] of a cohort config."""
    out = []
    for maze, episode in cohort['episodes']:
        for arm in cohort['arms']:
            out.append((arm, cohort['set'], int(maze), int(episode), f"{cohort['name']}_{arm}_{cohort['set']}_{maze:02d}_ep{episode}"))
    return out


def mission_command(cohort, arm, set_name, maze, episode, destination):
    command = [sys.executable, '-m', 'go2nav.eval.mission', '--arm', arm, '--set', set_name, '--maze', str(maze),
               '--episode', str(episode), '--recovery', cohort.get('recovery', 'off'), '--out', str(destination)]
    for key, flag in (('friction', '--friction'), ('strips', '--strips'), ('stall_stop_s', '--stall-stop-s'),
                      ('obstacles', '--obstacles'),
                      ('decoder_fit', '--decoder-fit'), ('degrade', '--degrade'), ('margin', '--margin'),
                      ('pessimistic', '--pessimistic'), ('unknown_level', '--unknown-level'), ('models', '--models'),
                      ('harness', '--harness'), ('coverage_fix', '--coverage-fix'),
                      ('map_decay', '--map-decay')):
        if cohort.get(key) is not None:
            command += [flag, str(cohort[key])]
    if cohort.get('demo_textures'):
        command.append('--demo-textures')
    return command


# -- rows -------------------------------------------------------------------------------------------------------------

def interventions(run):
    """How often each recovery mechanism changed a plan (from the planning log)."""
    path = Path(run)/'planning.json'
    rows = [r for r in json.loads(path.read_text()) if 'selection' in r] if path.exists() else []
    counts = Counter()
    for r in rows:
        s = r['selection']
        event = (s.get('clearance_turn') or {}).get('event')
        counts['latch_timeouts'] += event == 'DEV_LATCH_TIMEOUT_RELEASED'
        counts['latch_cooldown_suppressions'] += event == 'DEV_LATCH_SUPPRESSED_COOLDOWN'
        counts['reactive_escapes'] += bool(s.get('reactive_escape'))
        counts['terminal_spin_breaks'] += bool(s.get('terminal_spin_break'))
    return dict(decisions=len(rows), **{k: counts[k] for k in ('latch_timeouts', 'latch_cooldown_suppressions',
                                                              'reactive_escapes', 'terminal_spin_breaks')})


def pose_corrections(run):
    """Registered-pose changes between consecutive frames beyond the robot's limits, and floor-transport frames."""
    path = Path(run)/'poses.json'
    if not path.exists():
        return dict(pose_corrections=None, largest_pose_correction_m=None, floor_transport_frames=None)
    jumps, largest, transport, previous = 0, 0., 0, None
    for row in json.loads(path.read_text()):
        pose = row['registered_pose']
        transport += bool(pose['transported'])
        x, y = pose['position'][:2]
        R = pose['rotation']
        yaw = math.atan2(R[1][0], R[0][0])
        if previous is not None and row['frame'] == previous[0]+1:
            step = math.hypot(x-previous[1], y-previous[2])
            turn = abs(math.atan2(math.sin(yaw-previous[3]), math.cos(yaw-previous[3])))
            if step > POSE_JUMP_M or turn > POSE_JUMP_RAD:
                jumps += 1
                largest = max(largest, step)
        previous = (row['frame'], x, y, yaw)
    return dict(pose_corrections=jumps, largest_pose_correction_m=largest, floor_transport_frames=transport)


def mission_row(run, assignment, arm, code=None):
    row = dict(assignment=assignment, controller=arm, exit=code)
    evaluation = Path(run)/'episode_evaluation.json'
    if evaluation.exists():
        ev = json.loads(evaluation.read_text())
        s, stall = ev['safety'], ev['stall_by_phase']
        row.update(round_trip=ev['round_trip_success'], beacon=ev['beacon_success'], home=ev['home_success'],
                   contacts=ev['disallowed_contact_samples'], hard=s['hard']['confirmed_violation_samples'],
                   hard_unresolved=s['hard'].get('unresolved_sampled_samples'),
                   operating=s['operating']['confirmed_violation_samples'],
                   min_clearance_m=s['hard']['minimum_separation_lower_m'],
                   outbound_hold_rate=stall.get('OUTBOUND', {}).get('rate'), return_hold_rate=stall.get('RETURN', {}).get('rate'),
                   outbound_spl=ev['outbound']['spl'], return_spl=ev['return_leg']['spl'],
                   source_error=ev['source_error'], taxonomy=ev['failure_and_stall_taxonomy'], wall_s=ev['wall_s'])
    else:
        row.update(round_trip=None, unread=True)
    return row | interventions(run) | pose_corrections(run)


# -- running ----------------------------------------------------------------------------------------------------------

def set_aside(root, assignment):
    """Keep a previous attempt of a mission as runs/<assignment>.attempt<N> (and its log)."""
    n = 1
    while (root/'runs'/f'{assignment}.attempt{n}').exists():
        n += 1
    for path in (root/'runs'/assignment, root/'logs'/f'{assignment}.log'):
        if path.exists():
            path.rename(path.with_name(f'{assignment}.attempt{n}{path.suffix}'))


def run_cohort(cohort, out, *, workers=4, c3_lanes=2, lewm_lanes=6, retry=None, resume=False):
    root = Path(out)/cohort['name']
    plan = jobs(cohort)
    if resume and root.exists():
        if json.loads((root/'config.json').read_text())['plan'] != [list(j) for j in plan]:
            raise ValueError(f'{root} was run with another plan')
        rows, pending = {}, []
        for job in plan:
            run = root/'runs'/job[4]
            if (run/'result.json').exists():
                if not (run/'episode_evaluation.json').exists():
                    subprocess.run([sys.executable, '-m', 'go2nav.eval.episode', str(run)], check=True)
                rows[job[4]] = mission_row(run, job[4], job[0])
            else:
                set_aside(root, job[4])
                pending.append(job)
    elif retry is None:
        (root/'logs').mkdir(parents=True, exist_ok=False)
        (root/'runs').mkdir()
        (root/'config.json').write_text(json.dumps(dict(cohort=cohort, plan=plan), indent=1)+'\n')
        rows, pending = {}, list(plan)
    else:
        rows = {r['assignment']: r for r in json.loads((root/'result.json').read_text())['rows']}
        pending = [j for j in plan if j[4] in retry]
        for assignment in (j[4] for j in pending):
            set_aside(root, assignment)
    running, lock = {}, threading.Lock()
    caps = dict(c3=c3_lanes, lewm=lewm_lanes)

    def lane_free(arm):
        lane = GPU_LANES.get(arm)
        return lane is None or sum(1 for a in running.values() if GPU_LANES.get(a) == lane) < caps[lane]

    def attempt(arm, set_name, maze, episode, assignment):
        try:
            destination = root/'runs'/assignment
            with open(root/'logs'/f'{assignment}.log', 'x') as log:
                code = subprocess.run(mission_command(cohort, arm, set_name, maze, episode, destination), stdout=log,
                                      stderr=subprocess.STDOUT).returncode
                if (destination/'result.json').exists():
                    subprocess.run([sys.executable, '-m', 'go2nav.eval.episode', str(destination)], stdout=log,
                                   stderr=subprocess.STDOUT)
            row = mission_row(destination, assignment, arm, code)
        except Exception as exc:  # record and continue with the cohort
            row = dict(assignment=assignment, controller=arm, error=repr(exc)[:500])
        with lock:
            rows[assignment] = row
            running.pop(assignment, None)
        print(json.dumps(row), flush=True)

    while True:
        with lock:
            if not pending and not running:
                break
            candidate = next((j for j in pending if lane_free(j[0])), None)
            disk_ok = shutil.disk_usage(root).free > (RESERVE_GIB+RESERVE_MARGIN_GIB)*GIB
            memory_ok = candidate is not None and available_memory() >= (MEMORY_GIB[candidate[0]]+MEMORY_MARGIN_GIB)*GIB
            if candidate and len(running) < workers and disk_ok and memory_ok:
                pending.remove(candidate)
                running[candidate[4]] = candidate[0]
                threading.Thread(target=attempt, args=candidate).start()
            elif not disk_ok and not running:
                print(json.dumps(dict(stop='storage reserve', pending=len(pending))), flush=True)
                break
        time.sleep(5)
    ordered = [rows[j[4]] for j in plan if j[4] in rows]
    (root/'result.json').write_text(json.dumps(dict(cohort=cohort['name'], rows=ordered), indent=1)+'\n')
    (root/'summary.md').write_text(tables(ordered)+'\n')
    return ordered


def rebuild(root):
    """Re-read every mission of a cohort folder; rewrite result.json and summary.md (exit codes are not kept)."""
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    rows = [mission_row(root/'runs'/assignment, assignment, arm) for arm, _, _, _, assignment in config['plan']]
    (root/'result.json').write_text(json.dumps(dict(cohort=config['cohort']['name'], rows=rows), indent=1)+'\n')
    (root/'summary.md').write_text(tables(rows)+'\n')
    return rows


def available_memory():
    with open('/proc/meminfo') as stream:
        info = dict(line.split(':', 1) for line in stream)
    return int(info['MemAvailable'].split()[0])*1024


# -- tables -----------------------------------------------------------------------------------------------------------

def fmt(value, digits=3):
    return '-' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def tables(rows):
    """Per-mission rows, then per-controller totals (round trips, recovery interventions, contacts, clearance)."""
    lines = ['| Mission | Ctrl | Round trip | Latch timeouts | Reactive escapes | Spin breaks | Pose corrections | Contacts | Hard | Min clearance (m) | Error |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['assignment']} | {r['controller']} | {fmt(r.get('round_trip'))} | {r.get('latch_timeouts', 0)} "
                     f"| {r.get('reactive_escapes', 0)} | {r.get('terminal_spin_breaks', 0)} | {fmt(r.get('pose_corrections'))} "
                     f"| {fmt(r.get('contacts'))} | {fmt(r.get('hard'))} | {fmt(r.get('min_clearance_m'))} | {fmt(r.get('source_error') or r.get('error'))[:60]} |")
    by = defaultdict(list)
    for r in rows:
        by[r['controller']].append(r)
    lines += ['', '| Ctrl | Missions read | Round trips | Beacon | Home | Missions with any recovery | Contacts (samples) | Hard violations (samples) | Min clearance (m) | Median outbound SPL (all missions read) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ctrl, rs in sorted(by.items()):
        read = [r for r in rs if r.get('round_trip') is not None]
        helped = sum(1 for r in rs if r.get('latch_timeouts') or r.get('reactive_escapes') or r.get('terminal_spin_breaks'))
        clearances = [r['min_clearance_m'] for r in read if r.get('min_clearance_m') is not None]
        spl = [r['outbound_spl'] for r in read]
        lines.append(f"| {ctrl} | {len(read)} | {sum(bool(r['round_trip']) for r in read)} | {sum(bool(r['beacon']) for r in read)} "
                     f"| {sum(bool(r['home']) for r in read)} | {helped} | {sum(r.get('contacts') or 0 for r in read)} "
                     f"| {sum(r.get('hard') or 0 for r in read)} | {fmt(min(clearances) if clearances else None)} "
                     f"| {fmt(statistics.median(spl) if spl else None)} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', nargs='?', type=Path)
    p.add_argument('--cohort', action='append', help='run only these cohorts of the experiment')
    p.add_argument('--out', type=Path, default=Path('outputs/cohorts'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--c3-lanes', type=int, default=2)
    p.add_argument('--lewm-lanes', type=int, default=6)
    p.add_argument('--retry', nargs='+', metavar='ASSIGNMENT', help='rerun these missions of finished cohorts')
    p.add_argument('--resume', action='store_true', help='continue an interrupted experiment')
    p.add_argument('--summarise', nargs='+', type=Path, help='print the tables of finished cohort folders')
    p.add_argument('--rebuild', type=Path, help='re-read a cohort folder and rewrite its result and tables')
    a = p.parse_args()
    if a.rebuild:
        print(tables(rebuild(a.rebuild)))
        return
    if a.summarise:
        rows = [r for folder in a.summarise for r in json.loads((folder/'result.json').read_text())['rows']]
        print(tables(rows))
        return
    experiment = load_experiment(a.experiment)
    for cohort in experiment['cohorts']:
        if a.cohort and cohort['name'] not in a.cohort:
            continue
        if a.retry and not any(j[4] in a.retry for j in jobs(cohort)):
            continue
        if a.resume and (a.out/cohort['name']/'result.json').exists():
            print(f"{cohort['name']}: finished", flush=True)
            continue
        run_cohort(cohort, a.out, workers=a.workers, c3_lanes=a.c3_lanes, lewm_lanes=a.lewm_lanes, retry=a.retry,
                   resume=a.resume)


if __name__ == '__main__':
    main()
