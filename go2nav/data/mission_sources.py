"""Arrange regenerated mission runs the way the decoder contexts read them (`go2nav.data.decoder_contexts`).

    python -m go2nav.data.mission_sources ROOT --cohorts COHORTS_ROOT [--cohorts COHORTS_ROOT ...] [--workers 6]

The T7/T8 decoder data includes the decisions of three groups of missions: the C3-v3 on-policy round and the C3-v2
fresh checks (configs/experiments/decoder_sources.yaml) and the stage-2 recordings (the cohort s2rec2 of
configs/experiments/r6_dynamics_stage2.yaml). For each run of those cohorts this links the run folder into
ROOT/go2_navigation_capability_v1_attempt_001/runs under the original programme's run name, replays it to regenerate
its camera frames (`go2nav.data.rerender`, every frame checked against the run's recorded pixel hashes) into the
group's replay folder, and writes the s2rec2 plan (dev_cohorts/s2rec2/config.json) with the original names. ROOT is
then a recordings root for `python -m go2nav.data.samples decoder`, together with the collections
(`go2nav.data.recordings`). Each cohort is taken from the first --cohorts root that has it. Runs already replayed (a
`done` marker) are kept.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from go2nav.data.decoder_contexts import CAPABILITY, REPLAYS

REPO = Path(__file__).resolve().parents[2]
GROUPS = dict(  # cohort: (replay group, original run name of (arm, maze, episode))
    c3v3_onpolicy=('onpolicy', lambda arm, maze, episode: f'c3v3_onpolicy_{arm}_m{maze:02d}_ep{episode}_attempt001'),
    c3v2_check=('fresh', lambda arm, maze, episode: f'c3v2_check_{arm}_chk{maze:02d}_ep{episode}_attempt001'),
    s2rec2=('stage2', None),
)


def stage2_name(arm, maze, episode):
    registry = json.loads((REPO/'data/mazes/stage2_sets_v1/registry.json').read_text())
    role = next(e['role'] for e in registry['entries'] if e['maze_id'] == maze)
    return f'dev_s2rec2_{arm}_{role}{maze:02d}_ep{episode}'


def replay(run, out):
    """The verified replay of one run, in its own process (Genesis is process-global)."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out.parent/f'{out.name}.log', 'w') as log:
        code = subprocess.run([sys.executable, '-m', 'go2nav.data.rerender', str(run), str(out)], cwd=REPO, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f'replay of {run} failed (exit {code}); see {out.parent/f"{out.name}.log"}')
    (out/'done').touch()
    return out.name


def arrange(root, cohorts_roots, workers=6):
    root = Path(root)
    runs = root/CAPABILITY/'runs'
    runs.mkdir(parents=True, exist_ok=True)
    stage2_plan, replays = [], []
    for cohort, (group, name_of) in GROUPS.items():
        cohorts_root = next(Path(r) for r in cohorts_roots if (Path(r)/cohort/'config.json').exists())
        config = json.loads((cohorts_root/cohort/'config.json').read_text())['cohort']
        for maze, episode in config['episodes']:
            for arm in config['arms']:
                assignment = f"{cohort}_{arm}_{config['set']}_{maze:02d}_ep{episode}"
                run = cohorts_root/cohort/'runs'/assignment
                name = (name_of or stage2_name)(arm, maze, episode)
                if cohort == 's2rec2':
                    stage2_plan.append([arm, config['set'], maze, episode, name])
                link = runs/name
                if not link.exists():
                    link.symlink_to(run.resolve())
                out = root/REPLAYS[group]/name
                if not (out/'done').exists():
                    replays.append((run, out))
    with ThreadPoolExecutor(workers) as pool:
        for name in pool.map(lambda job: replay(*job), replays):
            print(f'replayed {name}', flush=True)
    plan = root/CAPABILITY/'dev_cohorts/s2rec2/config.json'
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(dict(plan=stage2_plan), indent=1)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--cohorts', type=Path, action='append', required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    arrange(a.root, a.cohorts, a.workers)


if __name__ == '__main__':
    main()
