"""Generate a corpus's bulk-rollout dataset, resumably: rollouts, render plans and labels (CPU), then camera frames
(GPU, in the render environment).

    python -m go2nav.corpus.datagen rollout CORPUS OUT [--split S ...] [--family F ...] [--scenes LIST] [--chunk 40]
                                    [--workers 16] [--cores-per-rollout 2] [--post-workers 8] [--seed 7] [--n-envs 48]
                                    [--n-blocks 200]
    python -m go2nav.corpus.datagen render CORPUS OUT --render-python PYTHON [--workers 8] [--egl-device 1]
                                    [--resolution 224] [--textures] [--max-frames N] [--frames-list FILE]

Scenes are taken in name order within each (split, family) and cut into chunks of --chunk scenes. A chunk is one bulk
rollout process (`go2nav.corpus.bulk`) pinned to its own --cores-per-rollout cores (a rollout is orchestration-bound
at about two cores); its scene k gets seed --seed + k. As a chunk finishes, each of its scenes gets a render plan
(`go2nav.corpus.render_plan`) and labels (`go2nav.corpus.labels`) in a pool of --post-workers processes. The render
stage replays every planned scene with `go2nav.corpus.render` under PYTHON (the Genesis 0.3.14 render environment)
on EGL device --egl-device. The May corpus used the defaults, with --textures for the render. --scenes rolls out only
the listed scenes (scene ids, or <scene>/rgb/<frame> lines as in data/training/*/frames.txt), each alone but with the
seed its chunk gives it, so the result equals that scene of a full run. --frames-list renders
only the listed frames (lines <scene>/rgb/<frame file>, as in data/training/*/frames.txt) of the listed scenes: the
T1 training frames are 18,101 of the corpus's 69.6 million.

Layout:
    OUT/rollout/<split>/<family>/chunk_<offset>/raw/<scene>/      messages.jsonl, summary.json
                                               /plan/<scene>/     frames.jsonl, render_replay_plan.json
                                               /labels/<scene>/   labels.jsonl, summary.json
    OUT/render/<scene>/                                           rgb/frame_<frame>_env_<env>.png, summary.json

Every finished unit leaves a .done marker: a chunk's raw/.done, each scene's plan/<scene>/.done and
labels/<scene>/.done, and each render's .done (which records its settings). A rerun skips finished units and clears
and redoes partial ones, so a run stopped at any point resumes; a render whose settings differ from the request is
redone. A scene the physics cannot run is skipped by the rollout, listed in the chunk's raw/run_summary.json and in
the final summary. Stop a run with Ctrl-C, or by signalling its process group, so the rollouts stop with it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys

from go2nav.corpus.labels import derive as derive_labels
from go2nav.corpus.packs import find_scene_dirs
from go2nav.corpus.render_plan import build_plan

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Chunk:
    split: str
    family: str
    offset: int
    size: int
    first_seed_index: int = 0  # the position of its first scene within its full chunk (scenes run alone)

    def directory(self, out: Path) -> Path:
        return out/'rollout'/self.split/self.family/f'chunk_{self.offset:04d}'

    def __str__(self):
        return f'{self.split}/{self.family}@{self.offset}'


def chunks(corpus: Path, size: int, splits=None, families=None, scenes=None) -> list[Chunk]:
    groups = {}
    for scene in find_scene_dirs(corpus):
        family, split = scene.parent.name, scene.parent.parent.name
        if (splits is None or split in splits) and (families is None or family in families):
            groups.setdefault((split, family), []).append(scene.name)
    if scenes is not None:
        return [Chunk(split, family, index, 1, index % size) for (split, family), names in sorted(groups.items())
                for index, name in enumerate(sorted(names)) if name in scenes]
    return [Chunk(split, family, offset, min(size, len(names)-offset))
            for (split, family), names in sorted(groups.items()) for offset in range(0, len(names), size)]


def listed_scenes(path):
    """Scene ids from a list of scene ids or of <scene>/rgb/<frame> paths."""
    return {line.split('/')[0] for line in Path(path).read_text().split()}


def rollout(corpus, out, *, splits=None, families=None, scenes=None, chunk_size=40, workers=16, cores_per_rollout=2,
            post_workers=8, seed=7, n_envs=48, n_blocks=200):
    corpus, out = Path(corpus).resolve(), Path(out).resolve()
    jobs = chunks(corpus, chunk_size, splits, families, listed_scenes(scenes) if scenes else None)
    print(f'{len(jobs)} chunks from {corpus}', flush=True)
    available = sorted(os.sched_getaffinity(0))
    core_sets = queue.Queue()
    for slot in range(workers):
        core_sets.put([available[(slot*cores_per_rollout+k) % len(available)] for k in range(cores_per_rollout)])
    failures, skipped = [], []
    with ThreadPoolExecutor(workers) as rollouts, ProcessPoolExecutor(post_workers) as post:
        active = {rollouts.submit(_rollout_chunk, chunk, corpus, out, core_sets, seed=seed, n_envs=n_envs,
                                  n_blocks=n_blocks): chunk for chunk in jobs}
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                unit = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    failures.append(f'{unit}: {type(error).__name__}: {error}')
                    print(f'[FAIL] {unit}: {error}', flush=True)
                    continue
                if isinstance(unit, Chunk):
                    print(f'[rollout] {unit} ({result["status"]}, {len(result["scenes"])} scenes)', flush=True)
                    skipped += result['skipped']
                    directory = unit.directory(out)
                    for scene in result['scenes']:
                        active[post.submit(_post_process, directory, scene, corpus)] = f'{unit} {scene}'
                else:
                    print(f'[scene] {unit} ({result})', flush=True)
    print(json.dumps(dict(chunks=len(jobs), failures=failures, skipped_scenes=skipped)), flush=True)
    return failures


def _rollout_chunk(chunk: Chunk, corpus: Path, out: Path, core_sets: queue.Queue, *, seed, n_envs, n_blocks):
    raw = chunk.directory(out)/'raw'
    if not (raw/'.done').exists():
        for stage in ('raw', 'plan', 'labels'):  # plans and labels of an earlier, partial rollout are stale
            shutil.rmtree(raw.parent/stage, ignore_errors=True)
        raw.parent.mkdir(parents=True, exist_ok=True)
        cores = core_sets.get()
        try:
            command = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, '-m', 'go2nav.corpus.bulk',
                       str(corpus), str(raw), '--split', chunk.split, '--family', chunk.family,
                       '--scene-offset', str(chunk.offset), '--scene-limit', str(chunk.size),
                       '--seed', str(seed+chunk.first_seed_index),
                       '--n-envs', str(n_envs), '--n-blocks', str(n_blocks)]
            _run(command, raw.parent/'rollout.log')
        finally:
            core_sets.put(cores)
        (raw/'.done').touch()
        status = 'done'
    else:
        status = 'already done'
    summary = json.loads((raw/'run_summary.json').read_text())
    return dict(status=status, scenes=summary['scenes'], skipped=summary['skipped'])


def _post_process(chunk_dir: Path, scene: str, corpus: Path) -> str:
    raw = chunk_dir/'raw'/scene
    steps = (('plan', lambda out: build_plan(raw, out)), ('labels', lambda out: derive_labels(raw, corpus, out)))
    done = []
    for name, make in steps:
        out = chunk_dir/name/scene
        if not (out/'.done').exists():
            shutil.rmtree(out, ignore_errors=True)
            make(out)
            (out/'.done').touch()
            done.append(name)
    return ' + '.join(done) if done else 'already done'


def render(corpus, out, *, render_python, workers=8, egl_device=1, resolution=224, textures=False, max_frames=None,
           frames_list=None):
    corpus, out = Path(corpus).resolve(), Path(out).resolve()
    plans = sorted(marker.parent for marker in out.glob('rollout/*/*/chunk_*/plan/*/.done'))
    listed = None
    if frames_list is not None:
        listed = {}
        for line in Path(frames_list).read_text().split():
            scene, _, name = line.split('/')
            listed.setdefault(scene, []).append(name)
        plans = [plan for plan in plans if plan.name in listed]
    print(f'{len(plans)} planned scenes to render under {out}', flush=True)
    failures = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {}
        for plan in plans:
            frames = sorted(listed[plan.name]) if listed is not None else None
            settings = dict(resolution=resolution, textures=textures, max_frames=max_frames,
                            frames_sha256=hashlib.sha256('\n'.join(frames).encode()).hexdigest() if frames else None)
            futures[pool.submit(_render_scene, plan, corpus, out/'render', render_python, egl_device, settings,
                                frames)] = plan
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                plan = futures.pop(future)
                try:
                    print(f'[render] {plan.name} ({future.result()})', flush=True)
                except Exception as error:
                    failures.append(f'{plan.name}: {error}')
                    print(f'[FAIL] {plan.name}: {error}', flush=True)
    print(json.dumps(dict(scenes=len(plans), failures=failures)), flush=True)
    return failures


def _render_scene(plan_dir: Path, corpus: Path, render_root: Path, render_python, egl_device, settings,
                  frames=None) -> str:
    plan = json.loads((plan_dir/'render_replay_plan.json').read_text())
    out = render_root/plan['scene_id']
    marker = out/'.done'
    if marker.exists() and json.loads(marker.read_text()).get('settings') == settings:
        return 'already done'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    command = [str(render_python), '-m', 'go2nav.corpus.render', '--plan', str(plan_dir/'render_replay_plan.json'),
               '--scene-corpus', str(corpus), '--out', str(out), '--resolution', str(settings['resolution'])]
    command += ['--textures'] if settings['textures'] else []
    command += ['--max-frames', str(settings['max_frames'])] if settings['max_frames'] else []
    if frames is not None:
        (out/'frames.txt').write_text('\n'.join(frames)+'\n')
        command += ['--frames', str(out/'frames.txt')]
    env = dict(EGL_DEVICE_ID=str(egl_device), PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1')
    exit_code = _run(command, out/'render.log', env=env, check=False)
    # Genesis 0.3.14 can exit non-zero at teardown after a complete render, so the output decides.
    available = len(frames) if frames is not None else plan['frame_count']
    expected = min(available, settings['max_frames'] or available)
    summary = json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    frames = sum(1 for _ in (out/'rgb').glob('*.png')) if (out/'rgb').is_dir() else 0
    if not summary.get('frame_count') == frames == expected:
        raise RuntimeError(f'{frames} of {expected} frames rendered (exit {exit_code}; see {out/"render.log"})')
    marker.write_text(json.dumps(dict(settings=settings, frames=frames, exit_code=exit_code)))
    return f'{frames} frames'


def _run(command, log_path: Path, *, env=None, check=True) -> int:
    environment = os.environ | (env or {})
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO), environment.get('PYTHONPATH')]))
    with open(log_path, 'w') as log:
        code = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    if check and code != 0:
        raise RuntimeError(f'exit {code}; see {log_path}')
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = p.add_subparsers(dest='stage', required=True)
    r = stages.add_parser('rollout', help='bulk rollouts, render plans and labels')
    r.add_argument('corpus', type=Path)
    r.add_argument('out', type=Path)
    r.add_argument('--split', action='append', dest='splits')
    r.add_argument('--family', action='append', dest='families')
    r.add_argument('--scenes', type=Path, help='roll out only these scenes (see the module docstring)')
    r.add_argument('--chunk', type=int, default=40)
    r.add_argument('--workers', type=int, default=16)
    r.add_argument('--cores-per-rollout', type=int, default=2)
    r.add_argument('--post-workers', type=int, default=8)
    r.add_argument('--seed', type=int, default=7)
    r.add_argument('--n-envs', type=int, default=48)
    r.add_argument('--n-blocks', type=int, default=200)
    g = stages.add_parser('render', help='camera frames for every planned scene')
    g.add_argument('corpus', type=Path)
    g.add_argument('out', type=Path)
    g.add_argument('--render-python', type=Path, required=True)
    g.add_argument('--workers', type=int, default=8)
    g.add_argument('--egl-device', type=int, default=1)
    g.add_argument('--resolution', type=int, default=224)
    g.add_argument('--textures', action='store_true')
    g.add_argument('--max-frames', type=int)
    g.add_argument('--frames-list', type=Path)
    a = p.parse_args()
    if a.stage == 'rollout':
        failures = rollout(a.corpus, a.out, splits=a.splits, families=a.families, scenes=a.scenes, chunk_size=a.chunk,
                           workers=a.workers, cores_per_rollout=a.cores_per_rollout, post_workers=a.post_workers,
                           seed=a.seed, n_envs=a.n_envs, n_blocks=a.n_blocks)
    else:
        failures = render(a.corpus, a.out, render_python=a.render_python, workers=a.workers, egl_device=a.egl_device,
                          resolution=a.resolution, textures=a.textures, max_frames=a.max_frames,
                          frames_list=a.frames_list)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
