"""The pose carriers offline (docs/dynamic/PLAN.md, P10 step 2): how far each carries the pose before it is 5 cm off.

    python -m go2nav.eval.pose_carry WORLD.pt HEADS.pt --corpus ROOT --out DIR [--windows 1500] [--steps 100]

Windows of corpus v2's validation units (explore, mission and free), each starting on a frame with a walker in the
forward view within `NEAR_M` (`data.mover_labels`), at least 16 frames into the unit, with every block valid and no
physical stop. From the window's first frame the position is carried for `steps` frames, each by one carrier's motion
over the next 100 ms, in the axes of the true heading (the gyro, which `perception.carried` uses for rotation), and
compared with the true position after each step.

The carriers read what they read online (`perception.carried.Carrier`): the frame's recorded applied-command history
(`applied_history`), the value recorded at the next frame as the command over the step.
- **model**: C5. The four frames up to now encoded, the last three recorded commands then the step's, each filling its
  five-tick block (as `controllers.lewm.LeWMForecaster`), rolled out one step and read by the decoder.
- **c1**: the command-history model (T6) on the four frames' histories (ages 0-1.4 s, all valid), its first horizon.
- **command**: the command integrated as if executed exactly.

Writes `trajectories.npz` (per window: the true path so far and each carrier's error, by step) and `summary.json`: the
90th-percentile and median error by carried path, and each carrier's **limit**, the largest path bin edge up to which
the 90th-percentile error stays at most `LIMIT_ERROR_M`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data.lewm_corpus import planar_yaw
from go2nav.models.command import CommandHistoryModel, history_features, nominal
from go2nav.models.lewm import COMMAND_AXES, control_stats, load_heads, load_world_model, normalise_frames

NEAR_M = .8
HISTORY = 4
FIRST = 16
LIMIT_ERROR_M = .05
SEED = 2026110601
KINDS = ('explore', 'mission', 'free')
PATH_EDGES_M = np.array([0., .02, .05, .1, .2, .3, .5, .75, 1., 1.5, 2., 3.])


def windows(root, units, steps, count):
    """(unit index, start) pairs: see the module docstring."""
    rng = np.random.default_rng(SEED)
    found = []
    for u, unit in enumerate(units):
        path = root/unit['path']
        movers = np.load(path/'mover_columns.npy', mmap_mode='r')
        with np.load(path/'steps.npz') as s:
            valid = s['block_valid'].astype(bool)
        n = len(valid)
        last = n-steps-1 if unit.get('physical_stop') is None else -1
        present = movers[..., 0] > 0
        near = (present & (np.linalg.norm(np.asarray(movers[..., 1:], float), axis=-1) <= NEAR_M)).any(1)
        ok = np.cumsum(~valid)
        for s0 in range(FIRST, max(FIRST, last)):
            if near[s0] and ok[s0+steps]-ok[s0-HISTORY] == 0:
                found.append((u, s0))
    found = np.asarray(found)
    chosen = rng.choice(len(found), size=min(count, len(found)), replace=False)
    return found[np.sort(chosen)], len(found)


def c1_history(applied, frame_ns, k):
    """The 420 history features at frame k from four frames' recorded applied commands (100 ms apart)."""
    values = [applied[j] for j in range(k-3, k+1)]
    measured = [frame_ns[j]-np.arange(14, -1, -1)*100_000_000 for j in range(k-3, k+1)]
    return history_features(values, [np.ones((15, 3))]*4, measured, [frame_ns[j] for j in range(k-3, k+1)])


class ModelSteps:
    """C5's one-step motion for every step of a window, in one batch."""

    def __init__(self, world, heads, device):
        self.device = torch.device(device)
        self.model = load_world_model(world).to(self.device).eval()
        self.decoder = load_heads(heads)[0].to(self.device).eval()
        dim = self.model.config.command_dim
        self.axes = COMMAND_AXES[dim]
        self.mean, self.std = (torch.as_tensor(v, device=self.device) for v in control_stats(dim))

    def normalised(self, commands):
        commands = torch.as_tensor(np.asarray(commands)[..., self.axes], dtype=torch.float32, device=self.device)
        return (commands-self.mean)/self.std

    @torch.inference_mode()
    def __call__(self, frames, applied, start, steps):
        """(steps, 2) motion over the steps from frame `start` (frames start-3 .. start+steps-1 are read)."""
        pixels = torch.from_numpy(np.asarray(frames[start-HISTORY+1:start+steps])).to(self.device)[None]
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            z = self.model.encode(normalise_frames(pixels))[0]                       # (steps+3, ...)
            k = torch.arange(steps, device=self.device)
            history = z[k[:, None]+torch.arange(HISTORY, device=self.device)]      # (steps, 4, ...)
            commands = np.stack([np.concatenate((applied[start+j][-3:], applied[start+j+1][-1:]))
                                 for j in range(steps)])                           # (steps, 4, 3)
            actions = self.normalised(commands)[:, :, None].expand(-1, -1, 5, -1).flatten(2)
            rolled = self.model.rollout(history, actions, 1)
        motion = self.decoder(self.model.cls(history).float(), self.model.cls(rolled).float()[:, 0])
        return motion[:, :2].float().cpu().numpy()


def carry(motion, yaw, start_xy):
    """Positions (steps+1, 2) from body-frame steps (steps, 2) in the axes of the true headings yaw (steps,)."""
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.stack((c*motion[:, 0]-s*motion[:, 1], s*motion[:, 0]+c*motion[:, 1]), 1)
    return np.vstack((start_xy, start_xy+np.cumsum(world, 0)))


def summarise(path, errors):
    """By carried path bin: windows' steps, and each carrier's median and 90th-percentile error; each carrier's limit."""
    rows, limits = [], {}
    flat_path = path.ravel()
    for lo, hi in zip(PATH_EDGES_M[:-1], PATH_EDGES_M[1:]):
        inside = (flat_path >= lo) & (flat_path < hi)
        row = dict(path_m=[float(lo), float(hi)], steps=int(inside.sum()))
        for name, e in errors.items():
            v = e.ravel()[inside]
            row[name] = dict(median_m=float(np.median(v)), p90_m=float(np.percentile(v, 90))) if len(v) else None
        rows.append(row)
    for name in errors:
        limit = 0.
        for row in rows:
            if row[name] is None or row['steps'] < 200 or row[name]['p90_m'] > LIMIT_ERROR_M:
                break
            limit = row['path_m'][1]
        limits[name] = limit
    return rows, limits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('world', type=Path)
    p.add_argument('heads', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--c1', type=Path, default=None, help='the C1 command model (default: the t6 artefact)')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--windows', type=int, default=1500)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    from go2nav import artifacts
    units = [u for u in json.loads((a.corpus/'index_12.json').read_text())['units']
             if u['role'] == 'lewm_val' and u['kind'] in KINDS]
    chosen, available = windows(a.corpus, units, a.steps, a.windows)
    model = ModelSteps(a.world, a.heads, a.device)
    c1 = CommandHistoryModel.load(a.c1 or artifacts.path('t6_c1_command_model'))
    paths, errors = [], {k: [] for k in ('model', 'c1', 'command')}
    cache = {}
    for i, (u, s0) in enumerate(chosen):
        if u not in cache:
            cache.clear()
            path = a.corpus/units[u]['path']
            with np.load(path/'steps.npz') as s:
                pose, applied, frame_ns = s['pose'].astype(float), s['applied_history'].astype(float), s['frame_ns']
            cache[u] = (np.load(path/'frames.npy', mmap_mode='r'), pose, applied, frame_ns)
        frames, pose, applied, frame_ns = cache[u]
        xy, yaw = pose[s0:s0+a.steps+1, :2], planar_yaw(pose[s0:s0+a.steps+1, 3:7])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        paths.append(np.cumsum(step))
        commands = [applied[s0+j+1][-1] for j in range(a.steps)]
        motions = dict(model=model(frames, applied, s0, a.steps),
                       c1=np.stack([c1.predict(np.tile(commands[j], (1, 8, 1)), c1_history(applied, frame_ns, s0+j))[0, 0, :2]
                                    for j in range(a.steps)]),
                       command=np.stack([nominal(np.tile(c, (8, 1)))[0, :2] for c in commands]))
        for name, motion in motions.items():
            errors[name].append(np.linalg.norm(carry(motion, yaw[:-1], xy[0])[1:]-xy[1:], axis=1))
        if i % 100 == 0:
            print(f'{i}/{len(chosen)}', {k: round(float(np.median([e[-1] for e in v])), 4) for k, v in errors.items()},
                  flush=True)
    paths = np.asarray(paths)
    errors = {k: np.asarray(v) for k, v in errors.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out/'trajectories.npz', path=paths, windows=chosen, **errors)
    rows, limits = summarise(paths, errors)
    report = dict(windows=len(chosen), available=int(available), steps=a.steps, near_m=NEAR_M, limit_error_m=LIMIT_ERROR_M,
                  world=str(a.world), heads=str(a.heads), limits_m=limits, by_path=rows,
                  end_of_window={k: dict(median_m=float(np.median(v[:, -1])), p90_m=float(np.percentile(v[:, -1], 90)))
                                 for k, v in errors.items()})
    (a.out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(dict(limits_m=limits, end_of_window=report['end_of_window']), indent=1))


if __name__ == '__main__':
    main()
