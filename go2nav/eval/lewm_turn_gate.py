"""The turn tail gate: C5's in-place turn forecasts near walls, on the LeWM corpus's validation mazes.

    python -m go2nav.eval.lewm_turn_gate --world-model LEWM.pt --corpus ROOT --decoder NAME=FIT.pt[:c5] \
        [--decoder NAME=READOUT.pt:rollout ...] --out FILE

Why: span8w C5 passed L3 and L4 (medians on C1's and C3's decisions) and still deadlocked in 12 benchmark episodes.
At those states (docs/lewm/STATUS.md, span8w benchmarks) the robot stands inside the clearance reserve and may only
leave by a turn whose forecast never loses more than 1 mm of clearance; C5's decoder forecast in-place turns that slid
21 mm (median) against 7.5 mm true, so no turn passed. Medians do not see this; the tail of the turn forecasts does.

Windows: every 12-frame window of the `lewm_val` explore, mission and free units whose eight future action blocks
command no translation and some yaw (in-place turns, from rest or not). Per window, C5 as deployed: frames t-3..t
encoded, the rollout under the executed blocks (three past, eight future), the decoder at each horizon; C6 (if the
fit has one) on the same z, applied-command history and per-tick commands. Error: the largest XY error against the
true body-frame motion over 100-800 ms. Reported for all turns and for turns starting within 0.6 m (and 0.5 m) of a
wall (base centre to the nearest wall box footprint, from the corpus maze files): median, p90 and the share above
10 mm, with the true and forecast translation. A decoder is given as NAME=PATH, with `:c5` (a T7 fit file, the
default) or `:rollout` (a `lewm_readout` base readout's rollout head).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from go2nav.data.lewm_corpus import body_motion
from go2nav.data.lewm_dataset import raise_open_file_limit
from go2nav.models.lewm import action_blocks, control_stats, load_heads, load_world_model, normalise_frames

KINDS = ('explore', 'mission', 'free')
NEAR_M = (.6, .5)
HORIZONS = 8


def wall_distance(boxes, xy):
    """Distance (m) from points xy (n, 2) to the nearest wall box footprint."""
    best = np.full(len(xy), np.inf)
    for w in boxes:
        c, h, a = np.asarray(w['centre_xyz'][:2]), np.asarray(w['size_xyz'][:2])/2, w['yaw_rad']
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = np.abs((xy-c)@R)
        best = np.minimum(best, np.linalg.norm(np.maximum(local-h, 0.), axis=1))
    return best


def turn_windows(root, units, index):
    """Per unit: window starts whose eight future blocks are in-place turns."""
    out = {}
    for u in units:
        starts = index['start'][index['unit'] == u]
        if not len(starts):
            continue
        with np.load(root/units[u]['path']/'steps.npz') as s:
            executed = s['executed']
        keep = []
        for st in starts:
            future = executed[st+3:st+3+HORIZONS]
            if np.all(future[..., :2] == 0) and np.any(future[..., 2] != 0):
                keep.append(int(st))
        if keep:
            out[u] = np.asarray(keep)
    return out


@torch.inference_mode()
def forecasts(model, heads, root, unit, starts, device):
    """True motion (n, 8, 3), wall distance at t (n,), and per head the forecast (n, 8, 3)."""
    meta = unit
    folder = root/meta['path']
    frames = np.load(folder/'frames.npy', mmap_mode='r')
    steps = np.load(folder/'steps.npz')
    t = starts+3
    need = np.unique(np.concatenate([t+k for k in range(-3, 1)]))
    x = torch.from_numpy(np.ascontiguousarray(frames[need])).to(device)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        z_need = model.encode(normalise_frames(x)[:, None])[:, 0].float()
    row = {int(f): i for i, f in enumerate(need)}
    zh = torch.stack([z_need[[row[int(f)] for f in t+k]] for k in range(-3, 1)], 1)
    cmean, cstd = control_stats()
    blocks = action_blocks(steps['executed'], cmean, cstd).to(device)
    tt = torch.from_numpy(t).to(device)
    acts = torch.stack([blocks[tt-3+k] for k in range(3+HORIZONS)], 1)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        rolled = model.rollout(zh, acts, HORIZONS).float()
    pose = steps['pose']
    truth = body_motion(pose, t[:, None], t[:, None]+np.arange(1, HORIZONS+1))
    mean, std = (torch.as_tensor(v, device=device, dtype=torch.float32) for v in (cmean, cstd))
    past = (torch.as_tensor(steps['applied_history'][t][..., [0, 2]], dtype=torch.float32, device=device)-mean)/std
    tape = (torch.as_tensor(steps['executed'][np.add.outer(t, np.arange(HORIZONS)), -1][..., [0, 2]],
                            dtype=torch.float32, device=device)-mean)/std
    n = len(t)
    out = {}
    for name, (kind, head) in heads.items():
        if kind == 'c6':
            hz = torch.arange(1, HORIZONS+1, device=device).repeat(n)
            m = head(zh.repeat_interleave(HORIZONS, 0), past.repeat_interleave(HORIZONS, 0),
                     tape.repeat_interleave(HORIZONS, 0), hz)
        else:
            m = head(zh.repeat_interleave(HORIZONS, 0), rolled.reshape(n*HORIZONS, -1))
        out[name] = m.reshape(n, HORIZONS, 3).cpu().numpy()
    rest = ~np.any(steps['applied_history'][t][:, -10:] != 0, axis=(1, 2))  # no command in the last second
    return truth, pose[t, :2], rest, out


def load_decoders(specs, device):
    from go2nav.training.lewm_readout import load_base
    heads = {}
    for spec in specs:
        name, path = spec.split('=', 1)
        path, _, kind = path.partition(':')
        if kind == 'rollout':
            heads[name] = ('c5', load_base(path, 'rollout', device).eval())
        else:
            c5, c6 = load_heads(path, device)
            heads[name] = ('c5', c5)
            heads[f'{name}/C6'] = ('c6', c6)
    return heads


def summary(err, trans_pred, trans_true):
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float('nan')
    return dict(n=int(len(err)), median_mm=q(err, 50), p90_mm=q(err, 90), over10mm=float(np.mean(err > 10)) if len(err) else float('nan'),
                translation_median_mm=q(trans_pred, 50), translation_p90_mm=q(trans_pred, 90),
                true_translation_median_mm=q(trans_true, 50), true_translation_p90_mm=q(trans_true, 90))


def run(world_model, corpus, decoders, out, *, device='cuda:0'):
    raise_open_file_limit()
    root = Path(corpus)
    meta = json.loads((root/'index_12.json').read_text())['units']
    index = np.load(root/'index_12.npz')
    units = {i: u for i, u in enumerate(meta) if u['role'] == 'lewm_val' and u['kind'] in KINDS}
    windows = turn_windows(root, units, index)
    model = load_world_model(world_model, device)
    heads = load_decoders(decoders, device)
    mazes = {}
    rows = {name: dict(err=[], tp=[], tt=[], dist=[], rest=[]) for name in heads}
    for u, starts in windows.items():
        m = units[u]['maze']
        if m not in mazes:
            mazes[m] = json.loads((root/f'mazes/lewm_sets_v1/maze_{m:02d}.json').read_text())['geometry']['wall_boxes']
        for k in range(0, len(starts), 512):
            truth, xy, rest, preds = forecasts(model, heads, root, units[u], starts[k:k+512], device)
            dist = wall_distance(mazes[m], xy)
            for name, p in preds.items():
                r = rows[name]
                r['err'].append(1000*np.linalg.norm(p[..., :2]-truth[..., :2], axis=-1).max(1))
                r['tp'].append(1000*np.linalg.norm(p[..., :2], axis=-1).max(1))
                r['tt'].append(1000*np.linalg.norm(truth[..., :2], axis=-1).max(1))
                r['dist'].append(dist)
                r['rest'].append(rest)
    report = dict(world_model=str(world_model), windows=int(sum(len(s) for s in windows.values())), decoders={})
    for name, r in rows.items():
        err, tp, tt, dist, rest = (np.concatenate(r[k]) for k in ('err', 'tp', 'tt', 'dist', 'rest'))
        subsets = {'all': np.ones(len(err), bool), **{f'near_{d}m': dist <= d for d in NEAR_M}, 'rest': rest,
                   'rest_near_0.5m': rest & (dist <= .5)}
        report['decoders'][name] = {k: summary(err[m], tp[m], tt[m]) for k, m in subsets.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--world-model', required=True, type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--decoder', action='append', required=True, help='NAME=PATH[:c5|:rollout]')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    report = run(a.world_model, a.corpus, a.decoder, a.out, device=a.device)
    for name, r in report['decoders'].items():
        print(f"{name:28s} " + '  '.join(f"{k}: n={v['n']} med {v['median_mm']:.1f} p90 {v['p90_mm']:.1f} "
                                         f"trans {v['translation_median_mm']:.1f}/{v['true_translation_median_mm']:.1f}"
                                         for k, v in r.items()))


if __name__ == '__main__':
    main()
