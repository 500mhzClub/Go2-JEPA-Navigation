"""What each H-JEPA level has captured, and how well its own predictions hold up (docs/hjepa/PLAN.md, gate H4).

    python -m go2nav.eval.hjepa_levels CHECKPOINT --corpus ROOT --out FILE [--fit-share 0.6] [--mazes N] [--device cuda:0]

Every frame of the validation units (`lewm_val`: explore, mission and free) is encoded through the hierarchy without
striding: the upper encoders are pointwise, so each frame has a latent at every level (paper §D). The validation mazes
are split: probes are fitted on the first `--fit-share` of them and scored on the rest, so nothing is scored on a maze
it saw.

**Probes** (paper §3, §D): per level and entity, a two-layer MLP (512, LayerNorm, GELU) on the frozen latent, scored as
variance-normalised MSE (NMSE: 1 is the marginal mean, 0 exact):
- slow: position in the maze (x, y) and heading (cos, sin of yaw);
- fast: body roll and pitch and their rates, which carry the gait's oscillation;
- motion: forward speed and yaw rate, set by the commands.
H-JEPA's prediction: upper levels keep the slow entities and lose the fast ones. Also per level, the within-unit share
of the latent variance (`chi`, as in `eval.lewm_corpus_probe`).

**Predictions**, on the evaluation mazes' mission and explore units: each upper level predicts from its context of
true states and the true macro-actions; the slow probes decode the prediction, and the decoded position and heading
are compared with where the robot really was. References: persistence (the last context state decoded), the encoded
true future (the probe's own ceiling), and a shuffled final macro-action. L2 is also rolled out for three steps
(1.6, 3.2 and 4.8 s); L3 predicts one step (12.8 s).

Absolute maze position is not visible in an egocentric frame of an unseen maze, so the predictions are also judged
by **displacement**: per upper level, a probe on a pair of consecutive states [S_t, S_(t+1) - S_t] reads the motion
between them (forward, left, turn) in the frame of S_t, fitted on true pairs of the fit mazes. It decodes each
predicted transition, composed over the rollout, against the true motion. References: no motion, the encoded true
transitions (the probe's ceiling), a shuffled final macro-action, and dead reckoning: the same probe fitted on the
transition's raw command blocks alone. A prediction that does not beat dead reckoning adds nothing to the commands.

**Cost geometry** (paper Fig. 7, Table 1): per evaluation unit, anchors every 100 frames; for every other frame, the
latent distance to the anchor at each level against the distance through the maze (breadth-first over the layout's
passages between the frames' cells, times the 1.3-m pitch) and the straight-line distance. Reported as Spearman
correlations, overall and on pairs where the maze path is at least 1.5 times the straight line (around walls), where a
useful planning cost must follow the maze rather than the line of sight.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from go2nav.data.lewm_corpus import SET, mazes_root, planar_yaw
from go2nav.models.hjepa import load_hjepa
from go2nav.models.lewm import control_stats, normalise_frames

KINDS = ('explore', 'mission', 'free')
ENTITIES = dict(position=(0, 2), heading=(2, 4), body=(4, 8), motion=(8, 10))
SLOW, FAST = ('position', 'heading'), ('body',)
L2_ROLLOUT = 3
ANCHOR_EVERY, DETOUR = 100, 1.5


def roll_pitch(q):
    x, y, z, w = np.moveaxis(np.asarray(q), -1, 0)
    return np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-z*x), -1, 1))


def labels(steps):
    """Per frame (n, 10): x, y, cos yaw, sin yaw, roll, pitch, roll rate, pitch rate, forward speed, yaw rate."""
    pose, twist = steps['pose'], steps['twist']
    yaw = planar_yaw(pose[:, 3:7])
    roll, pitch = roll_pitch(pose[:, 3:7])
    forward = np.cos(yaw)*twist[:, 0]+np.sin(yaw)*twist[:, 1]
    return np.stack([pose[:, 0], pose[:, 1], np.cos(yaw), np.sin(yaw), roll, pitch, twist[:, 3], twist[:, 4], forward,
                     twist[:, 5]], 1).astype(np.float32)


@torch.no_grad()
def encode_units(model, root, units, device, batch=1024):
    """Per unit: latents at every level (list of (n, D) float16 arrays), labels, normalised action blocks, validity."""
    mean, std = control_stats()
    out = []
    for u in units:
        folder = root/u['path']
        frames = np.load(folder/'frames.npy', mmap_mode='r')
        with np.load(folder/'steps.npz') as steps:
            y = labels(steps)
            blocks = ((steps['executed'][:, :, [0, 2]]-mean)/std).reshape(len(y), -1).astype(np.float32)
            valid = steps['block_valid'].copy()
        levels = [[] for _ in range(1+len(model.upper))]
        for s in range(0, len(frames), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames[s:s+batch])).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                z = model.level1.encode(normalise_frames(x)[:, None])
                for k in range(len(levels)):
                    if k:
                        z = model.upper[k-1].encode(z)
                    levels[k].append(z[:, 0].float().cpu().numpy().astype(np.float16))
        out.append(dict(unit=u, z=[np.concatenate(v) for v in levels], y=y, blocks=blocks, valid=valid))
    return out


class Probe(nn.Module):
    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def fit_probe(x, y, device, steps=4000, batch=1024, seed=0):
    """A probe on z-scored targets; returns a function from latents to de-normalised predictions."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    xm, xs, ym, ys = x.mean(0), x.std(0)+1e-6, y.mean(0), y.std(0)+1e-6
    probe = Probe(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xd, yd = ((x-xm)/xs).to(device), ((y-ym)/ys).to(device)
    for _ in range(steps):
        i = torch.randint(len(xd), (batch,), generator=g).to(device)
        loss = (probe(xd[i])-yd[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    probe.eval()

    @torch.no_grad()
    def apply(z):
        z = torch.as_tensor(np.asarray(z, np.float32))
        return torch.cat([probe(((z[k:k+65536]-xm)/xs).to(device)).cpu()
                          for k in range(0, len(z), 65536)]).mul(ys).add(ym).numpy()
    return apply


def nmse(pred, true):
    return float(np.mean(((pred-true)**2).mean(0)/true.var(0)))


def within_share(z, unit):
    z = z.astype(np.float64)
    total = ((z-z.mean(0))**2).sum()
    within = sum(((z[unit == u]-z[unit == u].mean(0))**2).sum() for u in np.unique(unit))
    return float(within/total)


def probes(data, fit, score, device):
    """Per level and entity, the fitted probes and their NMSE on the evaluation mazes."""
    report, fitted = {}, {}
    for level in range(len(data[0]['z'])):
        zf = np.concatenate([data[i]['z'][level] for i in fit])
        yf = np.concatenate([data[i]['y'] for i in fit])
        zs = np.concatenate([data[i]['z'][level] for i in score])
        ys = np.concatenate([data[i]['y'] for i in score])
        row = {}
        for name, (a, b) in ENTITIES.items():
            probe = fit_probe(zf, yf[:, a:b], device)
            fitted[(level, name)] = probe
            row[name] = nmse(probe(zs), ys[:, a:b])
        row['chi'] = within_share(zs, np.concatenate([np.full(len(data[i]['y']), i) for i in score]))
        report[f'l{level+1}'] = row
    return report, fitted


def relative(a, b):
    """Motion from labels a to labels b (..., 10) in a's frame: (..., 4) forward, left, cos and sin of the turn."""
    yaw_a, yaw_b = np.arctan2(a[..., 3], a[..., 2]), np.arctan2(b[..., 3], b[..., 2])
    dx, dy = b[..., 0]-a[..., 0], b[..., 1]-a[..., 1]
    c, s = np.cos(yaw_a), np.sin(yaw_a)
    return np.stack([c*dx+s*dy, -s*dx+c*dy, np.cos(yaw_b-yaw_a), np.sin(yaw_b-yaw_a)], -1).astype(np.float32)


def compose(motions):
    """Chain per-transition motions [(B, 4), ...] (each in the frame it starts from): (B, 3) x, y, yaw in the first."""
    x = np.zeros((len(motions[0]), 3))
    for m in motions:
        c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
        x = x+np.stack([c*m[:, 0]-s*m[:, 1], s*m[:, 0]+c*m[:, 1], np.arctan2(m[:, 3], m[:, 2])], 1)
    return x


def pair_features(a, b):
    return np.concatenate([a, b-a], -1)


def motion_probes(model, data, fit, device, rng, most=200000):
    """Per upper level, displacement probes fitted on the fit mazes' true transitions one period apart: from the
    pair of states, and (dead reckoning) from the transition's raw command blocks."""
    out = {}
    for k in range(len(model.upper)):
        n_level, period = k+2, model.config.periods[k+1]
        rows = [(i, s) for i in fit for s in windows(data, i, period+1, max(period//4, 4))]
        rows = [rows[j] for j in rng.permutation(len(rows))[:most]]
        pairs = np.stack([pair_features(data[i]['z'][n_level-1][s].astype(np.float32),
                                        data[i]['z'][n_level-1][s+period].astype(np.float32)) for i, s in rows])
        blocks = np.stack([data[i]['blocks'][s:s+period].reshape(-1) for i, s in rows])
        target = np.stack([relative(data[i]['y'][s], data[i]['y'][s+period]) for i, s in rows])
        out[n_level] = dict(latent=fit_probe(pairs, target, device), commands=fit_probe(blocks, target, device))
    return out


def motion_errors(estimate, truth):
    """Median position (m) and heading (deg) error of composed motions (B, 3) against the true ones."""
    turn = np.degrees(np.abs(np.angle(np.exp(1j*(estimate[:, 2]-truth[:, 2])))))
    return dict(position_m=float(np.median(np.linalg.norm(estimate[:, :2]-truth[:, :2], axis=1))),
                heading_deg=float(np.median(turn)))


def windows(data, i, span, step):
    """Starts of windows of `span` frames whose action blocks are all valid, every `step` frames."""
    valid = data[i]['valid']
    return [s for s in range(0, len(valid)-span, step) if valid[s:s+span].all()]


def upper_rollout(level, states, macros, steps):
    """Roll an upper level out `steps` states past its context: states (B, c, D) true context, macros (B, c-1+steps,
    macro) for every transition from the first context state."""
    c = level.config.context
    sequence = states
    for k in range(steps):
        predicted = level.predict(sequence[:, -c:], macros[:, k:k+c])[:, -1:]
        sequence = torch.cat((sequence, predicted), 1)
    return sequence[:, states.size(1):]


@torch.no_grad()
def predictions(model, data, units, fitted, motion, device, rng):
    """Decoded upper-level predictions against where the robot was (see the module docstring)."""
    report = {}
    periods = model.config.periods
    for k, level in enumerate(model.upper):
        n_level = k+2
        period, c = periods[k+1], level.config.context
        steps = L2_ROLLOUT if n_level == 2 else 1
        span = (c-1+steps)*period
        rows = []
        for i in units:
            for s in windows(data, i, span+1, max(period//2, 8)):
                rows.append((i, s))
        if not rows:
            continue
        idx = [rows[j] for j in rng.permutation(len(rows))[:4096]]
        z = torch.as_tensor(np.stack([data[i]['z'][n_level-1][s:s+span+1:period] for i, s in idx]), dtype=torch.float32,
                            device=device)
        blocks = torch.as_tensor(np.stack([data[i]['blocks'][s:s+span] for i, s in idx]), device=device)
        truth = np.stack([data[i]['y'][s:s+span+1:period] for i, s in idx])     # (B, c + steps, 10)
        macros = model.macros(blocks, n_level).float()
        context = z[:, :c]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            predicted = upper_rollout(level, context, macros, steps).float()
            shuffled_macros = macros.clone()
            shuffled_macros[:, c-1] = macros[torch.randperm(len(macros), device=device), c-1]
            shuffled = upper_rollout(level, context, shuffled_macros, 1).float()
        decode = (fitted[(n_level-1, 'position')], fitted[(n_level-1, 'heading')])

        def errors(latent, target, decode=decode):
            p, h = (probe(latent.cpu().numpy()) for probe in decode)
            yaw_err = np.degrees(np.abs(np.angle(np.exp(1j*(np.arctan2(h[:, 1], h[:, 0])-np.arctan2(target[:, 3], target[:, 2]))))))
            return dict(position_m=float(np.median(np.linalg.norm(p-target[:, :2], axis=1))),
                        heading_deg=float(np.median(yaw_err)))
        probe, reckon = motion[n_level]['latent'], motion[n_level]['commands']
        z_np, pred_np, shuffled_np = z.cpu().numpy(), predicted.cpu().numpy(), shuffled.cpu().numpy()
        chain = np.concatenate([z_np[:, c-1:c], pred_np], 1)
        moved = dict(predicted=[], encoded_truth=[], dead_reckoning=[])
        blocks_np = blocks.cpu().numpy()
        out = {}
        for j in range(steps):
            target = truth[:, c+j]
            seconds = round((j+1)*period*.1, 1)
            out[f'{seconds}s'] = dict(predicted=errors(predicted[:, j], target), persistence=errors(context[:, -1], target),
                                      encoded_truth=errors(z[:, c+j], target),
                                      true_motion_m=float(np.median(np.linalg.norm(target[:, :2]-truth[:, c-1, :2], axis=1))))
            moved['predicted'].append(probe(pair_features(chain[:, j], chain[:, j+1])))
            moved['encoded_truth'].append(probe(pair_features(z_np[:, c-1+j], z_np[:, c+j])))
            moved['dead_reckoning'].append(reckon(blocks_np[:, (c-1+j)*period:(c+j)*period].reshape(len(idx), -1)))
            true = compose([relative(truth[:, c-1+m], truth[:, c+m]) for m in range(j+1)])
            row = {key: motion_errors(compose(v), true) for key, v in moved.items()}
            row['no_motion'] = motion_errors(np.zeros_like(true), true)
            if j == 0:
                out[f'{seconds}s']['shuffled_macro'] = errors(shuffled[:, 0], target)
                row['shuffled_macro'] = motion_errors(compose([probe(pair_features(z_np[:, c-1], shuffled_np[:, 0]))]), true)
            row['true_turn_deg'] = float(np.median(np.degrees(np.abs(true[:, 2]))))
            out[f'{seconds}s']['displacement'] = row
        report[f'l{n_level}'] = dict(windows=len(idx), **out)
    return report


def maze_distance(spec):
    """Path length (m) through the maze between cells, by breadth-first search over the layout's passages."""
    layout = spec['evaluation_layout']
    cells = [tuple(c) for c in layout['cells']]
    adjacency = {c: [] for c in cells}
    for a, b in layout['edges']:
        adjacency[tuple(a)].append(tuple(b))
        adjacency[tuple(b)].append(tuple(a))
    hops = {}
    for start in cells:
        seen, queue = {start: 0}, deque([start])
        while queue:
            c = queue.popleft()
            for d in adjacency[c]:
                if d not in seen:
                    seen[d] = seen[c]+1
                    queue.append(d)
        hops[start] = seen
    return cells, hops, layout['pitch_m']


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def geometry(data, units, root):
    """Spearman correlations of latent distance with maze-path and straight-line distance, per level."""
    from go2nav.mazes.sets import MazeSet
    mazes = MazeSet(SET, root=mazes_root(root))
    levels = len(data[0]['z'])
    collected = {key: [[] for _ in range(levels)] for key in ('all', 'detour')}
    distances = dict(all=[[], []], detour=[[], []])
    for i in units:
        unit = data[i]['unit']
        cells, hops, pitch = maze_distance(mazes.maze(unit['maze']))
        centres = np.asarray(cells, float)*pitch
        xy = data[i]['y'][:, :2]
        cell = np.argmin(((xy[:, None]-centres[None])**2).sum(-1), axis=1)
        latents = [z.astype(np.float32) for z in data[i]['z']]
        for anchor in range(0, len(xy), ANCHOR_EVERY):
            path = np.array([hops[cells[cell[anchor]]].get(cells[c], np.nan) for c in cell])*pitch
            line = np.linalg.norm(xy-xy[anchor], axis=1)
            keep = np.isfinite(path) & (np.arange(len(xy)) != anchor)
            detour = keep & (path >= DETOUR*np.maximum(line, pitch))
            for key, mask in (('all', keep), ('detour', detour)):
                distances[key][0].append(path[mask])
                distances[key][1].append(line[mask])
                for level, z in enumerate(latents):
                    collected[key][level].append(np.linalg.norm(z[mask]-z[anchor], axis=1))
    out = {}
    for key in ('all', 'detour'):
        path, line = np.concatenate(distances[key][0]), np.concatenate(distances[key][1])
        out[key] = dict(pairs=int(len(path)), **{
            f'l{level+1}': dict(maze_path=spearman(np.concatenate(collected[key][level]), path),
                                straight_line=spearman(np.concatenate(collected[key][level]), line))
            for level in range(levels)})
    return out


def run(checkpoint, corpus, *, fit_share=.6, mazes=None, device='cuda:0'):
    device = torch.device(device)
    root = Path(corpus)
    model = load_hjepa(checkpoint, device)
    units = [u for u in json.loads((root/'index_12.json').read_text())['units'] if u['role'] == 'lewm_val' and u['kind'] in KINDS]
    maze_ids = sorted({u['maze'] for u in units})
    if mazes:
        maze_ids = maze_ids[:mazes]
        units = [u for u in units if u['maze'] in maze_ids]
    fit_set = set(maze_ids[:max(1, int(round(len(maze_ids)*fit_share)))])
    data = encode_units(model, root, units, device)
    fit = [i for i, d in enumerate(data) if d['unit']['maze'] in fit_set]
    score = [i for i, d in enumerate(data) if d['unit']['maze'] not in fit_set]
    probe_report, fitted = probes(data, fit, score, device)
    motion = motion_probes(model, data, fit, device, np.random.default_rng(1))
    coherent = [i for i in score if data[i]['unit']['kind'] in ('explore', 'mission')]
    return dict(checkpoint=str(checkpoint), units=dict(fit=len(fit), score=len(score)),
                frames=int(sum(len(d['y']) for d in data)), probes=probe_report,
                predictions=predictions(model, data, coherent, fitted, motion, device, np.random.default_rng(0)),
                geometry=geometry(data, coherent, root))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--fit-share', type=float, default=.6, help='share of validation mazes the probes are fitted on')
    p.add_argument('--mazes', type=int, help='use only the first N validation mazes (a quick check)')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    report = run(a.checkpoint, a.corpus, fit_share=a.fit_share, mazes=a.mazes, device=a.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(dict(probes=report['probes'], predictions=report['predictions'],
                          geometry=report['geometry']), indent=1))


if __name__ == '__main__':
    main()
