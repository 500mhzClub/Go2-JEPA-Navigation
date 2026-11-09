"""Phase C's gates (docs/dynamic/PLAN.md P15), fixed before any model is scored, on one H-JEPA training run.

    python -m go2nav.eval.phase_c_gates run RUN_DIR --out DIR [--obs-root ROOT] [--heads-root ROOT] [--device cuda:0]

RUN_DIR holds `hjepa.pt`, `lewm.pt` (its level 1) and `evaluation.json` (`training.hjepa`). The give-way labels
(`eval.mover_probe labels`, with the 12.8-s fields) are `--train-labels` and `--eval-labels`. Splits: `val` (the
validation units of both layout families, every kind but transfer), `transfer` (the held-out looks) and, where a
comparison with earlier models needs it, `val_v2` (corpus v2's own validation kinds). Family C is never used.

1. **Give-way** (`eval.mover_probe.give_way`, P4e's probe): the retreat-minus-hold contrast of the mover's position,
   from a rollout readout (ridge or MLP), at level 1 (`lewm.pt`) and level 2 (`hjepa.pt --level 2`). It passes if one
   readout at one level has AUC >= 0.80 at 1.6 s and at 3.2 s, on `val` and on `transfer`.
   - At the 12.8-s level (`clear_way`): from a give-way window, whether the way is clear 25.6 s after a side step
     (`mover_probe.side_step_tape`: back along the robot's own path to the first side opening, turn into it, go 1 m in,
     hold; `way_clear`). Level 3's states at t - 12.8 s and t, two steps of its predictor under the side step's
     macro-actions, read by ridge and MLP readouts fitted on the training labels. It passes if one readout has
     AUC >= 0.75 on `val` and on `transfer`. Reported beside it: the same readouts on the state at t alone (does the
     rollout add anything?) and the base rates after the side step, after holding 25.6 s, and after the straight
     12.8-s retreat this gate first used (8 November: a straight retreat is the wrong move against a walker that does
     not yield, which follows it).
2. **Contact**: on level-1 windows (every `STRIDE`-th frame with 4 frames of history and 0.8 s after, valid action
   blocks throughout), the contact head on the model's 8-step rollout under the executed commands, against contact at
   t + 0.8 s (`data.head_labels`: clearance under 0.10 m). AUC >= 0.9 on `val` and on `transfer`. A model without
   heads (the control) is read by readouts fitted on the training units' latents; with stop-gradient heads that is
   the same test. Both are reported for a model with heads, and its own head decides.
3. **Own motion at standstill**: on the windows held at zero command (the action blocks at t - 1 and t all zero), the
   one-step motion head's bias: the norm of the mean planar error of [z_t, z_(t+1) - z_t] -> (dx, dy), in mm a frame.
   Under 1 mm on `val`, and the same head must beat a zero prediction where the robot moves (skill above 0 on the
   other `val` windows: 1 - its median planar error / the median planar motion), so a head that always reads zero
   cannot pass.
4. **Walkers**: the probe's MLP localisation at h = 0 (median error, `val_v2`, the windows LeWM v2 was scored on), no
   worse than LeWM v2's 0.216 m.
5. **Health** (`evaluation.json`): level 1's action sensitivity (L2) and both upper levels' (H2) pass, as
   `go2_3level_v2`'s did.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from go2nav.eval import mover_probe as mp

STRIDE, TRAIN_STRIDE, TRAIN_UNIT_EVERY = 5, 10, 8
CONTACT_H = 8
GIVE_WAY_AUC, CLEAR_WAY_AUC, CONTACT_AUC, MOTION_BIAS_MM, LOCALISATION_M = .80, .75, .90, 1., .216
GIVE_WAY_HORIZONS_S = (1.6, 3.2)
SPLITS = ('val', 'transfer')


def units(root, role, every=1):
    from go2nav.data import lewm_corpus as v1
    folders = v1.unit_folders(Path(root), [role])
    return [str(f.relative_to(root)) for f in folders][::every]


def split_of(path):
    return 'transfer' if str(path).endswith('_transfer') else 'val'


def level1_windows(model, root, paths, device, *, obs_root=None, heads_root=None, stride=STRIDE, horizon=CONTACT_H):
    """Level-1 windows of `paths` (see the module docstring): per window the z history (W, n, D), the rollout's z at
    t + horizon (W, D), the true z at t + horizon and at t + 1, and the labels (contact at t and t + horizon, the
    body-frame motion t -> t + 1, held), its unit and split, the applied-command history at t (W, 15, 3) and the
    executed commands over the horizon (W, horizon, 3)."""
    import torch
    from go2nav.models.lewm import action_blocks, body_motion, control_stats, normalise_frames
    from go2nav.data.lewm_corpus import planar_yaw
    level1 = model.level1
    n, cmean, cstd = level1.config.history, *control_stats(level1.config.command_dim)
    out = {k: [] for k in ('zh', 'rolled', 'true_h', 'z_next', 'contact_t', 'contact_h', 'motion', 'held', 'unit',
                           'split', 'history', 'tape')}
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    with torch.inference_mode():
        for u, path in enumerate(paths):
            folder = Path(root)/path
            steps = dict(np.load(folder/'steps.npz'))
            valid, executed = steps['block_valid'], steps['executed']
            total = len(valid)
            t = np.array([s for s in range(n-1, total-horizon, stride) if valid[s-n+1:s+horizon].all()], int)
            if not len(t):
                continue
            needed = np.unique(np.concatenate([t[:, None]+np.arange(1-n, 1), t[:, None]+[1, horizon]], 1))
            frames = np.load(folder/'frames.npy', mmap_mode='r')
            z = torch.zeros(total, level1.config.latent, device=device)
            for s in range(0, len(needed), 512):
                chunk = needed[s:s+512]
                x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
                with autocast():
                    z[torch.from_numpy(chunk).to(device)] = level1.encode(
                        normalise_frames(x)[:, None], mp.unit_obs(obs_root, path, chunk, device))[:, 0].float()
            tt = torch.from_numpy(t).to(device)
            blocks = action_blocks(executed, cmean, cstd).to(device)
            zh = torch.stack([z[tt+k] for k in range(1-n, 1)], 1)
            actions = torch.stack([blocks[tt+k] for k in range(1-n, horizon)], 1)
            with autocast():
                rolled = level1.rollout(zh, actions, horizon)[:, -1].float()
            pose = steps['pose'].astype(np.float64)
            planar = torch.from_numpy(np.stack([pose[:, 0], pose[:, 1], planar_yaw(pose[:, 3:7])], 1))
            motion = body_motion(torch.stack([planar[t], planar[t+1]], 1), 1)[:, 0].numpy()
            held = np.array([not np.any(executed[s-1:s+1]) for s in t])
            clearance = np.load(Path(heads_root)/path/'heads.npz')['clearance'].astype(np.float32) \
                if heads_root else np.full(total, np.nan, np.float32)
            for key, value in (('zh', zh), ('rolled', rolled), ('true_h', z[tt+horizon]), ('z_next', z[tt+1])):
                out[key].append(value.half().cpu())
            out['contact_t'].append(clearance[t] < mp_contact_m())
            out['contact_h'].append(clearance[t+horizon] < mp_contact_m())
            out['motion'].append(motion.astype(np.float32))
            out['held'].append(held)
            out['history'].append(steps['applied_history'][t].astype(np.float32))      # P16: the scene-blind null's inputs
            out['tape'].append(np.stack([executed[t+k].mean(1) for k in range(horizon)], 1).astype(np.float32))
            out['unit'].append(np.full(len(t), u))
            out['split'].append(np.full(len(t), split_of(path)))
    import torch as _t
    return {k: (_t.cat(v) if k in ('zh', 'rolled', 'true_h', 'z_next') else np.concatenate(v)) for k, v in out.items()}


def mp_contact_m():
    from go2nav.data.head_labels import CONTACT_M
    return CONTACT_M


def readouts(x_train, y_train, groups, device, updates=mp.UPDATES):
    """Ridge and MLP readouts (`mover_probe.fit_ridge`, `fit_mlp`) fitted on (x, y); returns {name: predictor}."""
    import torch
    x, y = x_train.float().to(device), torch.as_tensor(y_train, dtype=torch.float32, device=device)
    y = y[:, None] if y.dim() == 1 else y
    return dict(ridge=mp.fit_ridge(x, y, groups), mlp=mp.fit_mlp(x, y, updates=updates))


def by_split(values, label, split):
    """AUC of `values` for `label` per split, with its unit-bootstrap interval left out (the gate reads the point)."""
    return {s: round(mp.auc(values[split == s], label[split == s]), 3) if (split == s).any() else None for s in SPLITS}


def contact_and_motion(model, root, device, *, obs_root, heads_root, updates):
    """Gates 2 and 3 (see the module docstring)."""
    import torch
    from go2nav.models.hjepa import MOTION_SCALE_PER_STEP
    train = level1_windows(model, root, units(root, 'lewm_train', TRAIN_UNIT_EVERY), device, obs_root=obs_root,
                           heads_root=heads_root, stride=TRAIN_STRIDE)
    val = level1_windows(model, root, units(root, 'lewm_val'), device, obs_root=obs_root, heads_root=heads_root)
    pairs = lambda d: torch.cat((d['zh'][:, -1].float(), (d['z_next']-d['zh'][:, -1]).float()), -1)
    scale = np.asarray(MOTION_SCALE_PER_STEP, np.float32)
    report = dict(train_windows=int(len(train['unit'])), val_windows=int(len(val['unit'])),
                  contact_share={s: round(float(val['contact_h'][val['split'] == s].mean()), 4) for s in SPLITS},
                  held_windows=int((val['held'] & (val['split'] == 'val')).sum()), contact={}, motion={})
    contact_fits = readouts(train['zh'][:, -1], train['contact_t'].astype(np.float32), train['unit'], device, updates)
    motion_fits = readouts(pairs(train), train['motion']/scale, train['unit'], device, updates)
    held = val['held'] & (val['split'] == 'val')

    moving = ~val['held'] & (val['split'] == 'val')

    def motion_bias(predicted):
        error = predicted*scale-val['motion']
        planar = np.linalg.norm(error[:, :2], axis=1)
        return dict(bias_mm=round(1000*float(np.linalg.norm(error[held, :2].mean(0))), 3),
                    median_error_mm=round(1000*float(np.median(planar[held])), 3),
                    moving_skill=round(1-float(np.median(planar[moving]))
                                       / float(np.median(np.linalg.norm(val['motion'][moving, :2], axis=1))), 3))
    with torch.no_grad():
        for name, fit in contact_fits.items():
            score = lambda z: fit(z.float().to(device))[:, 0].cpu().numpy()
            report['contact'][f'readout_{name}'] = dict(
                rollout=by_split(score(val['rolled']), val['contact_h'], val['split']),
                true_at_h=by_split(score(val['true_h']), val['contact_h'], val['split']),
                at_t=by_split(score(val['zh'][:, -1]), val['contact_t'], val['split']))
        for name, fit in motion_fits.items():
            report['motion'][f'readout_{name}'] = motion_bias(fit(pairs(val).to(device)).cpu().numpy())
        if model.heads is not None:
            h = model.heads
            logit = lambda z: h.run(h.contact, z.float().to(device))[..., 0].float().cpu().numpy()
            report['contact']['head'] = dict(
                rollout=by_split(logit(val['rolled']), val['contact_h'], val['split']),
                true_at_h=by_split(logit(val['true_h']), val['contact_h'], val['split']),
                at_t=by_split(logit(val['zh'][:, -1]), val['contact_t'], val['split']))
            report['motion']['head'] = motion_bias(h.run(h.motion, pairs(val).to(device)).float().cpu().numpy())
    decider = 'head' if model.heads is not None else None
    contact_pass = (lambda e: all((e['rollout'][s] or 0) >= CONTACT_AUC for s in SPLITS))
    motion_pass = (lambda e: e['bias_mm'] < MOTION_BIAS_MM and e['moving_skill'] > 0)
    if decider:
        report['contact_pass'], report['motion_pass'] = contact_pass(report['contact'][decider]), \
            motion_pass(report['motion'][decider])
    else:
        report['contact_pass'] = any(contact_pass(e) for e in report['contact'].values())
        report['motion_pass'] = any(motion_pass(e) for e in report['motion'].values())
    report['decided_by'] = decider or 'readouts'
    return report


def clear_way(model, root, labels_train, labels_eval, device, *, obs_root, updates):
    """Gate 1's 12.8-s part (see the module docstring)."""
    import torch
    from go2nav.models.lewm import action_blocks, control_stats, normalise_frames
    level = 3
    upper, p = model.upper[level-2], model.config.periods[level-1]
    n = upper.config.context
    cmean, cstd = control_stats(model.config.level1.command_dim)
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)

    from go2nav.data import lewm_corpus_v2 as v2
    from go2nav.models.hjepa import upper_rollout
    steps_ahead = mp.SIDE_FRAMES//p

    def features(data):
        rows_all = np.flatnonzero(~np.isnan(data['clear_side']))
        out = dict(now=[], back=[], rolled=[], rows=[])
        with torch.inference_mode():
            for u, path in enumerate(data['units']):
                rows = rows_all[data['unit'][rows_all] == u]
                if not len(rows):
                    continue
                steps = dict(np.load(Path(root)/str(path)/'steps.npz'))
                t = data['t'][rows]
                ok = np.array([s >= (n-1)*p and steps['block_valid'][s-(n-1)*p:s].all() for s in t], bool)
                rows, t = rows[ok], t[ok]
                if not len(rows):
                    continue
                offsets = np.arange(-(n-1), 1)*p
                needed = np.unique((t[:, None]+offsets).ravel())
                frames = np.load(Path(root)/str(path)/'frames.npy', mmap_mode='r')
                z = torch.zeros(len(frames), model.config.level1.latent, device=device)
                for s in range(0, len(needed), 512):
                    chunk = needed[s:s+512]
                    x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
                    with autocast():
                        z[torch.from_numpy(chunk).to(device)] = model.level1.encode(
                            normalise_frames(x)[:, None], mp.unit_obs(obs_root, path, chunk, device))[:, 0].float()
                tt = torch.from_numpy(t).to(device)
                states = model.states(torch.stack([z[tt+o] for o in offsets], 1), level).float()
                blocks = action_blocks(steps['executed'], cmean, cstd).to(device)
                past = torch.stack([blocks[tt-(n-1)*p+k] for k in range((n-1)*p)], 1)
                geometry = v2.FreeDriverV2(mp.unit_spec(root, json.loads((Path(root)/str(path)/'meta.json').read_text())),
                                           np.random.default_rng(0))
                pose = steps['pose']
                tape = np.stack([mp.side_step_tape(geometry, pose[s], pose[s::-1, :2], previous)[0]
                                 for s, previous in zip(t, data['previous'][rows])])
                macros = model.macros(torch.cat((past, action_blocks(tape, cmean, cstd).to(device)), 1), level)
                with autocast():
                    out['rolled'].append(upper_rollout(upper, states, macros.float(), steps_ahead)[:, -1].float().cpu())
                out['now'].append(states[:, -1].cpu())
                out['back'].append(states[:, -2].cpu())
                out['rows'].append(rows)
        return {k: (np.concatenate(v) if k == 'rows' else torch.cat(v)) for k, v in out.items()}
    train, ev = (dict(np.load(f)) for f in (labels_train, labels_eval))
    ftrain, feval = features(train), features(ev)
    rich = lambda f: torch.cat((f['now'], f['rolled']-f['now'], f['now']-f['back']), -1)
    plain = lambda f: torch.cat((f['now'], f['now']-f['back']), -1)
    label_train, label_eval = train['clear_side'][ftrain['rows']], ev['clear_side'][feval['rows']] > .5
    split = mp.kinds(ev)[feval['rows']]
    share = lambda key: {s: round(float((ev[key][feval['rows']] > .5)[split == s].mean()), 3) for s in SPLITS}
    report = dict(train_windows=int(len(ftrain['rows'])), eval_windows=int(len(feval['rows'])),
                  clear_after_side_step=share('clear_side'), clear_after_hold_25s=share('clear_hold_long'),
                  clear_after_straight_retreat_12s=share('clear_retreat'), opening_found=share('side_found'),
                  rollout={}, state_only={})
    for key, inputs in (('rollout', rich), ('state_only', plain)):
        for name, fit in readouts(inputs(ftrain), label_train, train['unit'][ftrain['rows']], device, updates).items():
            with torch.no_grad():
                score = fit(inputs(feval).to(device))[:, 0].cpu().numpy()
            report[key][name] = by_split(score, label_eval, split)
    report['pass'] = any(all((v[s] or 0) >= CLEAR_WAY_AUC for s in SPLITS) for v in report['rollout'].values())
    return report


def give_way_gate(probes):
    """Gate 1 at 1.6 and 3.2 s from the probe results {level: results.json}."""
    table, passed = {}, False
    for level, results in probes.items():
        for readout in ('ridge_rollout', 'mlp_rollout'):
            entry = {}
            for s in SPLITS:
                rows = {round(r['horizon_s'], 1): r for r in results['heads'][readout][s]}
                entry[s] = {h: (rows[h]['give_way']['retreat'].get('auc') if h in rows else None)
                            for h in GIVE_WAY_HORIZONS_S}
            table[f'level{level}_{readout}'] = entry
            passed |= all((entry[s][h] or 0) >= GIVE_WAY_AUC for s in SPLITS for h in GIVE_WAY_HORIZONS_S)
    return dict(retreat_auc=table, pass_=passed)


def localisation(probe_l1):
    m = probe_l1['heads']['mlp_true']['val_v2'][0]
    return dict(mlp_h0_median_m=m['median_m'], reference_m=LOCALISATION_M, pass_=m['median_m'] <= LOCALISATION_M)


def contact_motion_cached(model, out, root, device, **kw):
    """Gates 2 and 3, kept in OUT/contact_motion.json: they need no give-way labels, so they can run ahead (`contact`)."""
    cache = Path(out)/'contact_motion.json'
    if cache.exists():
        return json.loads(cache.read_text())
    report = contact_and_motion(model, root, device, **kw)
    cache.write_text(json.dumps(report, indent=1, default=str)+'\n')
    return report


def health(run):
    e = json.loads((Path(run)/'evaluation.json').read_text())
    checks = dict(l1_action_sensitivity=e['validation']['l2_action_sensitivity'],
                  **{f'{k}_action_sensitivity': v['h2_action_sensitivity'] for k, v in e['upper'].items()})
    return dict(**checks, l1_effective_rank=round(e['validation']['effective_rank'], 1), pass_=all(checks.values()))


def run(run_dir, out, *, root, train_labels, eval_labels, obs_root=None, heads_root=None, device='cuda:0',
        updates=mp.UPDATES):
    from go2nav.models.hjepa import load_hjepa
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(run=str(run_dir))
    save = lambda: (out/'gates.json').write_text(json.dumps(report, indent=1, default=str)+'\n')
    probes = {}
    for level, checkpoint in ((1, run_dir/'lewm.pt'), (2, run_dir/'hjepa.pt')):
        probes[level] = mp.probe(root, checkpoint, train_labels, eval_labels, out/f'probe_level{level}', updates=updates,
                                 device=device, level=level, obs_root=obs_root)
    report['give_way'] = give_way_gate(probes)
    report['walkers'] = localisation(probes[1])
    save()
    model = load_hjepa(run_dir/'hjepa.pt', device)
    report['clear_way'] = clear_way(model, root, train_labels, eval_labels, device, obs_root=obs_root, updates=updates)
    save()
    report['contact_motion'] = contact_motion_cached(model, out, root, device, obs_root=obs_root,
                                                     heads_root=heads_root, updates=updates)
    finished = (run_dir/'evaluation.json').exists()       # an interim checkpoint (`interim`) has no health check
    report['health'] = health(run_dir) if finished else None
    report['gates'] = dict(give_way=report['give_way']['pass_'] and report['clear_way']['pass'],
                           contact=report['contact_motion']['contact_pass'],
                           motion_at_standstill=report['contact_motion']['motion_pass'],
                           walkers=report['walkers']['pass_'], health=report['health']['pass_'] if finished else None)
    save()
    return report


def interim(checkpoint, folder):
    """A run folder for an interim checkpoint (hjepa_stepN.pt): the checkpoint as hjepa.pt and its level 1 exported
    as lewm.pt, so `run` scores it (without the health check, which needs the finished run's evaluation)."""
    from go2nav.models.hjepa import export_level1, load_hjepa
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder/'hjepa.pt').exists():
        (folder/'hjepa.pt').symlink_to(Path(checkpoint).resolve())
    if not (folder/'lewm.pt').exists():
        export_level1(load_hjepa(checkpoint), folder/'lewm.pt', extra=dict(interim=str(checkpoint)))
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('run_dir', type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--root', type=Path, default=Path('/mnt/workspace_drive/go2nav_lewm_v2'))
    r.add_argument('--train-labels', required=True, type=Path)
    r.add_argument('--eval-labels', required=True, type=Path)
    r.add_argument('--obs-root', type=Path)
    r.add_argument('--heads-root', type=Path, required=True, help='data.head_labels (the contact labels)')
    r.add_argument('--device', default='cuda:0')
    r.add_argument('--updates', type=int, default=mp.UPDATES)
    i = sub.add_parser('interim', help='make a run folder for an interim checkpoint (then `run` it)')
    i.add_argument('checkpoint', type=Path)
    i.add_argument('folder', type=Path)
    c = sub.add_parser('contact', help='gates 2 and 3 only, ahead of the give-way labels')
    c.add_argument('run_dir', type=Path)
    c.add_argument('--out', required=True, type=Path)
    c.add_argument('--root', type=Path, default=Path('/mnt/workspace_drive/go2nav_lewm_v2'))
    c.add_argument('--obs-root', type=Path)
    c.add_argument('--heads-root', type=Path, required=True)
    c.add_argument('--device', default='cuda:0')
    c.add_argument('--updates', type=int, default=mp.UPDATES)
    a = p.parse_args()
    if a.command == 'interim':
        print(interim(a.checkpoint, a.folder))
        return
    if a.command == 'contact':
        from go2nav.models.hjepa import load_hjepa
        a.out.mkdir(parents=True, exist_ok=True)
        report = contact_motion_cached(load_hjepa(a.run_dir/'hjepa.pt', a.device), a.out, a.root, a.device,
                                       obs_root=a.obs_root, heads_root=a.heads_root, updates=a.updates)
        print(json.dumps(dict(contact=report['contact_pass'], motion=report['motion_pass']), indent=1))
        return
    report = run(a.run_dir, a.out, root=a.root, train_labels=a.train_labels, eval_labels=a.eval_labels,
                 obs_root=a.obs_root, heads_root=a.heads_root, device=a.device, updates=a.updates)
    print(json.dumps(report['gates'], indent=1))


if __name__ == '__main__':
    main()
