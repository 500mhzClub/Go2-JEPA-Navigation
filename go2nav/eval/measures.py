"""Measures that isolate prediction (docs/dynamic/PLAN.md P16; the review of 9 November, docs/REVIEW_2026-11-09.md).

    python -m go2nav.eval.measures tapes --labels LABELS.npz --out TAPES.npy [--workers N]
    python -m go2nav.eval.measures run RUN_DIR --gates GATES_DIR --out DIR --train-labels T.npz --eval-labels E.npz
                                       --train-tapes T.npy --eval-tapes E.npy [--obs-root ROOT] --heads-root ROOT

P15's gates were matched by things that predict nothing: the give-way AUC by the robot's own command history and by the
walker's distance, the 12.8-s readout by the same readout without its rollout, the contact gate by "in contact now".
Each measure here is read beside what a model that predicts nothing scores (its **nulls**) and beside a privileged
**ceiling**, and an AUC passes by one rule (`half_gap`): the model closes at least half the gap between its best null
and the ceiling, on `val` and on `transfer`.

1. **The reaction** (`reaction`), on the **matched** give-way windows: those where the robot's applied commands were
   zero over the last 1.5 s and just before t. There the command history and every contrast tape are identical from
   window to window, so a scene-blind model's predicted contrast is a constant, and whatever varies in a model's
   contrast comes from the scene. Per tape (retreat, advance), horizon (1.6, 3.2 s) and split: the least-squares
   **slope** of the predicted contrast (the readout on the tape's rollout minus on hold's) on the true one, with an
   intercept (1 for a correct model; 95% interval over units); the **offset**, the intercept's size in metres (what
   the model moves every walker by, reacting or not: the leak; 0 for a correct model); and the AUC of the contrast's
   size for whether the walker responds, whose null is 0.5 by construction and whose ceiling is a classifier on the
   simulator's state (`reaction_ceiling`). It passes if one readout at one level has, for the retreat tape at both
   horizons on both splits, the slope's interval above `SLOPE_FLOOR` and the offset under `OFFSET_M`.
2. **The advantage of the side step over holding** (`advantage`), 25.6 s on: the label is "the side step leaves the
   way clear and holding does not". One readout is fitted on both tapes' rollouts against each tape's own outcome, so
   it can tell them apart only through the rollout; the model's score is its value under the side step minus under
   hold. Nulls: **scene-blind** (the same fit on the command history and the tape alone; the side step's tape is planned
   on the true walls, so it carries some of the scene) and **present-only** (no tape: 0.5 by construction). Ceiling:
   the same fit on the simulator's state at t with the tape. Beside them, a **critic on the state**: one readout per
   tape on the states at t; where it matches the rollout, the predictor adds nothing, though the decision can still be
   read from the state.
3. **New contacts** (`new_contacts`): windows not in contact at t, whether the body outline is within 0.10 m of a wall
   or walker 0.8 s on. The model: its head (or readouts) on the rollout under the executed commands. Nulls:
   **present-only** (the same reader, and readouts fitted for the future label, on the state at t) and **scene-blind**
   (readouts on the command history and the executed commands). Ceiling 1: under the executed commands the future is
   set by the present, up to the walkers' draws.
4. **Health** (`go2nav.training.lewm.input_shares`, `validate`'s `action_share`): how much of the latent each input
   sets, and how much one step's action sets.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from go2nav.eval import mover_probe as mp

SPLITS = ('val', 'transfer')
HORIZON_S = {2: 1.6, 3: 3.2}              # indices of `mover_probe.HORIZONS`
GAP_SHARE, SLOPE_FLOOR, OFFSET_M = .5, .5, mp.EVENT_M
STILL = 1e-3                              # a command under this is zero


def half_gap(model, null, ceiling):
    """The one pass rule for an AUC: `skill` is the share of the gap from the best null to the ceiling the model
    closes; it passes at `GAP_SHARE`. A ceiling no better than the null leaves nothing to close: no pass."""
    if model is None or ceiling is None or ceiling <= null:
        return dict(skill=None, passed=False)
    skill = (model-null)/(ceiling-null)
    return dict(skill=round(float(skill), 3), passed=bool(skill >= GAP_SHARE))


def command_history(root, data):
    """Each window's applied-command history at t (W, 15, 3), from the units' records."""
    out = np.zeros((len(data['t']), 15, 3), np.float32)
    for u, path in enumerate(data['units']):
        rows = np.flatnonzero(data['unit'] == u)
        if len(rows):
            with np.load(Path(root)/str(path)/'steps.npz') as steps:
                out[rows] = steps['applied_history'][data['t'][rows]]
    return out


def matched(data, history, rows, j):
    """The matched give-way windows among `rows` at horizon index j (see the module docstring)."""
    still = (np.abs(history[rows]).max((1, 2)) < STILL) & (np.abs(data['previous'][rows]).max(1) < STILL)
    return mp.give_way_windows(data, rows) & ~np.isnan(data['cf_hold'][rows, j, 0]) & still


def slope(predicted, true, groups, draws=1000):
    """The least-squares slope of the predicted contrast (n, 2) on the true one with an intercept per axis, its 95%
    interval over draws of `groups`, and the intercept's size in metres."""
    def fit(i):
        p, t = predicted[i], true[i]
        pc, tc = p-p.mean(0), t-t.mean(0)
        b = float((pc*tc).sum()/max(float((tc*tc).sum()), 1e-12))
        return b, p.mean(0)-b*t.mean(0)
    b, a = fit(np.arange(len(true)))
    return dict(slope=round(b, 3), slope_95=mp.bootstrap(groups, lambda i: fit(i)[0], draws=draws),
                offset_m=round(float(np.linalg.norm(a)), 3))


def state_features(data, rows):
    """The simulator's state at t of the windows' walker (the ceilings' input): its position, range, bearing, velocity,
    speed, whether it waits and moves, and the command executed before t."""
    xy = data['xy_t'][rows]
    velocity = (data['constant_velocity'][rows, 0]-xy)/(mp.HORIZONS[0]/10)
    return np.column_stack([xy, np.linalg.norm(xy, axis=-1), np.arctan2(xy[:, 1], xy[:, 0]), velocity,
                            np.linalg.norm(velocity, axis=-1), data['waiting'][rows], data['moving'][rows],
                            data['previous'][rows]]).astype(np.float32)


def fit_classifier(x, y, device='cpu', steps=3000, seed=mp.PROBE_SEED):
    """A small MLP classifier on standardised inputs (class-balanced cross-entropy); returns a scorer."""
    import torch
    torch.manual_seed(seed)
    x, y = torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)
    mean, std = x.mean(0), x.std(0)+1e-6
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                              torch.nn.GELU(), torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    positive = y.mean().clamp(1e-3, 1-1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1-positive)/positive)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(x), (min(4096, len(x)),), device=device, generator=generator)
        loss = loss_fn(net((x[i]-mean)/std).squeeze(-1), y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval().requires_grad_(False)
    return lambda q: net((torch.as_tensor(q, dtype=torch.float32, device=device)-mean)/std).squeeze(-1).cpu().numpy()


def auc_or_none(score, label):
    return round(mp.auc(score, label), 3) if label.sum() >= 10 and (~label).sum() >= 10 else None


# -- 1. the reaction ---------------------------------------------------------------------------------------------------

def reaction_ceiling(dtrain, deval, htrain, heval, device='cpu'):
    """The ceiling of the matched windows' AUC: a classifier on the simulator's state at t, fitted on the training
    units' matched windows; {tape: {horizon: {split: AUC}}}."""
    split, out = mp.kinds(deval), {}
    rt, re = np.arange(len(dtrain['t'])), np.arange(len(deval['t']))
    for tape in mp.GIVE_WAY_CONTRASTS:
        out[tape] = {}
        for j, h in HORIZON_S.items():
            mt, me = matched(dtrain, htrain, rt, j), matched(deval, heval, re, j)
            label = lambda d, m: np.linalg.norm(d[f'cf_{tape}'][m, j]-d['cf_hold'][m, j], axis=-1) > mp.EVENT_M
            score = fit_classifier(state_features(dtrain, rt[mt]), label(dtrain, mt), device)(state_features(deval, re[me]))
            y = label(deval, me)
            out[tape][h] = {s: auc_or_none(score[split[me] == s], y[split[me] == s]) for s in SPLITS}
    return out


def reaction(dtrain, ftrain, deval, feval, heval, ceiling, *, device='cpu', updates=mp.UPDATES, draws=1000):
    """The reaction's measures for one level's cached probe features (`mover_probe.lewm_features` or
    `hjepa_features`): {readout: {tape: {horizon: {split: entry}}}}."""
    import torch
    rt, re = np.flatnonzero(ftrain['ok'].numpy()), np.flatnonzero(feval['ok'].numpy())
    reached, split = ftrain.get('horizons', list(range(len(mp.HORIZONS)))), mp.kinds(deval)
    fits = dict(ridge=lambda x, y: mp.fit_ridge(x, y, dtrain['unit'][rt]), mlp=lambda x, y: mp.fit_mlp(x, y, updates=updates))
    inputs = lambda f, rows, key, j: mp.head_inputs(f['zh'][rows].to(device), f[key][rows, j].to(device))
    out = {name: {tape: {} for tape in mp.GIVE_WAY_CONTRASTS} for name in fits}
    for name, fit in fits.items():
        for j, h in HORIZON_S.items():
            if j not in reached:
                continue
            target = torch.from_numpy(dtrain['target'][rt, j]-dtrain['xy_t'][rt]).float().to(device)
            predict = fit(inputs(ftrain, rt, 'executed', j), target)
            rows = re[matched(deval, heval, re, j)]
            with torch.no_grad():
                p = {tape: predict(inputs(feval, rows, tape, j)).cpu().numpy() for tape in ('hold', *mp.GIVE_WAY_CONTRASTS)}
            for tape in mp.GIVE_WAY_CONTRASTS:
                predicted = p[tape]-p['hold']
                true = deval[f'cf_{tape}'][rows, j]-deval['cf_hold'][rows, j]
                label = np.linalg.norm(true, axis=-1) > mp.EVENT_M
                out[name][tape][h] = {}
                for s in SPLITS:
                    k = split[rows] == s
                    if k.sum() < 20:
                        out[name][tape][h][s] = dict(windows=int(k.sum()))
                        continue
                    auc = auc_or_none(np.linalg.norm(predicted[k], axis=-1), label[k])
                    out[name][tape][h][s] = dict(
                        windows=int(k.sum()), responsive=int(label[k].sum()),
                        **slope(predicted[k], true[k], deval['unit'][rows][k], draws=draws),
                        true_x_m=round(float(true[k][label[k], 0].mean()), 3) if label[k].any() else None,
                        predicted_x_m=round(float(predicted[k][label[k], 0].mean()), 3) if label[k].any() else None,
                        predicted_x_unresponsive_m=round(float(predicted[k][~label[k], 0].mean()), 3)
                        if (~label[k]).any() else None,
                        auc=auc, auc_ceiling=ceiling[tape][h][s], **{f'auc_{key}': v for key, v in
                                                                      half_gap(auc, .5, ceiling[tape][h][s]).items()})
    return out


def reaction_passes(entry):
    """Whether one readout's retreat entries {horizon: {split: entry}} pass (see the module docstring)."""
    cells = [entry.get(h, {}).get(s, {}) for h in HORIZON_S.values() for s in SPLITS]
    return all('slope_95' in c and c['slope_95'][0] > SLOPE_FLOOR and c['offset_m'] < OFFSET_M for c in cells)


# -- 2. the advantage of the side step over holding ---------------------------------------------------------------------

def _unit_tapes(args):
    root, path, t, previous = args
    from go2nav.data import lewm_corpus_v2 as v2
    folder = Path(root)/path
    with np.load(folder/'steps.npz') as steps:
        pose = steps['pose']
    geometry = v2.FreeDriverV2(mp.unit_spec(root, json.loads((folder/'meta.json').read_text())), np.random.default_rng(0))
    return np.stack([mp.side_step_tape(geometry, pose[s], pose[s::-1, :2], p)[0] for s, p in zip(t, previous)]).astype(np.float16)


def side_tapes(root, data, out, *, workers=None):
    """The side step's tape (`mover_probe.side_step_tape`) of every window with a 12.8-s label, written once per labels
    file as OUT (windows, SIDE_FRAMES, 5, 3) float16 (zeros where there is no label); it does not depend on a model."""
    out = Path(out)
    if out.exists():
        return out
    labelled = ~np.isnan(data['clear_side'])
    jobs, rows = [], []
    for u, path in enumerate(data['units']):
        r = np.flatnonzero(labelled & (data['unit'] == u))
        if len(r):
            jobs.append((str(root), str(path), data['t'][r], data['previous'][r]))
            rows.append(r)
    partial = out.with_name(out.stem+'.tmp.npy')
    tapes = np.lib.format.open_memmap(partial, 'w+', np.float16, (len(data['t']), mp.SIDE_FRAMES, 5, 3))
    with ProcessPoolExecutor(workers or os.cpu_count(), mp_context=multiprocessing.get_context('forkserver')) as pool:
        for r, tape in zip(rows, pool.map(_unit_tapes, jobs, chunksize=4)):
            tapes[r] = tape
    tapes.flush()
    del tapes
    partial.rename(out)
    return out


def tape_summary(tape):
    """A tape (W, frames, 5, 3) as its mean command over eight equal spans (W, 24): the scene-blind readouts' input."""
    first = np.asarray(tape[:, :, 0], np.float32)
    return first.reshape(len(first), 8, -1, 3).mean(2).reshape(len(first), -1)


def side_features(model, root, data, tapes, device, *, obs_root=None, level=3):
    """Per labelled window (those with enough valid history): level 3's states at t (`now`) and one step before
    (`back`), its two-step rollout under the side step (`side`) and under hold (`hold`), the window's row (`rows`) and
    the two tapes' summaries (`side_tape`, `hold_tape`)."""
    import torch
    from go2nav.models.hjepa import upper_rollout
    from go2nav.models.lewm import action_blocks, control_stats, normalise_frames
    upper, p = model.upper[level-2], model.config.periods[level-1]
    n, steps_ahead = upper.config.context, mp.SIDE_FRAMES//p
    cmean, cstd = control_stats(model.config.level1.command_dim)
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    labelled = np.flatnonzero(~np.isnan(data['clear_side']))
    out = {k: [] for k in ('now', 'back', 'side', 'hold', 'rows', 'side_tape', 'hold_tape')}
    with torch.inference_mode():
        for u, path in enumerate(data['units']):
            rows = labelled[data['unit'][labelled] == u]
            if not len(rows):
                continue
            folder = Path(root)/str(path)
            with np.load(folder/'steps.npz') as steps:
                valid, executed = steps['block_valid'], steps['executed']
            t = data['t'][rows]
            ok = np.array([s >= (n-1)*p and valid[s-(n-1)*p:s].all() for s in t], bool)
            rows, t = rows[ok], t[ok]
            if not len(rows):
                continue
            offsets = np.arange(-(n-1), 1)*p
            needed = np.unique((t[:, None]+offsets).ravel())
            frames = np.load(folder/'frames.npy', mmap_mode='r')
            z = torch.zeros(len(frames), model.config.level1.latent, device=device)
            for s in range(0, len(needed), 512):
                chunk = needed[s:s+512]
                x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
                with autocast():
                    z[torch.from_numpy(chunk).to(device)] = model.level1.encode(
                        normalise_frames(x)[:, None], mp.unit_obs(obs_root, path, chunk, device))[:, 0].float()
            tt = torch.from_numpy(t).to(device)
            states = model.states(torch.stack([z[tt+o] for o in offsets], 1), level).float()
            blocks = action_blocks(executed, cmean, cstd).to(device)
            past = torch.stack([blocks[tt-(n-1)*p+k] for k in range((n-1)*p)], 1)
            side = np.asarray(tapes[rows], np.float32)
            hold = np.stack([mp.contrast_tape(mp.CONTRAST['hold'], previous, mp.SIDE_FRAMES)
                             for previous in data['previous'][rows]])
            for name, tape in (('side', side), ('hold', hold)):
                macros = model.macros(torch.cat((past, action_blocks(tape, cmean, cstd).to(device)), 1), level)
                with autocast():
                    out[name].append(upper_rollout(upper, states, macros.float(), steps_ahead)[:, -1].float().cpu())
                out[f'{name}_tape'].append(tape_summary(tape))
            out['now'].append(states[:, -1].cpu())
            out['back'].append(states[:, -2].cpu())
            out['rows'].append(rows)
    return {k: (np.concatenate(v) if k in ('rows', 'side_tape', 'hold_tape') else torch.cat(v)) for k, v in out.items()}


def advantage_scores(inputs_train, labels_train, groups, inputs_eval, *, device='cpu', updates=mp.UPDATES, pooled=True):
    """The predicted advantage of the side step over holding per evaluation window, from ridge and MLP readouts.
    `inputs_*`: {'side': x, 'hold': x}; `labels_train`: {'side': y, 'hold': y}. Pooled: one readout fitted on both
    tapes' inputs against each tape's own outcome. Not pooled (the critic on the state): one readout per tape."""
    import torch
    from go2nav.eval.phase_c_gates import readouts
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if pooled:
        x = torch.cat([tensor(inputs_train[k]) for k in ('side', 'hold')])
        y = np.concatenate([labels_train[k] for k in ('side', 'hold')]).astype(np.float32)
        fits = readouts(x, y, np.concatenate([groups, groups]), device, updates)
        fits = {name: dict(side=f, hold=f) for name, f in fits.items()}
    else:
        per = {k: readouts(tensor(inputs_train[k]), labels_train[k].astype(np.float32), groups, device, updates)
               for k in ('side', 'hold')}
        fits = {name: {k: per[k][name] for k in per} for name in per['side']}
    with torch.no_grad():
        return {name: (f['side'](tensor(inputs_eval['side']).to(device))-f['hold'](tensor(inputs_eval['hold']).to(device)))
                [:, 0].cpu().numpy() for name, f in fits.items()}


def advantage(ftrain, feval, dtrain, deval, htrain, heval, *, device='cpu', updates=mp.UPDATES):
    """The advantage measure (see the module docstring) from `side_features` of the training and evaluation labels."""
    import torch
    rows_t, rows_e = ftrain['rows'], feval['rows']
    clear = lambda d, rows: dict(side=d['clear_side'][rows] > .5, hold=d['clear_hold_long'][rows] > .5)
    y_t, y_e = clear(dtrain, rows_t), clear(deval, rows_e)
    label, split, groups = y_e['side'] & ~y_e['hold'], mp.kinds(deval)[rows_e], dtrain['unit'][rows_t]
    rich = lambda f, tape: torch.cat((f['now'], f[tape]-f['now'], f['now']-f['back']), -1)
    plain = lambda f, tape: torch.cat((f['now'], f['now']-f['back']), -1)
    blind = lambda f, h, rows, tape: np.column_stack([h[rows].reshape(len(rows), -1), f[f'{tape}_tape']])
    privileged = lambda f, d, rows, tape: np.column_stack([state_features(d, rows), d['side_found'][rows] > .5,
                                                           f[f'{tape}_tape']])
    families = dict(
        rollout=(lambda f, d, h, rows: {k: rich(f, k) for k in ('side', 'hold')}, True),
        critic_on_state=(lambda f, d, h, rows: {k: plain(f, k) for k in ('side', 'hold')}, False),
        scene_blind=(lambda f, d, h, rows: {k: blind(f, h, rows, k) for k in ('side', 'hold')}, True),
        ceiling=(lambda f, d, h, rows: {k: privileged(f, d, rows, k) for k in ('side', 'hold')}, True))
    report = dict(train_windows=int(len(rows_t)), eval_windows=int(len(rows_e)),
                  side_only={s: round(float(label[split == s].mean()), 3) for s in SPLITS},
                  hold_only={s: round(float((~y_e['side'] & y_e['hold'])[split == s].mean()), 3) for s in SPLITS})
    best = {}
    for name, (make, pooled) in families.items():
        scores = advantage_scores(make(ftrain, dtrain, htrain, rows_t), y_t, groups, make(feval, deval, heval, rows_e),
                                  device=device, updates=updates, pooled=pooled)
        report[name] = {r: {s: auc_or_none(v[split == s], label[split == s]) for s in SPLITS} for r, v in scores.items()}
        best[name] = {s: max((e[s] for e in report[name].values() if e[s] is not None), default=None) for s in SPLITS}
    report['present_only'] = {s: .5 for s in SPLITS}        # no tape, so no difference between the tapes
    report['best'] = best
    report['skill'] = {s: half_gap(best['rollout'][s], max(best['scene_blind'][s] or .5, .5), best['ceiling'][s])
                       for s in SPLITS}
    report['rollout_over_critic'] = {s: None if None in (best['rollout'][s], best['critic_on_state'][s])
                                     else round(best['rollout'][s]-best['critic_on_state'][s], 3) for s in SPLITS}
    report['pass'] = all(report['skill'][s]['passed'] for s in SPLITS)
    return report


# -- 3. new contacts ---------------------------------------------------------------------------------------------------

def new_contacts(model, train, val, *, device='cpu', updates=mp.UPDATES):
    """The new-contact measure (see the module docstring) from `phase_c_gates.level1_windows` of training and
    validation units (with their `history` and `tape`)."""
    import torch
    from go2nav.eval.phase_c_gates import readouts
    keep_t, keep_v = ~train['contact_t'], ~val['contact_t']
    now = lambda d: d['zh'][:, -1].float()
    with_rollout = lambda d: torch.cat((now(d), d['rolled'].float()-now(d)), -1)
    blind = lambda d: torch.from_numpy(np.column_stack([d['history'].reshape(len(d['history']), -1),
                                                        d['tape'].reshape(len(d['tape']), -1)]).astype(np.float32))
    label, split = val['contact_h'][keep_v], val['split'][keep_v]
    by_split = lambda score: {s: auc_or_none(score[split == s], label[split == s]) for s in SPLITS}
    future = train['contact_h'][keep_t].astype(np.float32)
    report = dict(train_windows=int(keep_t.sum()), val_windows=int(keep_v.sum()),
                  become_contacts={s: round(float(label[split == s].mean()), 4) for s in SPLITS},
                  model={}, present_only={}, scene_blind={})
    readers = {f'now_{k}': f for k, f in readouts(now(train), train['contact_t'].astype(np.float32), train['unit'],
                                                  device, updates).items()}
    if model.heads is not None:
        readers['head'] = lambda z: model.heads.run(model.heads.contact, z).float()
    with torch.no_grad():
        for name, read in readers.items():         # a reader of contact now, on the rollout and on the state at t
            score = lambda z: read(z.float().to(device))[:, 0].cpu().numpy()[keep_v]
            report['model'][f'{name}_on_rollout'] = by_split(score(val['rolled']))
            report['present_only'][f'{name}_on_state'] = by_split(score(now(val)))
    for family, inputs in (('model', with_rollout), ('present_only', now), ('scene_blind', blind)):
        fits = readouts(inputs(train)[torch.from_numpy(keep_t)], future, train['unit'][keep_t], device, updates)
        with torch.no_grad():
            for name, fit in fits.items():         # readouts fitted for the future label
                score = fit(inputs(val)[torch.from_numpy(keep_v)].to(device))[:, 0].cpu().numpy()
                report[family][f'future_{name}'] = by_split(score)
    best = {f: {s: max((e[s] for e in report[f].values() if e[s] is not None), default=None) for s in SPLITS}
            for f in ('model', 'present_only', 'scene_blind')}
    report['best'] = best
    report['skill'] = {s: half_gap(best['model'][s], max(best['present_only'][s] or .5, best['scene_blind'][s] or .5), 1.)
                       for s in SPLITS}
    report['pass'] = all(report['skill'][s]['passed'] for s in SPLITS)
    return report


# -- the report --------------------------------------------------------------------------------------------------------

def shares(model, root, obs_root, device, *, units_per_kind=16, frames_per_unit=12, seed=0):
    """`training.lewm.input_shares` on a fixed draw of validation frames (the review's check)."""
    import torch
    from go2nav.data import lewm_corpus as v1
    from go2nav.models.lewm import normalise_frames
    from go2nav.training.lewm import input_shares
    rng = np.random.default_rng(seed)
    folders = v1.unit_folders(Path(root), ['lewm_val'])
    frames, obs = [], []
    for kind in sorted({f.name.split('_', 1)[1] for f in folders} - {'transfer'}):
        pool = [f for f in folders if f.name.split('_', 1)[1] == kind]
        for f in rng.choice(np.array(pool, object), min(units_per_kind, len(pool)), replace=False):
            unit = np.load(f/'frames.npy', mmap_mode='r')
            if len(unit) < 60:
                continue
            t = np.sort(rng.choice(np.arange(20, len(unit)-1), frames_per_unit, replace=False))
            frames.append(np.asarray(unit[t]))
            if obs_root is not None:
                obs.append(mp.unit_obs(obs_root, f.relative_to(root), t, 'cpu')[:, 0])
    pixels = normalise_frames(torch.from_numpy(np.concatenate(frames)).to(device))
    return input_shares(model.level1, pixels, torch.cat(obs).to(device) if obs else None,
                        torch.Generator().manual_seed(seed))


def run(run_dir, gates, out, *, root, train_labels, eval_labels, train_tapes, eval_tapes, obs_root=None, heads_root=None,
        device='cuda:0', updates=mp.UPDATES):
    """Every measure for one run; OUT/measures.json. `gates`: the folder of the run's P15 gates, whose
    probe_level{1,2}/features_{train,eval}.pt are read."""
    import torch
    from go2nav.eval import phase_c_gates as g
    from go2nav.models.hjepa import load_hjepa
    run_dir, gates, out = Path(run_dir), Path(gates), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(run=str(run_dir))
    save = lambda: (out/'measures.json').write_text(json.dumps(report, indent=1, default=str)+'\n')
    dtrain, deval = dict(np.load(train_labels)), dict(np.load(eval_labels))
    histories = {}
    for name, data, labels in (('train', dtrain, train_labels), ('eval', deval, eval_labels)):
        cache = Path(labels).with_name(Path(labels).stem+'_history.npy')
        if not cache.exists():
            np.save(cache, command_history(root, data))
        histories[name] = np.load(cache)
    htrain, heval = histories['train'], histories['eval']
    model = load_hjepa(run_dir/'hjepa.pt', device)

    report['shares'] = shares(model, root, obs_root, device)
    save()
    ceiling = reaction_ceiling(dtrain, deval, htrain, heval, device)
    report['reaction'] = dict(ceiling_auc=ceiling)
    for level in (1, 2):
        ftrain, feval = (torch.load(gates/f'probe_level{level}'/f'features_{k}.pt') for k in ('train', 'eval'))
        report['reaction'][f'level{level}'] = reaction(dtrain, ftrain, deval, feval, heval, ceiling, device=device,
                                                       updates=updates)
        save()
    report['reaction']['pass'] = any(reaction_passes(readout['retreat']) for key, level in report['reaction'].items()
                                     if key.startswith('level') for readout in level.values())
    save()

    features = {}
    for name, data, tapes in (('train', dtrain, train_tapes), ('eval', deval, eval_tapes)):
        cache = out/f'side_features_{name}.pt'
        if not cache.exists():
            torch.save(side_features(model, root, data, np.load(tapes, mmap_mode='r'), device, obs_root=obs_root), cache)
        features[name] = torch.load(cache, weights_only=False)
    report['advantage'] = advantage(features['train'], features['eval'], dtrain, deval, htrain, heval, device=device,
                                    updates=updates)
    save()

    windows = {}
    for name, paths, stride in (('train', g.units(root, 'lewm_train', g.TRAIN_UNIT_EVERY), g.TRAIN_STRIDE),
                                ('val', g.units(root, 'lewm_val'), g.STRIDE)):
        cache = out/f'contact_windows_{name}.pt'
        if not cache.exists():
            torch.save(g.level1_windows(model, root, paths, device, obs_root=obs_root, heads_root=heads_root,
                                        stride=stride), cache)
        windows[name] = torch.load(cache, weights_only=False)
    report['new_contacts'] = new_contacts(model, windows['train'], windows['val'], device=device, updates=updates)
    report['measures'] = dict(reaction=report['reaction']['pass'], advantage=report['advantage']['pass'],
                              new_contacts=report['new_contacts']['pass'])
    save()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('tapes', help='the side step\'s tapes of a labels file (once; no model)')
    t.add_argument('--labels', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--root', type=Path, default=Path('/mnt/workspace_drive/go2nav_lewm_v2'))
    t.add_argument('--workers', type=int)
    r = sub.add_parser('run')
    r.add_argument('run_dir', type=Path)
    r.add_argument('--gates', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--root', type=Path, default=Path('/mnt/workspace_drive/go2nav_lewm_v2'))
    for name in ('train-labels', 'eval-labels', 'train-tapes', 'eval-tapes', 'heads-root'):
        r.add_argument(f'--{name}', required=True, type=Path)
    r.add_argument('--obs-root', type=Path)
    r.add_argument('--device', default='cuda:0')
    r.add_argument('--updates', type=int, default=mp.UPDATES)
    a = p.parse_args()
    if a.command == 'tapes':
        print(side_tapes(a.root, dict(np.load(a.labels)), a.out, workers=a.workers))
        return
    report = run(a.run_dir, a.gates, a.out, root=a.root, train_labels=a.train_labels, eval_labels=a.eval_labels,
                 train_tapes=a.train_tapes, eval_tapes=a.eval_tapes, obs_root=a.obs_root, heads_root=a.heads_root,
                 device=a.device, updates=a.updates)
    print(json.dumps(report['measures'], indent=1))


if __name__ == '__main__':
    main()
