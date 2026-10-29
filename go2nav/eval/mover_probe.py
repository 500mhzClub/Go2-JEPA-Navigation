"""The mover-forecast probe (docs/dynamic/PLAN.md, P4: obstacle forecasting): does a world model forecast where a
moving obstacle will be, and how it will respond to the robot's own commands?

    python -m go2nav.eval.mover_probe labels ROOT --out LABELS.npz [--roles lewm_val] [--stride 5] [--workers N]
    python -m go2nav.eval.mover_probe probe ROOT --world-model LEWM.pt --train TRAIN.npz --eval EVAL.npz --out DIR
    python -m go2nav.eval.mover_probe probe ROOT --world-model HJEPA.pt --level 2 ...      # H-JEPA's 1.6 s level

**Windows** (`labels`): in corpus v2's units with obstacles, the decision frames t (every `stride`-th, with 4 frames of
history and `HORIZONS[-1]` frames of future) whose primary depth image has an obstacle in it (`MIN_OBSTACLE_PIXELS`),
each with its **mover**: the nearest obstacle within the primary camera's field of view whose centre the walls do not
hide. Labels, all from the simulator's records (never a model input):
- `target`: the mover's position at t+h, in the robot's body frame at t (x forward, y left; the robot's own motion
  after t is not in it);
- baselines: `persistence` (it stays where it is at t) and `constant_velocity` (it keeps its velocity over the last
  0.4 s, from its true positions);
- conditions: `waiting` (stopped for the robot at t), `polite`, `moving` (faster than 0.05 m/s at t), and per
  horizon `events` (the mover changes its motion before t+h; walkers walk straight lines at a constant speed, so
  constant velocity is exact elsewhere and a probe is scored on these).

**Counterfactual labels:** a walker (`go2nav.sim.obstacles.Walker`) reacts only to the robot's base position, every
2-ms physics step. `replay` runs each unit's walkers from their placement against the recorded poses (interpolated
between the 100-ms frames) and checks them against the recorded positions; frames up to the first one off by more than
`REPLAY_TOLERANCE_M` replay. From a replayed t, `branch` runs the walkers on against a kinematic robot under a command
tape (the contrast tapes `CONTRAST`: hold still, advance at 0.3 m/s, retreat at 0.1 m/s), executed through the
continuous safety block's rate limits from the command executed before t, translations stopping short of walls and
obstacles. The difference
between the mover's positions under advance and hold is how much it responds to the robot (`response`); 0 where it
cannot sense the robot. The same branch under the executed tape (`cf_executed`) measures the kinematic robot's error
against the recorded future.

**Probe** (`probe`): a world model's latents of each window (`lewm_features`): z of frames t-3..t, the true z at t+h
(the **encoder ceiling**: is the mover's future in what the encoder sees then?) and the rollout to t+h under the
executed commands, and under the hold and advance tapes. Per horizon, and at h = 0 (the mover where it is at t: does
the encoder see it at all?), two heads read the mover's position from [z_t, z(t+h) - z_t, z_t - z_(t-1),
z_t - z_(t-3)] (C5's decoder input): ridge (`fit_ridge`; is it linearly readable?) and an MLP (`fit_mlp`), trained on
the training mazes' windows and scored on the validation mazes' (by kind: `val` explore, mission and free units, and
`transfer`, the held-out looks). At h = 0 a head reads the mover's position (**localisation**); at h > 0 it reads the
mover's displacement from t, and its forecast is the true position at t plus that displacement: the baselines start
from the true position too, so the scores compare forecasts of the mover's motion, not localisation. Scores (`score`):
error (m) over all windows, the event windows, the responsive ones, waiting movers, non-polite ones, and event windows
with the mover within `NEAR_M` (the decisions that matter); skill on event windows (1 - error / the baseline's error,
against persistence and constant velocity; walkers choose their turns and pauses at random, so even a head trained on
the frames (`go2nav.eval.mover_ceiling`) has none); and the **velocity skill**, 1 - error / persistence's error on the
`straight` windows (moving, no event), where constant velocity is exact: 1 is a perfect velocity estimate.
**Response test** (`response`, the rollout heads): on the windows with a counterfactual, the predicted response (the
head on the advance rollout minus on the hold rollout) against the true one (`cf_advance - cf_hold`): the **response
gain** (the least-squares slope of predicted on true, over the responsive windows; 95% interval from 1,000 bootstrap
draws of units), the **spurious response** (the median predicted response where the true one is 0: the head misreading
the robot's own motion under the advance tape as the mover's), and `r2`, 1 - the summed squared error of the predicted
response over the summed squared true response, over all the windows with a counterfactual (below 0: the predicted
responses are more noise than signal).
**Give-way test** (`give_way`, the rollout heads; PLAN P4e): on the windows whose mover is near and ahead
(`give_way_windows`), whether the predicted contrast of retreating (and advancing) against holding finds the movers
that respond to it (`auc`, 95% interval from unit bootstrap draws) and how much of it it explains (`r2`); lv2's
constant-velocity prediction has no contrast (AUC 0.5, R2 0).
"""
import argparse
import copy
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

from go2nav.data import lewm_corpus as v1
from go2nav.data import lewm_corpus_v2 as v2
from go2nav.locomotion.safety import BLOCKS
from go2nav.sim import obstacles as moving

HORIZONS = (4, 8, 16, 32)          # frames of 100 ms: level 1 reaches 0.8 s, the upper levels 1.6 s and more
VELOCITY_FRAMES = 4                # constant velocity over the last 0.4 s
SUBSTEPS, PHYSICS_DT_S = 50, .002  # walker steps per frame
REPLAY_TOLERANCE_M = .01
CONTRAST = dict(hold=(0., 0., 0.), advance=(.3, 0., 0.), retreat=(-.1, 0., 0.))  # retreat: the lattice's reverse
EVENT_M = .05                      # a mover's motion changes (`events`), or it responds to the robot (`responsive`)
WALL_STAND_OFF_M = .25             # a kinematic translation stops this far from a wall (the body is 0.24 m wide)
CAMERA_X_M = .326                  # the primary camera, ahead of the body origin
HALF_FOV_RAD = math.radians(78.323/2)
MAX_VIEW_M = 5.
GIVE_WAY_M, GIVE_WAY_HALF_WIDTH_M = 1.5, .75  # give-way windows: a mover this near, ahead, this close to the path
GIVE_WAY_CONTRASTS = ('retreat', 'advance')


def body_frame(pose, xy):
    """World xy (..., 2) in the body frame of a pose (x, y, z, qx, qy, qz, qw)."""
    yaw = float(v1.planar_yaw(np.asarray(pose)[3:7]))
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, float)-np.asarray(pose)[:2]
    return np.stack([c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]], -1)


def in_view(pose, movers_xy, sight):
    """(distance from the camera, index) of every mover in the primary camera's view whose centre the walls do not
    hide, nearest first. `sight(a, b)` says whether the walls leave the segment a-b clear."""
    seen = []
    for k, m in enumerate(movers_xy):
        x, y = body_frame(pose, m)
        x -= CAMERA_X_M
        distance = math.hypot(x, y)
        if not .2 < distance < MAX_VIEW_M:
            continue
        if abs(math.atan2(y, x)) > HALF_FOV_RAD+math.asin(min(1., moving.RADIUS_M/distance)):
            continue
        if sight(np.asarray(pose[:2], float), np.asarray(m, float)):
            seen.append((distance, k))
    return sorted(seen)


def visible(pose, movers_xy, sight):
    """Index of the nearest mover in the primary camera's view whose centre the walls do not hide (or None)."""
    seen = in_view(pose, movers_xy, sight)
    return seen[0][1] if seen else None


def sight_line(clearance):
    """`sight(a, b)`: the walls (`clearance(xy)`, the distance to the nearest wall) leave the segment a-b clear."""
    def sight(a, b):
        d = b-a
        distance = float(np.linalg.norm(d))
        return all(clearance(a+s*d/distance) >= .05 for s in np.arange(.05, max(distance-.2, .05), .05))
    return sight


def walkers(spec, placement):
    neighbours = moving.passages(spec)
    return [moving.Walker(neighbours, tuple(o['start_cell']), o['speed_m_s'], np.random.default_rng(o['walk_seed']),
                          polite=o.get('polite', True)) for o in placement]


def replay(spec, placement, pose, recorded, keep=()):
    """Run the walkers against the recorded poses. Returns (error (n,) the largest distance of a replayed walker from
    its recorded position at each frame, {frame: walker states} for the frames in `keep`)."""
    state, n = walkers(spec, placement), len(pose)
    error, kept = np.zeros(n), {}
    for i in range(n):
        error[i] = max(float(np.linalg.norm(w.xy-r)) for w, r in zip(state, recorded[i]))
        if i in keep:
            kept[i] = copy.deepcopy(state)
        if i+1 < n:
            for k in range(SUBSTEPS):
                robot = pose[i, :2]+(pose[i+1, :2]-pose[i, :2])*k/SUBSTEPS
                for w in state:
                    w.step(PHYSICS_DT_S, robot)
    return error, kept


def contrast_tape(command, previous, frames):
    """The executed commands (frames, 5, 3) of a constant request after the command `previous` (3,) executed before
    it: the continuous block's bounds, and its rate limit once per 100-ms window (as `sim.session` applies it)."""
    lower, upper, delta = BLOCKS['continuous']
    out, last = np.empty((frames, v1.FRAME_STEPS, 3), np.float32), np.asarray(previous, np.float32)
    for f in range(frames):
        last = np.clip(np.clip(np.asarray(command, np.float32), lower, upper), last-delta, last+delta)
        out[f] = last
    return out


def branch(state, pose, tape, clearance, horizons=HORIZONS):
    """The walkers (copied from `state`) run on against a kinematic robot from `pose` under `tape` (frames, 5, 3) of
    executed commands. A translation that would bring the robot within `WALL_STAND_OFF_M` of a wall (`clearance(xy)`)
    or within reach of a walker is dropped. Returns the walkers' xy (len(horizons), k, 2) and the robot's
    (len(horizons), 2) after each horizon."""
    state = copy.deepcopy(state)
    xy, yaw = np.asarray(pose[:2], float).copy(), float(v1.planar_yaw(np.asarray(pose)[3:7]))
    reach = moving.RADIUS_M+moving.ROBOT_REACH_M
    out, robot, per_policy = [], [], SUBSTEPS//v1.FRAME_STEPS
    for f in range(max(horizons)):
        for p in range(v1.FRAME_STEPS):
            vx, vy, wz = (float(v) for v in tape[f, p])
            for _ in range(per_policy):
                for w in state:
                    w.step(PHYSICS_DT_S, xy)
                c, s = math.cos(yaw), math.sin(yaw)
                step = np.array([c*vx-s*vy, s*vx+c*vy])*PHYSICS_DT_S
                ahead = xy+step
                if (clearance(ahead) >= WALL_STAND_OFF_M or clearance(ahead) > clearance(xy)) and \
                        all(np.linalg.norm(ahead-w.xy) >= reach or np.linalg.norm(ahead-w.xy) > np.linalg.norm(xy-w.xy)
                            for w in state):
                    xy = ahead
                yaw += wz*PHYSICS_DT_S
        if f+1 in horizons:
            out.append([w.xy.copy() for w in state])
            robot.append(xy.copy())
    return np.asarray(out).reshape(len(robot), len(state), 2), np.asarray(robot)


def unit_spec(root, meta):
    from go2nav.eval.mission import episode_spec
    episode = 0 if meta['kind'] in ('free', 'transfer') else v1.EPISODES[meta['kind']]
    spec, _ = episode_spec(v1.SET, meta['maze'], episode, root=v1.mazes_root(root))
    return spec


def unit_labels(args):
    """The windows of one unit (see the module docstring), as a dict of arrays (empty if it has none)."""
    root, folder, stride = Path(args[0]), Path(args[1]), args[2]
    meta = json.loads((folder/'meta.json').read_text())
    placement = meta['obstacles']['placement']
    if not placement:
        return None
    with np.load(folder/'steps.npz') as steps:
        pose, executed = steps['pose'], steps['executed']
    with np.load(folder/'dynamic.npz') as dynamic:
        recorded, waiting = dynamic['obstacles_xy'].astype(float), dynamic['waiting']
        pixels = dynamic['obstacle_pixels']
    spec = unit_spec(root, meta)
    geometry = v2.FreeDriverV2(spec, np.random.default_rng(0))
    clearance = geometry.wall_clearance
    sight = sight_line(clearance)

    H, n = np.asarray(HORIZONS), len(pose)
    rows = []
    for t in range(VELOCITY_FRAMES, n-H[-1], stride):
        if pixels[t] < v2.MIN_OBSTACLE_PIXELS:
            continue
        k = visible(pose[t], recorded[t], sight)
        if k is not None:
            rows.append((t, k))
    if not rows:
        return None
    error, states = replay(spec, placement, pose, recorded, keep={t for t, _ in rows})
    replays = np.cumsum(error > REPLAY_TOLERANCE_M) == 0          # every frame up to here replayed
    out = {key: [] for key in ('t', 'mover', 'xy_t', 'target', 'persistence', 'constant_velocity', 'waiting',
                               'moving', 'previous', 'replayed', 'cf_executed', *(f'cf_{c}' for c in CONTRAST))}
    for t, k in rows:
        p = pose[t]
        velocity = (recorded[t, k]-recorded[t-VELOCITY_FRAMES, k])/(VELOCITY_FRAMES*.1)
        out['t'].append(t)
        out['mover'].append(k)
        out['xy_t'].append(body_frame(p, recorded[t, k]))
        out['target'].append(body_frame(p, recorded[t+H, k]))
        out['persistence'].append(np.repeat(body_frame(p, recorded[t, k])[None], len(H), 0))
        out['constant_velocity'].append(body_frame(p, recorded[t, k]+velocity*H[:, None]*.1))
        out['waiting'].append(bool(waiting[t, k]))
        out['moving'].append(bool(np.linalg.norm(recorded[t, k]-recorded[t-1, k]) > .005))
        previous = executed[t-1, -1]
        out['previous'].append(previous)
        out['replayed'].append(bool(replays[t+H[-1]]))
        if replays[t]:
            tapes = dict(executed=executed[t:t+H[-1]],
                         **{name: contrast_tape(c, previous, H[-1]) for name, c in CONTRAST.items()})
            for name, tape in tapes.items():
                out[f'cf_{name}'].append(body_frame(p, branch(states[t], p, tape, clearance)[0][:, k]))
        else:
            for name in ('executed', *CONTRAST):
                out[f'cf_{name}'].append(np.full((len(H), 2), np.nan))
    out = {key: np.asarray(v) for key, v in out.items()}
    out.update(polite=np.full(len(rows), placement[0].get('polite', True)),
               replay_error=np.asarray([error[t] for t, _ in rows]))
    return out


def labels(root, out, *, roles=('lewm_val',), stride=5, workers=None):
    """Label every unit of `roles` (see the module docstring) into one npz; `unit` indexes `units` (their paths)."""
    root = Path(root)
    folders = v1.unit_folders(root, list(roles))
    jobs = [(str(root), str(f), stride) for f in folders]
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2)-2),
                             mp_context=multiprocessing.get_context('forkserver')) as pool:
        results = list(pool.map(unit_labels, jobs, chunksize=1))
    units, parts = [], []
    for folder, result in zip(folders, results):
        if result is not None:
            result['unit'] = np.full(len(result['t']), len(units))
            units.append(str(folder.relative_to(root)))
            parts.append(result)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, units=np.asarray(units), horizons=np.asarray(HORIZONS), **data)
    return summary(data)


def events(data):
    """(windows, horizons) whether the mover changes its motion before t+h: constant velocity misses it by more
    than `EVENT_M` (it turns at a cell, pauses, stops for the robot or sets off). Elsewhere constant velocity is exact
    (walkers walk straight lines at a constant speed), so a probe is scored on these."""
    return np.linalg.norm(data['constant_velocity']-data['target'], axis=-1) > EVENT_M


def responsive(data):
    """(windows, horizons) whether the mover responds to the robot: advance and hold leave it more than `EVENT_M`
    apart at t+h (False where there is no counterfactual)."""
    return np.nan_to_num(np.linalg.norm(data['cf_advance']-data['cf_hold'], axis=-1)) > EVENT_M


def summary(data):
    """Window counts, the replay and the kinematic robot's fidelity, and the baselines' errors (m) per horizon: median
    over all windows, mean over all, and median over the event windows (`events`)."""
    error = lambda a: np.linalg.norm(a-data['target'], axis=-1)
    cf = ~np.isnan(data['cf_hold'][:, 0, 0])
    event, respond = events(data), responsive(data)
    by = lambda e, mask: [round(float(np.nanmedian(e[mask[:, j], j])), 3) if mask[:, j].any() else None
                          for j in range(len(HORIZONS))]
    everywhere = np.ones_like(event)
    report = dict(windows=int(len(data['t'])), units=int(len(np.unique(data['unit']))),
                  waiting=int(data['waiting'].sum()), moving=int(data['moving'].sum()),
                  impolite=int((~data['polite']).sum()), counterfactual=int(cf.sum()),
                  replayed_to_end=int(data['replayed'].sum()), horizons_s=[h/10 for h in HORIZONS],
                  event_share=np.round(event.mean(0), 3).tolist(),
                  responsive_share=np.round(respond.mean(0), 3).tolist(),
                  responsive_impolite_share=np.round(respond[~data['polite']].mean(0), 3).tolist()
                  if (~data['polite']).any() else None)
    for name in ('persistence', 'constant_velocity'):
        e = error(data[name])
        report[name] = dict(median_m=by(e, everywhere), mean_m=np.round(e.mean(0), 3).tolist(),
                            event_median_m=by(e, event), responsive_median_m=by(e, respond))
    e = np.where(data['replayed'][:, None], error(data['cf_executed']), np.nan)
    report['cf_executed'] = dict(median_m=by(e, everywhere), event_median_m=by(e, event),
                                 responsive_median_m=by(e, respond))
    return report


# -- probe ------------------------------------------------------------------------------------------------------------

PROBE_SEED, UPDATES, BATCH, LR, WEIGHT_DECAY, GRAD_CLIP = 2026102701, 10000, 512, 3e-4, 1e-4, 1.
RIDGE_LAMBDAS = (1e-1, 1., 1e1, 1e2, 1e3, 1e4)
NEAR_M = 2.
TAPES = ('executed', *CONTRAST)


def usable(steps, t):
    """Windows whose action blocks are valid from t-3 to the last horizon (the rollouts need them)."""
    valid = steps['block_valid']
    return np.array([valid[s-3:s+HORIZONS[-1]].all() for s in t], bool)


def lewm_features(model, root, data, device):
    """Per window of `data` (labels): `zh` (W, history, D) z of frames t-3..t, `true` (W, len(HORIZONS), D) and, per
    tape of `TAPES`, the rollout to each horizon (W, len(HORIZONS), D), as float16; and `ok` (W,) the usable windows
    (`usable`; the others are zeros). With a grid (P4b), D is the whole state's: [CLS] and the grid tokens,
    flattened."""
    import torch
    from go2nav.models.lewm import action_blocks, control_stats, normalise_frames
    root, H, n = Path(root), np.asarray(HORIZONS), model.config.history
    cmean, cstd = control_stats(model.config.command_dim)
    state = (model.config.tokens, model.config.latent) if model.config.grid else (model.config.latent,)
    D, W = math.prod(state), len(data['t'])
    out = dict(zh=torch.zeros(W, n, D, dtype=torch.float16), true=torch.zeros(W, len(H), D, dtype=torch.float16),
               **{tape: torch.zeros(W, len(H), D, dtype=torch.float16) for tape in TAPES},
               ok=torch.zeros(W, dtype=bool))
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    with torch.inference_mode():
        for u, path in enumerate(data['units']):
            rows = np.flatnonzero(data['unit'] == u)
            if not len(rows):
                continue
            folder = root/str(path)
            steps = dict(np.load(folder/'steps.npz'))
            t = data['t'][rows]
            ok = usable(steps, t)
            rows, t = rows[ok], t[ok]
            if not len(rows):
                continue
            needed = np.unique(np.concatenate([t[:, None]+np.arange(1-n, 1), t[:, None]+H], 1))
            frames = np.load(folder/'frames.npy', mmap_mode='r')
            z = torch.zeros(len(frames), *state, device=device)
            for s in range(0, len(needed), 512):
                chunk = needed[s:s+512]
                x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
                with autocast():
                    z[torch.from_numpy(chunk).to(device)] = model.encode(normalise_frames(x)[:, None])[:, 0].float()
            tt = torch.from_numpy(t).to(device)
            zh = torch.stack([z[tt+k] for k in range(1-n, 1)], 1)
            blocks = action_blocks(steps['executed'], cmean, cstd).to(device)
            past = torch.stack([blocks[tt-n+1+k] for k in range(n-1)], 1)
            tapes = dict(executed=torch.stack([blocks[tt+k] for k in range(H[-1])], 1))
            for name, command in CONTRAST.items():
                tape = np.stack([contrast_tape(command, previous, H[-1]) for previous in data['previous'][rows]])
                tapes[name] = action_blocks(tape, cmean, cstd).to(device)
            out['zh'][rows] = zh.flatten(2).half().cpu()
            out['true'][rows] = torch.stack([z[tt+h] for h in H], 1).flatten(2).half().cpu()
            for name, tape in tapes.items():
                with autocast():
                    rolled = model.rollout(zh, torch.cat((past, tape), 1), int(H[-1])).float()
                out[name][rows] = rolled[:, H-1].flatten(2).half().cpu()
            out['ok'][rows] = True
    return out


def hjepa_features(model, root, data, device, level=2):
    """As `lewm_features`, for an H-JEPA's upper `level` (period p frames, `context` states): `zh` (W, context, D) its
    states at t-(context-1)p..t, and `true` and each tape's rollout (the level's own predictor, one step per p frames,
    each transition's macro-action from its p command blocks) at the horizons that are multiples of p (zeros at the
    others). `ok`: windows with t >= (context-1)p and valid command blocks over the whole span."""
    import torch
    from go2nav.models.lewm import action_blocks, control_stats, normalise_frames
    root, H = Path(root), np.asarray(HORIZONS)
    upper, p = model.upper[level-2], model.config.periods[level-1]
    n, D, W = upper.config.context, upper.config.latent, len(data['t'])
    reached = [j for j, h in enumerate(H) if h % p == 0]
    steps = int(H[reached[-1]]//p)
    cmean, cstd = control_stats(model.config.level1.command_dim)
    out = dict(zh=torch.zeros(W, n, D, dtype=torch.float16), true=torch.zeros(W, len(H), D, dtype=torch.float16),
               **{tape: torch.zeros(W, len(H), D, dtype=torch.float16) for tape in TAPES},
               ok=torch.zeros(W, dtype=bool))
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    with torch.inference_mode():
        for u, path in enumerate(data['units']):
            rows = np.flatnonzero(data['unit'] == u)
            if not len(rows):
                continue
            folder = root/str(path)
            steps_npz = dict(np.load(folder/'steps.npz'))
            t = data['t'][rows]
            valid = steps_npz['block_valid']
            ok = np.array([s >= (n-1)*p and valid[s-(n-1)*p:s+steps*p].all() for s in t], bool)
            rows, t = rows[ok], t[ok]
            if not len(rows):
                continue
            offsets = np.arange(-(n-1), steps+1)*p
            needed = np.unique((t[:, None]+offsets).ravel())
            frames = np.load(folder/'frames.npy', mmap_mode='r')
            z = torch.zeros(len(frames), model.config.level1.latent, device=device)
            for s in range(0, len(needed), 512):
                chunk = needed[s:s+512]
                x = torch.from_numpy(np.ascontiguousarray(frames[chunk])).to(device)
                with autocast():
                    encoded = model.level1.encode(normalise_frames(x)[:, None])[:, 0]
                z[torch.from_numpy(chunk).to(device)] = encoded.float()
            tt = torch.from_numpy(t).to(device)
            states = model.states(torch.stack([z[tt+o] for o in offsets], 1), level).float()
            blocks = action_blocks(steps_npz['executed'], cmean, cstd).to(device)
            past = torch.stack([blocks[tt-(n-1)*p+k] for k in range((n-1)*p)], 1)
            tapes = dict(executed=torch.stack([blocks[tt+k] for k in range(steps*p)], 1))
            for name, command in CONTRAST.items():
                tape = np.stack([contrast_tape(command, previous, steps*p) for previous in data['previous'][rows]])
                tapes[name] = action_blocks(tape, cmean, cstd).to(device)
            out['zh'][rows] = states[:, :n].half().cpu()
            for j in reached:
                out['true'][rows, j] = states[:, n-1+H[j]//p].half().cpu()
            for name, tape in tapes.items():
                macros = model.macros(torch.cat((past, tape), 1), level).float()
                context, pending, rolled = states[:, :n], macros[:, :n-1], []
                for k in range(steps):
                    m = torch.cat((pending, macros[:, n-1+k:n+k]), 1)
                    with autocast():
                        nxt = upper.predict(context, m)[:, -1].float()
                    rolled.append(nxt)
                    context, pending = torch.cat((context[:, 1:], nxt[:, None]), 1), m[:, 1:]
                for j in reached:
                    out[name][rows, j] = rolled[H[j]//p-1].half().cpu()
            out['ok'][rows] = True
    out['horizons'] = reached
    return out


def head_inputs(zh, future):
    """C5's decoder input [z_t, future - z_t, z_t - z_(t-1), z_t - z_(t-3)] (float32)."""
    import torch
    zh, future = zh.float(), future.float()
    current = zh[:, -1]
    return torch.cat((current, future-current, current-zh[:, -2], current-zh[:, 0]), -1)


def fit_ridge(x, y, groups, seed=PROBE_SEED):
    """Ridge on standardised inputs, its weight chosen on a tenth of the groups (units) held out; returns a
    predictor."""
    import torch
    x, y = x.double(), y.double()
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    ymean = y.mean(0)
    xs = (x-mean)/scale
    held = np.random.default_rng(seed).permutation(np.unique(groups))[:max(1, len(np.unique(groups))//10)]
    fold = torch.from_numpy(np.isin(groups, held)).to(x.device)

    def solve(rows, lam):
        a = xs[rows]
        return torch.linalg.solve(a.T@a+lam*torch.eye(a.shape[1], dtype=a.dtype, device=a.device), a.T@(y[rows]-ymean))
    errors = [float(((xs[fold]@solve(~fold, lam)+ymean-y[fold]).norm(dim=-1)).mean()) for lam in RIDGE_LAMBDAS]
    w = solve(torch.ones_like(fold), RIDGE_LAMBDAS[int(np.argmin(errors))])
    return lambda q: (((q.double()-mean)/scale)@w+ymean).float()


def fit_mlp(x, y, seed=PROBE_SEED, updates=UPDATES):
    """Two 1,024-unit GELU layers on normalised targets (AdamW, cosine, batch 512); returns a predictor."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    ymean, yscale = y.mean(0), y.std(0).clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(x.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                        nn.Linear(1024, y.shape[1])).to(x.device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates)
    g = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(updates):
        i = torch.randint(len(x), (BATCH,), device=x.device, generator=g)
        loss = ((net(x[i])-(y[i]-ymean)/yscale)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
    net.eval().requires_grad_(False)
    return lambda q: net(q)*yscale+ymean


def bootstrap(groups, statistic, draws=1000, seed=PROBE_SEED):
    """95% interval of `statistic(rows)` over draws of the groups (units) with replacement."""
    rng, units = np.random.default_rng(seed), np.unique(groups)
    members = {u: np.flatnonzero(groups == u) for u in units}
    stats = [statistic(np.concatenate([members[u] for u in rng.choice(units, len(units))])) for _ in range(draws)]
    return [round(float(np.quantile(stats, q)), 3) for q in (.025, .975)]


def gain(predicted, true):
    """Least-squares slope of the predicted response on the true one (n, 2)."""
    return float((predicted*true).sum()/max((true*true).sum(), 1e-12))


def score(data, rows, predicted, j):
    """The errors and skills (see the module docstring) of `predicted` (n, 2) at horizon index j (None: h = 0)."""
    target = data['xy_t'][rows] if j is None else data['target'][rows, j]
    error = np.linalg.norm(predicted-target, axis=-1)
    out = dict(median_m=round(float(np.median(error)), 3), mean_m=round(float(error.mean()), 3))
    if j is None:
        return out
    event, respond = events(data)[rows, j], responsive(data)[rows, j]
    near = np.linalg.norm(data['xy_t'][rows], axis=-1) < NEAR_M
    straight = ~event & data['moving'][rows]
    masks = dict(event=event, responsive=respond, waiting=data['waiting'][rows],
                 impolite_event=event & ~data['polite'][rows], near_event=event & near, straight=straight)
    for name, mask in masks.items():
        out[f'{name}_median_m'] = round(float(np.median(error[mask])), 3) if mask.any() else None
    for name in ('persistence', 'constant_velocity'):
        base = np.linalg.norm(data[name][rows, j]-target, axis=-1)
        out[f'event_skill_vs_{name}'] = round(1-float(np.median(error[event]))/float(np.median(base[event])), 3) \
            if event.any() else None
    base = np.linalg.norm(data['persistence'][rows, j]-target, axis=-1)
    out['velocity_skill'] = round(1-float(np.median(error[straight]))/float(np.median(base[straight])), 3) \
        if straight.any() else None
    return out


def response(data, rows, hold, advance, j):
    """The response test (see the module docstring) at horizon index j, from the rollout head's predictions under the
    hold and advance tapes (n, 2)."""
    true = data['cf_advance'][rows, j]-data['cf_hold'][rows, j]
    predicted = advance-hold
    cf = ~np.isnan(true[:, 0])
    respond, still = responsive(data)[rows, j] & cf, cf & (np.nan_to_num(np.linalg.norm(true, axis=-1)) == 0)
    groups = data['unit'][rows]
    out = dict(responsive=int(respond.sum()), unresponsive=int(still.sum()))
    if cf.any():
        out['r2'] = round(1-float(((predicted[cf]-true[cf])**2).sum())/max(float((true[cf]**2).sum()), 1e-12), 3)
    if respond.sum() >= 10:
        r = np.flatnonzero(respond)
        out.update(gain=round(gain(predicted[r], true[r]), 3),
                   gain_95=bootstrap(groups[r], lambda i: gain(predicted[r][i], true[r][i])),
                   true_median_m=round(float(np.median(np.linalg.norm(true[r], axis=-1))), 3),
                   predicted_median_m=round(float(np.median(np.linalg.norm(predicted[r], axis=-1))), 3))
    if still.any():
        out['spurious_median_m'] = round(float(np.median(np.linalg.norm(predicted[still], axis=-1))), 3)
    return out


def give_way_windows(data, rows):
    """Whether each window's mover is near and ahead at t (`GIVE_WAY_M`, within `GIVE_WAY_HALF_WIDTH_M` of the path):
    the situations in which the robot chooses between holding, backing off and going on."""
    xy = data['xy_t'][rows]
    return (xy[:, 0] > 0) & (np.abs(xy[:, 1]) < GIVE_WAY_HALF_WIDTH_M) & (np.linalg.norm(xy, axis=-1) < GIVE_WAY_M)


def auc(score, label):
    """The probability that a positive's score exceeds a negative's (ties count half); 0.5 without both classes."""
    positives, negatives = score[label], score[~label]
    if not len(positives) or not len(negatives):
        return .5
    order = np.argsort(np.concatenate([positives, negatives]), kind='stable')
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order)+1)
    values = np.concatenate([positives, negatives])
    for v in np.unique(values):                        # average ranks over ties
        tied = values == v
        ranks[tied] = ranks[tied].mean()
    return float((ranks[:len(positives)].sum()-len(positives)*(len(positives)+1)/2)/(len(positives)*len(negatives)))


def give_way(data, rows, predicted, j):
    """The give-way test (see the module docstring) at horizon index j, from a rollout head's predictions per tape
    {tape: (n, 2)}: on the give-way windows with a counterfactual, for each of `GIVE_WAY_CONTRASTS` against hold, the
    true contrast (the mover's position under that tape minus under hold) against the predicted one: `auc` of the
    predicted contrast's size for whether the true one exceeds `EVENT_M` (95% interval over units), and `r2`. Constant
    velocity, lv2's prediction, ignores the robot: its contrast is 0, so its AUC is 0.5 and its R2 0."""
    near = give_way_windows(data, rows) & ~np.isnan(data['cf_hold'][rows, j, 0])
    out, groups = dict(windows=int(near.sum())), data['unit'][rows][near]
    for c in GIVE_WAY_CONTRASTS:
        true = (data[f'cf_{c}'][rows, j]-data['cf_hold'][rows, j])[near]
        guess = (predicted[c]-predicted['hold'])[near]
        label, score = np.linalg.norm(true, axis=-1) > EVENT_M, np.linalg.norm(guess, axis=-1)
        entry = dict(responsive=int(label.sum()))
        if near.any():
            entry['r2'] = round(1-float(((guess-true)**2).sum())/max(float((true**2).sum()), 1e-12), 3)
        if label.sum() >= 10 and (~label).sum() >= 10:
            entry['auc'] = round(auc(score, label), 3)
            entry['auc_95'] = bootstrap(groups, lambda i: auc(score[i], label[i]))
        out[c] = entry
    return out


def kinds(data):
    """Each window's split: `transfer` for transfer units, else `val`."""
    kind = np.array([str(u).rsplit('_', 1)[-1] for u in data['units']])[data['unit']]
    return np.where(kind == 'transfer', 'transfer', 'val')


def probe(root, world_model, train, evaluation, out, *, updates=UPDATES, device='cuda:0', level=1):
    """Fit the heads on `train`'s windows and score them on `evaluation`'s (see the module docstring); results.json
    in `out`, with the latents cached per labels file. `level` > 1: `world_model` is an H-JEPA checkpoint and the
    heads read that level's states (`hjepa_features`), at the horizons it reaches."""
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if level == 1:
        from go2nav.models.lewm import load_world_model
        model, features = load_world_model(world_model, device), lewm_features
    else:
        from go2nav.models.hjepa import load_hjepa
        model = load_hjepa(world_model, device)
        features = lambda *a: hjepa_features(*a, level=level)
    sets = {}
    for name, path in (('train', train), ('eval', evaluation)):
        data = dict(np.load(path))
        cache = out/f'features_{name}.pt'
        if not cache.exists():
            torch.save(features(model, root, data, device), cache)
        sets[name] = (data, torch.load(cache))
    (dtrain, ftrain), (deval, feval) = sets['train'], sets['eval']
    rtrain, reval = np.flatnonzero(ftrain['ok'].numpy()), np.flatnonzero(feval['ok'].numpy())
    split = kinds(deval)[reval]
    gpu = lambda f, rows, key, j: head_inputs(f['zh'][rows].to(device), (f['zh'][rows, -1] if j is None
                                                                         else f[key][rows, j]).to(device))
    reached = ftrain.get('horizons', list(range(len(HORIZONS))))
    results = dict(world_model=str(world_model), level=level, train_windows=int(len(rtrain)),
                   eval_windows=int(len(reval)), horizons_s=[0.]+[HORIZONS[j]/10 for j in reached], heads={})
    for head, fit in (('ridge', lambda x, y: fit_ridge(x, y, dtrain['unit'][rtrain])),
                      ('mlp', lambda x, y: fit_mlp(x, y, updates=updates))):
        for source in ('true', 'rollout'):
            rows_out = {s: [] for s in ('val', 'transfer')}
            for j in (None, *reached):
                if j is None and source == 'rollout':
                    continue
                key = 'executed' if source == 'rollout' else 'true'
                y = dtrain['xy_t'][rtrain] if j is None else dtrain['target'][rtrain, j]-dtrain['xy_t'][rtrain]
                predict = fit(gpu(ftrain, rtrain, key, j), torch.from_numpy(y).float().to(device))
                start = 0. if j is None else deval['xy_t'][reval]          # displacements start from the true xy
                with torch.no_grad():
                    p = {tape: start+predict(gpu(feval, reval, tape, j)).cpu().numpy()
                         for tape in ((key,) if source == 'true' or j is None else TAPES)}
                for s in rows_out:
                    m = split == s
                    entry = dict(horizon_s=0. if j is None else HORIZONS[j]/10,
                                 **score(deval, reval[m], p[key][m], j))
                    if source == 'rollout' and j is not None:
                        entry['response'] = response(deval, reval[m], p['hold'][m], p['advance'][m], j)
                        entry['give_way'] = give_way(deval, reval[m], {t: v[m] for t, v in p.items()}, j)
                    rows_out[s].append(entry)
            results['heads'][f'{head}_{source}'] = rows_out
            (out/'results.json').write_text(json.dumps(results, indent=1)+'\n')
    windows = len(deval['t'])
    subset = lambda rows: {k: v[rows] if isinstance(v, np.ndarray) and len(v) == windows else v
                           for k, v in deval.items()}
    results['baselines'] = {s: summary(subset(reval[split == s])) for s in ('val', 'transfer')}
    (out/'results.json').write_text(json.dumps(results, indent=1)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('labels')
    p.add_argument('root')
    p.add_argument('--out', required=True)
    p.add_argument('--roles', nargs='+', default=['lewm_val'])
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--workers', type=int)
    p = sub.add_parser('probe')
    p.add_argument('root')
    p.add_argument('--world-model', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--eval', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--updates', type=int, default=UPDATES)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--level', type=int, default=1, help='an H-JEPA checkpoint\'s upper level (2: 1.6 s)')
    args = parser.parse_args()
    if args.command == 'probe':
        probe(args.root, args.world_model, args.train, args.eval, args.out, updates=args.updates, device=args.device,
              level=args.level)
    if args.command == 'labels':
        print(json.dumps(labels(args.root, args.out, roles=args.roles, stride=args.stride, workers=args.workers),
                         indent=1))


if __name__ == '__main__':
    main()
