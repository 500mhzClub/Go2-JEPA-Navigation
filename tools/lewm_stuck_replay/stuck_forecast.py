"""At a deadlocked span8w C5 decision: rerun the forecast with span8w C5 (validation against the log), idm100 C5 and
both C6 heads; score each forecast's candidates with the planner's clearance rules on a local half-plane wall fitted to
the logged clearances (residual reported)."""
import json, sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import torch
from go2nav.controllers.lewm import LeWMDirectForecaster, LeWMForecaster
from go2nav.models.lewm import load_heads, load_world_model
from go2nav.planning import selection as sel

MODELS = dict(span8w=('outputs/lewm/chain_span8w/world/lewm.pt', 'outputs/lewm/chain_span8w/fit_contexts_s2026092205/fit_decoder_fix_s2026092205.pt'),
              idm100=('outputs/lewm/chain_idm/world/lewm.pt', 'outputs/lewm/chain_idm/fit_contexts_s2026093011/fit_decoder_fix_s2026093011.pt'))
EXTRA = dict(span8w_corpus=('outputs/lewm/chain_span8w/world/lewm.pt', 'outputs/lewm/readout/chain_span8w_corpus/base_only/fit_base_only.pt'),
             span8w_proto=('outputs/lewm/chain_span8w/world/lewm.pt', 'outputs/lewm/protocol/span8w/fit.pt'))


def forecasters(device='cuda:0', models=MODELS):
    out = {}
    for name, (w, f) in models.items():
        world, (c5, c6) = load_world_model(w), load_heads(f)
        out[f'C5_{name}'] = LeWMForecaster(world, c5, device=device)
        out[f'C6_{name}'] = LeWMDirectForecaster(world, c6, device=device)
    return out


def wall(d):
    """Half-plane c - n.p (body frame) fitted to the logged segment clearances of the logged forecasts."""
    F = np.asarray(d['forecast_xy_m'])
    segs, vals = [], []
    for i, c in enumerate(d['selection']['memory_forecast_candidates']):
        pts = np.vstack([np.zeros(2), F[i]])
        for k, v in enumerate(c['segment_clearances_m']):
            if v is not None:
                segs.append((pts[k], pts[k+1])); vals.append(v)
    vals = np.asarray(vals)
    A, B = np.array([s[0] for s in segs]), np.array([s[1] for s in segs])
    def pred(x):
        n = np.array([np.cos(x[1]), np.sin(x[1])])
        return x[0]-np.maximum(A@n, B@n)
    best = min((least_squares(lambda x: pred(x)-vals, [vals.max(), th]) for th in np.linspace(-3, 3, 12)), key=lambda r: r.cost)
    return best.x, float(np.abs(pred(best.x)-vals).max())


class HalfPlane:
    def __init__(self, x):
        self.c, self.n = x[0], np.array([np.cos(x[1]), np.sin(x[1])])
    def minimum(self, a, b):
        return float(self.c-max(self.n@np.asarray(a, float), self.n@np.asarray(b, float)))


def score(d, prediction, clearance):
    """The planner's clearance check then the turn re-check (with exits) on a forecast; per action its mode."""
    base = {k: v for k, v in d['selection'].items() if k in ('candidates', 'waypoint_body_xy_m', 'alignment_scale_m', 'selection_objective')}
    base = deepcopy(base) | dict(action=d['selection']['before_memory_filter_action'])
    base['action_index'] = sel.ACTIONS.index(base['action'])
    r = sel.clearance_check(base, prediction, clearance, np.zeros(3), np.eye(3))
    r = sel.reserve_turns(r)
    return {c['action']: dict(mode=c['clearance_check_mode'], seg=[round(v, 4) for v in c['segment_clearances_m']])
            for c in r['memory_forecast_candidates']}


def context(frames_dir, hist, d):
    f, ns = d['frame'], d['measured_ns']
    j = int(np.flatnonzero(hist['image_ns'] == ns)[0])
    applied = hist['applied_command_values'][j]
    if not hist['applied_command_valid'][j].all():
        raise ValueError('incomplete applied history')
    packets = [dict(image=dict(measured_ns=ns-(3-k)*100_000_000, rgb=np.asarray(Image.open(frames_dir/f'ego_frames/{f-3+k:04d}.png'))),
                    sensor_state=dict(control=dict(applied_command=dict(values=applied)))) for k in range(4)]
    return SimpleNamespace(history=packets)


def main(run, frames_dir, out, indices):
    run, frames_dir = Path(run), Path(frames_dir)
    plan = json.loads((run/'planning.json').read_text())
    hist = dict(np.load(run/'native/policy_histories.npz'))
    fc = forecasters()
    rows = []
    for i in indices:
        d = plan[i]
        x, resid = wall(d)
        ctx = context(frames_dir, hist, d)
        row = dict(decision=i, frame=d['frame'], wall_offset_m=round(float(x[0]), 4), wall_resid_mm=round(1000*resid, 2),
                   logged=score(d, np.concatenate([np.asarray(d['forecast_xy_m']), np.zeros((6, 8, 3))], -1), HalfPlane(x)))
        for name, f in fc.items():
            pred = f.forecast(ctx, d['committed_prefix'], pulse=False)
            row[name] = score(d, pred, HalfPlane(x))
            if name == 'C5_span8w':
                row['replay_vs_log_mm'] = round(1000*float(np.abs(pred[:, :, :2]-np.asarray(d['forecast_xy_m'])).max()), 2)
            row[name+'_xy'] = np.round(pred[:, :, :2], 4).tolist()
        rows.append(row)
        clear = {k: [a for a, v in row[k].items() if a != 'hold' and v['mode'] != 'BLOCKED'] for k in ['logged', *fc]}
        print(json.dumps(dict(decision=i, resid_mm=row['wall_resid_mm'], replay_vs_log_mm=row['replay_vs_log_mm'], clear=clear)), flush=True)
    Path(out).write_text(json.dumps(rows))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(v) for v in sys.argv[4].split(',')])
