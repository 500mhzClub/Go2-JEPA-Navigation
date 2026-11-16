"""Two diagnostics of why the models do not predict the walkers' reaction (8-9 November), from the gate runs' cached
probe features (no re-encoding).

1. Leakage: in the give-way windows, the walker readout's predicted contrast (tape minus hold) regressed on the robot's
   own displacement under the tape (minus hold's) and on the true contrast, pooled over the retreat and advance tapes.
   The labels are in the body frame at t, so a correct model has leak 0 and response 1; leak -1 is a walker reported
   relative to the moving robot. Also readout-free: the AUC of the latent contrast's size |z_tape - z_hold| for whether
   the walker responds.
2. Budget: the rollout's latent error (executed rollout against the true future z) per window type, relative to the
   latent's change over the horizon, and each type's share of windows and of the summed error.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b')
from go2nav.data import lewm_corpus as v1
from go2nav.eval import mover_probe as mp

C = '/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/phase_c'
torch.set_num_threads(8)
train = dict(np.load(f'{C}/give_way_labels_train.npz'))
evaluation = dict(np.load(f'{C}/give_way_labels_val.npz'))
SUBSTEP_S = .1/v1.FRAME_STEPS


def robot_dx(data, rows, tape, j):
    """The kinematic robot's forward displacement under the tape minus under hold (walls ignored), metres."""
    frames = mp.HORIZONS[j]
    out = []
    for previous in data['previous'][rows]:
        a = mp.contrast_tape(mp.CONTRAST[tape], previous, frames)[..., 0].sum()*SUBSTEP_S
        b = mp.contrast_tape(mp.CONTRAST['hold'], previous, frames)[..., 0].sum()*SUBSTEP_S
        out.append(a-b)
    return np.asarray(out)


def leakage(f_train, f_eval, j):
    rt = np.flatnonzero(f_train['ok'].numpy())
    re = np.flatnonzero(f_eval['ok'].numpy())
    near = mp.give_way_windows(evaluation, re) & ~np.isnan(evaluation['cf_hold'][re, j, 0]) & (mp.kinds(evaluation)[re] == 'val')
    re = re[near]
    x = mp.head_inputs(f_train['zh'][rt], f_train['executed'][rt, j])
    y = torch.from_numpy(train['target'][rt, j]-train['xy_t'][rt]).float()
    predict = mp.fit_ridge(x, y, train['unit'][rt])
    readout = lambda tape: predict(mp.head_inputs(f_eval["zh"][re], f_eval[tape][re, j])).detach().numpy()
    hold = readout('hold')
    rows, out = [], {}
    for tape in ('retreat', 'advance'):
        pred = readout(tape)-hold
        true = evaluation[f'cf_{tape}'][re, j]-evaluation['cf_hold'][re, j]
        dx = robot_dx(evaluation, re, tape, j)
        rows.append((pred, true, dx))
        label = np.linalg.norm(true, axis=-1) > mp.EVENT_M
        latent = (f_eval[tape][re, j]-f_eval['hold'][re, j]).float().norm(dim=-1).numpy()
        out[tape] = dict(windows=int(len(re)), responsive=round(float(label.mean()), 3), robot_dx_m=round(float(dx.mean()), 3),
                         predicted_x_m_unresponsive=round(float(pred[~label, 0].mean()), 3),
                         predicted_x_m_responsive=round(float(pred[label, 0].mean()), 3),
                         true_x_m_responsive=round(float(true[label, 0].mean()), 3),
                         latent_contrast_auc=round(mp.auc(latent, label), 3),
                         readout_contrast_auc=round(mp.auc(np.linalg.norm(pred, axis=-1), label), 3))
    pred = np.concatenate([r[0] for r in rows]); true = np.concatenate([r[1] for r in rows])
    dx = np.concatenate([r[2] for r in rows])
    design = np.column_stack([dx, true[:, 0], np.ones(len(dx))])
    coef, *_ = np.linalg.lstsq(design, pred[:, 0], rcond=None)
    fitted = design@coef
    r2 = 1-((pred[:, 0]-fitted)**2).sum()/((pred[:, 0]-pred[:, 0].mean())**2).sum()
    out['regression_x'] = dict(leak=round(float(coef[0]), 3), response=round(float(coef[1]), 3), r2=round(float(r2), 3))
    return out


def budget(f_eval, j):
    re = np.flatnonzero(f_eval['ok'].numpy())
    re = re[mp.kinds(evaluation)[re] == 'val']
    err = (f_eval['executed'][re, j]-f_eval['true'][re, j]).float().pow(2).sum(-1).numpy()
    change = (f_eval['true'][re, j]-f_eval['zh'][re, -1]).float().pow(2).sum(-1).numpy()
    near = mp.give_way_windows(evaluation, re)
    cf = ~np.isnan(evaluation['cf_hold'][re, j, 0])
    respond = np.zeros(len(re), bool)
    for tape in ('retreat', 'advance'):
        respond |= np.nan_to_num(np.linalg.norm(evaluation[f'cf_{tape}'][re, j]-evaluation['cf_hold'][re, j], axis=-1)) > mp.EVENT_M
    event = mp.events(evaluation)[re, j]
    types = {'near, responds to the robot': near & cf & respond, 'near, does not respond': near & ~(cf & respond),
             'not near, walker changes motion': ~near & event, 'not near, walker steady': ~near & ~event}
    return {k: dict(window_share=round(float(m.mean()), 3), error_share=round(float(err[m].sum()/err.sum()), 3),
                    relative_error=round(float(err[m].mean()/change[m].mean()), 3)) for k, m in types.items()}


report = {}
NAMES = sys.argv[1:] or ['gates_v2', 'gates_control']
for name in NAMES:
    for level in (1, 2):
        folder = f'{C}/{name}/probe_level{level}'
        if not __import__('os').path.exists(f'{folder}/features_eval.pt'):
            continue
        f_train, f_eval = torch.load(f'{folder}/features_train.pt'), torch.load(f'{folder}/features_eval.pt')
        reached = f_train.get('horizons', list(range(len(mp.HORIZONS))))
        for j in (2, 3):
            if j not in reached:
                continue
            key = f'{name}_level{level}_{mp.HORIZONS[j]/10}s'
            report[key] = dict(leakage=leakage(f_train, f_eval, j), budget=budget(f_eval, j))
            print(key, json.dumps(report[key]), flush=True)
json.dump(report, open(f'{C}/leakage_and_budget' + ('_' + '_'.join(sys.argv[1:]) if sys.argv[1:] else '') + '.json', 'w'), indent=1)
