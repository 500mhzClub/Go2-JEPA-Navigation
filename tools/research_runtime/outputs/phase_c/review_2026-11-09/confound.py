"""Can the give-way gate's AUC be reached from the robot's own command history alone (no scene at all)?"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b')
from go2nav.eval import mover_probe as mp

C = Path('/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/phase_c')
ROOT = Path('/mnt/workspace_drive/go2nav_lewm_v2')
ev = dict(np.load(C/'give_way_labels_val.npz', allow_pickle=True))
print({k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in ev.items()})
rows = np.arange(len(ev['t']))
split = mp.kinds(ev)
# the robot's applied-command history at t, per window (what the proprio token holds)
hist = np.zeros((len(rows), 15, 3), np.float32)
for u, path in enumerate(ev['units']):
    r = np.flatnonzero(ev['unit'] == u)
    if not len(r):
        continue
    with np.load(ROOT/str(path)/'steps.npz') as s:
        hist[r] = s['applied_history'][ev['t'][r]]
speed15 = np.linalg.norm(hist[:, :, :2], axis=-1).mean(1)        # mean planar command speed over the last 1.5 s
yaw15 = np.abs(hist[:, :, 2]).mean(1)
last = np.linalg.norm(hist[:, -1, :2], axis=-1)
still = -(speed15+0.3*yaw15)                                      # higher = the robot has been standing still
prev = -np.linalg.norm(ev['previous'][:, :2], axis=-1) if ev['previous'].ndim == 2 else None
out = {}
for j, name in ((2, '1.6s'), (3, '3.2s')):
    near = mp.give_way_windows(ev, rows) & ~np.isnan(ev['cf_hold'][rows, j, 0])
    for s in ('val', 'transfer'):
        m = near & (split == s)
        true = ev['cf_retreat'][m, j]-ev['cf_hold'][m, j]
        label = np.linalg.norm(true, axis=-1) > mp.EVENT_M
        dist = np.linalg.norm(ev['xy_t'][m], axis=-1)
        entry = dict(windows=int(m.sum()), responsive=round(float(label.mean()), 3),
                     auc_robot_was_still_1p5s=round(mp.auc(still[m], label), 3),
                     auc_last_command_small=round(mp.auc(-last[m], label), 3),
                     auc_walker_near=round(mp.auc(-dist, label), 3),
                     auc_waiting_flag=round(mp.auc(ev['waiting'][m].astype(float), label), 3))
        # a tiny logistic fit on command history only, units held out
        out[f'{name}_{s}'] = entry
        print(name, s, json.dumps(entry), flush=True)

# logistic regression from the 45 command-history values alone, fitted on the TRAIN labels, scored on val/transfer
tr = dict(np.load(C/'give_way_labels_train.npz', allow_pickle=True))
rt = np.arange(len(tr['t']))
hist_t = np.zeros((len(rt), 15, 3), np.float32)
for u, path in enumerate(tr['units']):
    r = np.flatnonzero(tr['unit'] == u)
    if not len(r):
        continue
    with np.load(ROOT/str(path)/'steps.npz') as s:
        hist_t[r] = s['applied_history'][tr['t'][r]]
import torch
torch.set_num_threads(6)
for j, name in ((2, '1.6s'), (3, '3.2s')):
    near_t = mp.give_way_windows(tr, rt) & ~np.isnan(tr['cf_hold'][rt, j, 0])
    yt = (np.linalg.norm(tr['cf_retreat'][near_t, j]-tr['cf_hold'][near_t, j], axis=-1) > mp.EVENT_M).astype(np.float32)
    feats = lambda h: np.concatenate([h.reshape(len(h), -1), np.abs(h).reshape(len(h), -1)], 1)
    x = torch.from_numpy(feats(hist_t[near_t])); y = torch.from_numpy(yt)
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(3000):
        i = torch.randint(len(x), (512,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net((x[i]-mu)/sd)[:, 0], y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    near = mp.give_way_windows(ev, rows) & ~np.isnan(ev['cf_hold'][rows, j, 0])
    for s in ('val', 'transfer'):
        m = near & (split == s)
        label = np.linalg.norm(ev['cf_retreat'][m, j]-ev['cf_hold'][m, j], axis=-1) > mp.EVENT_M
        with torch.no_grad():
            sc = net((torch.from_numpy(feats(hist[m]))-mu)/sd)[:, 0].numpy()
        print(name, s, 'AUC of a classifier on the command history alone:', round(mp.auc(sc, label), 3), 'train windows', int(near_t.sum()), flush=True)
