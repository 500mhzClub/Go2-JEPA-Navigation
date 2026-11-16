"""Which input does the phase C latent depend on? Permutation attribution on validation frames (CPU only)."""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b')
from go2nav.models.hjepa import load_hjepa
from go2nav.models.lewm import normalise_frames

torch.set_num_threads(8)
ROOT = Path('/mnt/workspace_drive/go2nav_lewm_v2')
OBS = Path('/mnt/workspace_drive/go2nav_lewm_v2_obs')
H = Path('/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/hjepa')
rng = np.random.default_rng(0)
kinds = ['mission', 'explore', 'free', 'standoff', 'standoff_b']
units = sorted(p.name for p in (ROOT/'units/lewm_val').iterdir())
chosen = []
for k in kinds:
    pool = [u for u in units if u.split('_', 1)[1] == k]
    chosen += list(rng.choice(pool, 16, replace=False))
frames, obs, kind, unit_id = [], [], [], []
for i, u in enumerate(chosen):
    f = np.load(ROOT/'units/lewm_val'/u/'frames.npy', mmap_mode='r')
    o = np.load(OBS/'units/lewm_val'/u/'obs.npy', mmap_mode='r')
    n = min(len(f), len(o))
    if n < 60:
        continue
    t = np.sort(rng.choice(np.arange(20, n-1), 12, replace=False))
    frames.append(np.asarray(f[t])); obs.append(np.asarray(o[t], np.float32))
    kind += [u.split('_', 1)[1]]*12; unit_id += [i]*12
frames = torch.from_numpy(np.concatenate(frames)); obs = torch.from_numpy(np.concatenate(obs))
unit_id = np.asarray(unit_id)
N = len(frames)
print('frames', N, 'obs', tuple(obs.shape), flush=True)
RING = slice(0, 1440); GYRO = slice(1440, 1446); HIST = slice(1446, 1491)

def embed(model, fr, ob):
    out = []
    with torch.inference_mode():
        for s in range(0, len(fr), 96):
            out.append(model.level1.embed(normalise_frames(fr[s:s+96]), None if ob is None else ob[s:s+96]).float())
    return torch.cat(out)

report = {}
for name, ckpt in (('phase_c_40k', H/'go2_3level_c/hjepa_step40000.pt'), ('phase_c_20k', H/'go2_3level_c/hjepa_step20000.pt'),
                   ('control_60k', H/'go2_3level_p14_control/hjepa.pt')):
    model = load_hjepa(ckpt, 'cpu')
    has_obs = model.level1.observations is not None
    z = embed(model, frames, obs if has_obs else None)
    total = float(z.var(0).sum())
    row = dict(total_var=round(total, 2), z_std_mean=round(float(z.std(0).mean()), 3))
    perms = [torch.from_numpy(np.random.default_rng(s).permutation(N)) for s in (1, 2, 3)]
    def share(fr_fn, ob_fn):
        vals = []
        for p in perms:
            zp = embed(model, fr_fn(p), ob_fn(p) if has_obs else None)
            vals.append(float((z-zp).pow(2).sum(1).mean())/(2*total))
        return round(float(np.mean(vals)), 3)
    def swap(sl):
        def f(p):
            o = obs.clone(); o[:, sl] = obs[p][:, sl]; return o
        return f
    same_f, same_o = (lambda p: frames), (lambda p: obs)
    row['perm_image'] = share(lambda p: frames[p], same_o)
    if has_obs:
        row['perm_ring'] = share(same_f, swap(RING))
        row['perm_proprio_all'] = share(same_f, swap(slice(1440, 1491)))
        row['perm_command_history'] = share(same_f, swap(HIST))
        row['perm_gyro_gravity'] = share(same_f, swap(GYRO))
        row['perm_everything'] = share(lambda p: frames[p], lambda p: obs[p])
    # linear: how much of z is a linear function of the command history / how well z gives the last applied command
    held = np.isin(unit_id, np.unique(unit_id)[::4]); tr, te = torch.from_numpy(~held), torch.from_numpy(held)
    def ridge_r2(x, y, lam=1.0):
        x = x.double(); y = y.double()
        m, s = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6); xs = (x-m)/s
        ym = y[tr].mean(0)
        w = torch.linalg.solve(xs[tr].T@xs[tr]+lam*torch.eye(xs.shape[1], dtype=xs.dtype), xs[tr].T@(y[tr]-ym))
        pred = xs[te]@w+ym
        return round(float(1-((pred-y[te])**2).sum()/((y[te]-y[te].mean(0))**2).sum()), 3)
    hist = obs[:, HIST]
    row['r2_z_from_command_history'] = ridge_r2(hist, z, lam=1.0)
    row['r2_last_command_from_z'] = ridge_r2(z, hist[:, -3:], lam=10.0)
    row['r2_whole_history_from_z'] = ridge_r2(z, hist, lam=10.0)
    report[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(report, open(Path(__file__).with_name('attribution.json'), 'w'), indent=1)
