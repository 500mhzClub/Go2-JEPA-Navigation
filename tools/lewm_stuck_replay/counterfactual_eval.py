"""Score the counterfactual ground truth (counterfactual.py) against the forecasters at the same decisions:
- `truth`: the simulator's motion (a perfect forecaster);
- `C5_<model>`: rollout and decoder, as deployed; `C5_<model>_truez`: the same decoder on the true future frames' z
  (the predictor replaced by the encoder of what really happened); `C6_<model>`.
Per source: decisions with an escape (a moving candidate not BLOCKED under the planner's rules and the fitted wall),
and per in-place turn the largest translation error against the truth over 100-800 ms and the forecast's dip below the
start clearance.

    python tools/lewm_stuck_replay/counterfactual_eval.py
"""
import json, statistics, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from stuck_forecast import EXTRA, MODELS, HalfPlane, forecasters, score, wall  # noqa: E402
from go2nav.models.lewm import normalise_frames, resize_frame
from types import SimpleNamespace

O = Path('outputs/lewm/stuck_counterfactual')
ACTIONS = ['hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn']


def as_forecast(motion):
    m = torch.as_tensor(motion, dtype=torch.float32)
    return torch.cat((m[..., :2], m[..., 2:3].sin(), m[..., 2:3].cos(), torch.full((*m.shape[:2], 1), -1000.)), -1).numpy()


@torch.inference_mode()
def encode(model, frames, device):
    x = torch.from_numpy(np.stack([resize_frame(f) for f in frames])).to(device)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        return model.encode(normalise_frames(x)[:, None])[:, 0].float()


def main():
    runs = {f"{e['cond']}_{Path(e['run']).name.split('_')[-2]}": Path(e['run'])
            for e in json.loads(Path('outputs/lewm/stuck_replay/stuck_list.json').read_text())}
    fc = forecasters(models=MODELS | EXTRA)
    rows = []
    for folder in sorted(p for p in O.iterdir() if p.is_dir()):
        run = runs[folder.name]
        plan = json.loads((run/'planning.json').read_text())
        hist = dict(np.load(run/'native/policy_histories.npz'))
        for f in sorted(folder.glob('d*.npz')):
            z = np.load(f)
            d = plan[int(z['decision'])]
            j = int(np.flatnonzero(hist['image_ns'] == d['measured_ns'])[0])
            applied = hist['applied_command_values'][j]
            ctx = SimpleNamespace(history=[dict(image=dict(measured_ns=d['measured_ns']-(3-k)*100_000_000, rgb=z['history'][k]),
                                                sensor_state=dict(control=dict(applied_command=dict(values=applied))))
                                           for k in range(4)])
            x, resid = wall(d)
            plane = HalfPlane(x)
            preds = dict(truth=as_forecast(z['motion']))
            for name, f_ in fc.items():
                preds[name] = f_.forecast(ctx, d['committed_prefix'], pulse=False)
                if name.startswith('C5'):
                    zh = f_.inputs(ctx, d['committed_prefix'], False)[0]
                    zt = encode(f_.model, z['frames'].reshape(-1, *z['frames'].shape[2:]), f_.device)
                    m = f_.decoder(zh.expand(48, -1, -1), zt).reshape(6, 8, 3).cpu()
                    preds[name+'_truez'] = as_forecast(m.numpy())
            row = dict(case=folder.name, decision=int(z['decision']), resid_mm=round(1000*resid, 2))
            truth = preds['truth']
            for name, p in preds.items():
                s = score(d, p, plane)
                row[name] = dict(escape=[a for a in ACTIONS[1:] if s[a]['mode'] != 'BLOCKED'],
                                 turn_err_mm=[float(1000*np.abs(np.linalg.norm(p[i, :, :2]-truth[i, :, :2], axis=1)).max()) for i in (4, 5)],
                                 turn_dip_mm=[float(1000*(s[a]['seg'][0]-min(s[a]['seg']))) for a in ('left_turn', 'right_turn')],
                                 turn_trans_mm=[float(1000*np.linalg.norm(p[i, :, :2], axis=1).max()) for i in (4, 5)],
                                 xy=np.round(p[:, :, :2], 4).tolist())
            rows.append(row)
            print(row['case'], row['decision'], {k: row[k]['escape'] for k in preds}, flush=True)
    (O/'eval_with_corpus.json').write_text(json.dumps(rows))
    q = lambda v: f'{statistics.median(v):5.1f} / {statistics.quantiles(v, n=10)[8]:5.1f}'
    print(f'\n{len(rows)} decisions; wall fit residual median {statistics.median(r["resid_mm"] for r in rows):.2f} mm')
    print(f'{"source":18s} escapes  turn error vs truth (med/p90 mm)  turn dip (med/p90)  turn translation (med/p90)')
    for name in rows[0]:
        if isinstance(rows[0][name], dict):
            v = [r[name] for r in rows]
            e = [x for r in v for x in r['turn_err_mm']]
            print(f'{name:18s} {sum(bool(r["escape"]) for r in v):3d}/{len(v)}   {q(e)}                     '
                  f'{q([x for r in v for x in r["turn_dip_mm"]])}       {q([x for r in v for x in r["turn_trans_mm"]])}')


if __name__ == '__main__':
    main()
