"""P28's measures (docs/dynamic/PLAN.md P28 item 3): forecasts of the F004 validation scenes against the walkers' true
positions, for M5, the references and the object-token JEPA.

    python -m go2nav.eval.ring_tokens references --data ROOT --corpus CORPUS --m5 MODEL --out DIR
    python -m go2nav.eval.ring_tokens score --data ROOT --corpus CORPUS --run RUN
    python -m go2nav.eval.ring_tokens table --references DIR --runs RUN [RUN ...]

A **forecast** is every slot's position in the world after each of the 16 steps of each of a scene's 7 futures,
(N, 7, 16, 6, 2). `summary` scores one at 1.6, 3.2 and 6.4 s:
- **(a)** the median error on the recorded future over matched walkers (`fde`), and over those waiting on the robot;
- **(b)** on the 15 pairs of the 6 branches: the mean response error and zero-response's, on all pairs and on
  responsive ones (true response over 5 cm); the false-response rate (predicted over 5 cm where the true response is
  at most 5 cm); the slope of predicted on true response vectors with its 95% interval over layouts
  (`eval.paired_forecast_metrics`); the same by the walker's kind.
The references are M5 (`models.object_jepa`, its own decoded rollout with the walls looked up at each step), the
tracker's constant velocity and holding. The JEPA's forecast reads its latent rollout through a head trained on
encoded tokens only (`train_head`); its retention is the head's error on the encoded true future tokens.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from go2nav.data import ring_scenes
from go2nav.eval import paired_forecast_metrics as pfm
from go2nav.models.object_jepa import STEP_S

RESPONSIVE_M = .05
KINDS = {v: k for k, v in pfm.KINDS.items()}


def collect(scenes, forecast, batch=1024):
    """`forecast(b)`: a batch (`Scenes.batch`) to positions in the scenes' frame (B, 16, 6, 2). Returns the world
    positions for every scene and future (N, 7, 16, 6, 2), numpy."""
    out = np.zeros((scenes.count, ring_scenes.FUTURES, ring_scenes.STEPS, 6, 2), np.float32)
    for i in range(0, scenes.count, batch):
        index = torch.arange(i, min(i+batch, scenes.count), device=scenes.device)
        for future in range(ring_scenes.FUTURES):
            b = scenes.batch(index, torch.full_like(index, future))
            out[i:i+batch, future] = scenes.world(b['frame'], forecast(b)).cpu().numpy()
    return out


def m5_forecast(model, scenes):
    @torch.no_grad()
    def forecast(b):
        return model.rollout(b['objects'], b['robot'], b['actions'], b['future_robot'],
                             lambda p: scenes.walls(b['frame'], p), ring_scenes.STEPS)
    return forecast


def summary(data, prediction, draws=1000):
    """The module's measures of one forecast: {horizon: {...}}."""
    bootstrap = pfm.LayoutBootstrap(data['maze'], draws=draws)
    out = {}
    for horizon, endpoint in pfm.HORIZONS.items():
        fde = pfm.trajectory_rows(data, prediction, endpoint, 'factual')['fde']
        row = pfm.response_rows(data, prediction, endpoint)
        responsive, quiet = row['true_norm'] > RESPONSIVE_M, row['true_norm'] <= RESPONSIVE_M

        def part(mask):
            fit = bootstrap.regression(row['true'][mask], row['predicted'][mask], row['layout'][mask]) \
                if mask.sum() > 10 else dict(slope=None, ci95=None)
            r, q = mask & responsive, mask & quiet
            mean = lambda v: float(v.mean()) if len(v) else None
            return dict(pairs=int(mask.sum()), error=mean(row['error'][mask]), zero=mean(row['true_norm'][mask]),
                        responsive=int(r.sum()), responsive_error=mean(row['error'][r]),
                        responsive_zero=mean(row['true_norm'][r]), quiet=int(q.sum()),
                        false_rate=mean((row['pred_norm'][q] > RESPONSIVE_M).astype(float)), slope=fit['slope'],
                        ci=fit['ci95'])
        everything = np.ones(len(row['error']), bool)
        out[str(horizon)] = dict(
            fde=dict(n=len(fde['value']), median=float(np.median(fde['value'])), mean=float(fde['value'].mean())),
            fde_waiting=dict(n=int(fde['waiting'].sum()), median=float(np.median(fde['value'][fde['waiting']]))),
            response=part(everything), **{f'response_{KINDS[k]}': part(row['kind'] == k) for k in (1, 2, 3)})
    return out


def table(rows, m5=None):
    """Markdown: per forecast and horizon, (a) and (b); with M5's row, the bars of PLAN P28 item 4."""
    f = lambda v, d=3: '-' if v is None else f'{v:.{d}f}'
    lines = ['| Forecast | Horizon | (a) median m | waiting | (b) error / zero | responsive: error / zero | false | '
             'slope [95%] | bars |', '|---|---|---|---|---|---|---|---|---|']
    for name, row in rows.items():
        for h, v in row.items():
            if h in ('R', 'meta'):
                continue
            r = v['response']
            ci = '-' if r['ci'] is None else f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
            mark = ''
            if m5 is not None and name not in ('M5', 'constant velocity', 'hold'):
                ref = m5[h]
                a = v['fde']['median'] <= 1.5*ref['fde']['median']
                b = r['false_rate'] < ref['response']['false_rate'] and r['slope'] >= ref['response']['slope']
                mark = f"a{'✓' if a else '✗'} b{'✓' if b else '✗'}"
            lines.append(f"| {name} | {h} s | {f(v['fde']['median'])} | {f(v['fde_waiting']['median'])} | "
                         f"{f(r['error'])} / {f(r['zero'])} | {f(r['responsive_error'])} / {f(r['responsive_zero'])} | "
                         f"{f(r['false_rate'], 2)} | {f(r['slope'], 2)} {ci} | {mark} |")
    return '\n'.join(lines)


# -- the JEPA's head and forecast -------------------------------------------------------------------------------------

HEADS = dict(small=dict(steps=4000, batch=256, hidden=512, layers=2),          # P28's first head
             mlp=dict(steps=10000, batch=512, hidden=1024, layers=3))           # P27b's Q0


def train_head(model, scenes, steps=4000, batch=256, lr=1e-3, seed=0, hidden=512, layers=2):
    """`models.ring_tokens.PositionHead` on encoded tokens (history slots seen, and the true future tokens)."""
    from go2nav.models.ring_tokens import PositionHead
    from go2nav.training.balls import schedule
    device = scenes.device
    torch.manual_seed(2026111670+seed)
    head = PositionHead(model.config, hidden, layers).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(200, steps))
    g = torch.Generator(device=device).manual_seed(2026111671+seed)
    for step in range(steps):
        index = torch.randint(0, scenes.count, (batch,), generator=g, device=device)
        b = scenes.batch(index, torch.randint(0, ring_scenes.FUTURES, (batch,), generator=g, device=device))
        with torch.no_grad():
            z = torch.cat((model.encode_objects(b['objects']), model.encode_objects(b['future_objects'])), 1)
        target = torch.cat((b['objects'][..., :2], b['truth']), 1)/model.config.position_scale
        count = torch.cat((b['seen'], b['valid']), 1)
        loss = ((head(z)-target).pow(2).sum(-1)*count).sum()/count.sum()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step+1) % 1000 == 0:
            print(f'head step {step+1}: rms {model.config.position_scale*float(loss.detach())**.5:.4f} m', flush=True)
    return head.eval().requires_grad_(False)


def jepa_forecast(model, head):
    @torch.no_grad()
    def forecast(b):
        z = model.encode(b['objects'], b['robot'])
        rolled = model.rollout(z, b['actions'], model.encode_robot(b['future_robot']), ring_scenes.STEPS,
                               b['present'][:, -1])
        return head(rolled[:, :, 1:])*model.config.position_scale
    return forecast


@torch.no_grad()
def retention(model, head, scenes, batch=1024):
    """The head's error (m) on the encoded true future tokens of the recorded future: the median and the mean."""
    errors = []
    for i in range(0, scenes.count, batch):
        index = torch.arange(i, min(i+batch, scenes.count), device=scenes.device)
        b = scenes.batch(index, torch.full_like(index, ring_scenes.FACTUAL))
        read = head(model.encode_objects(b['future_objects']))*model.config.position_scale
        errors.append((read-b['truth']).norm(dim=-1)[b['valid']])
    errors = torch.cat(errors)
    return dict(median=float(errors.median()), mean=float(errors.mean()))


# -- commands ---------------------------------------------------------------------------------------------------------

def setup(root, corpus, split='val'):
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.3)
    rasters = ring_scenes.rasters_for(root, corpus, device)
    return ring_scenes.Scenes(root, split, rasters, device), rasters


def references(root, corpus, m5, out):
    from go2nav.training.object_jepa import load_model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scenes, _ = setup(root, corpus)
    data = scenes.numpy
    predictions = dict(pfm.baselines(data), m5=collect(scenes, m5_forecast(load_model(m5, scenes.device), scenes)))
    rows = {name: summary(data, np.ascontiguousarray(predictions[key]))
            for name, key in (('hold', 'persistence'), ('constant velocity', 'cv'), ('M5', 'm5'))}
    (out/'references.json').write_text(json.dumps(rows, indent=1))
    print(table(rows))


def score(root, corpus, run, head_kind='small'):
    """RUN/bench.json with P28's first head; with `head_kind` 'mlp', RUN/head_mlp.pt and RUN/bench_mlp.json."""
    from go2nav.training.ring_tokens import load
    run, tag, options = Path(run), '' if head_kind == 'small' else f'_{head_kind}', HEADS[head_kind]
    scenes, rasters = setup(root, corpus)
    model, state = load(run/'model.pt', scenes.device)
    if (run/f'head{tag}.pt').exists():
        from go2nav.models.ring_tokens import PositionHead
        head = PositionHead(model.config, options['hidden'], options['layers']).to(scenes.device)
        head.load_state_dict(torch.load(run/f'head{tag}.pt'))
        head.eval().requires_grad_(False)
    else:
        head = train_head(model, ring_scenes.Scenes(root, 'train', rasters, scenes.device), **options)
        torch.save(head.state_dict(), run/f'head{tag}.pt')
    row = summary(scenes.numpy, collect(scenes, jepa_forecast(model, head)))
    row['R'] = retention(model, head, scenes)
    row['meta'] = dict(recipe=state['recipe'], seed=state['seed'], steps=state['step'])
    (run/f'bench{tag}.json').write_text(json.dumps(row, indent=1))
    print(table({run.name: row}))
    print('R:', row['R'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('references')
    r.add_argument('--m5', required=True)
    r.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--run', required=True)
    for command in (r, s):
        command.add_argument('--data', required=True)
        command.add_argument('--corpus', required=True)
    t = sub.add_parser('table')
    for command in (s, t):
        command.add_argument('--head', choices=sorted(HEADS), default='small')
    t.add_argument('--references', required=True)
    t.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    if args.command == 'references':
        references(args.data, args.corpus, args.m5, args.out)
    elif args.command == 'score':
        score(args.data, args.corpus, args.run, args.head)
    else:
        refs = json.loads((Path(args.references)/'references.json').read_text())
        rows = {'constant velocity': refs['constant velocity'], 'M5': refs['M5'],
                **{Path(run).name: json.loads((Path(run)/('bench.json' if args.head == 'small' else f'bench_{args.head}.json')).read_text())
                   for run in args.runs}}
        print(table(rows, refs['M5']))


if __name__ == '__main__':
    main()
