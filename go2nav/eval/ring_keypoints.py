"""P31 stage 2's measures (docs/dynamic/PLAN.md P31 item 5): both models on F004's validation scenes.

    python -m go2nav.eval.ring_keypoints score --corpus CORPUS --scenes F004 --run RUN [--keypoints DIR/keypoints_val.npy]
    python -m go2nav.eval.ring_keypoints compare --corpus CORPUS --scenes F004 --tracker RUN --learned RUN --m5 MODEL --out FILE

**score:** the read-out head (P27b's Q0 size: an MLP of 3 x 1,024, 10,000 steps), trained on encoded training tokens
against each token's true walker (`data.ring_windows.Frames.truth`; labels for the head only); then every scene's
history from the corpus's scans at its frames, rolled out 16 steps under each of its 7 futures (F004's poses and
requests), read by the head into world positions. Each of F004's matched walkers is given the token nearest it at the
branch point, within 0.3 m (else it is not found). RUN/forecast.npz: the forecasts per F004 slot (N, 7, 16, 6, 2)
and `found` (N, 6).

**compare:** P28's (a) and (b) (`eval.ring_tokens.summary`) for both runs, M5 and the tracker's constant velocity,
all on the walkers both runs find; the stage-2 bars of PLAN P31 item 6.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data import ring_rasters as rr
from go2nav.data import ring_scenes
from go2nav.data.ring_windows import HISTORY, Frames
from go2nav.eval import paired_forecast_metrics as pfm
from go2nav.eval import ring_tokens
from go2nav.models.ring_tokens import PositionHead
from go2nav.training.balls import schedule
from go2nav.training.ring_windows import load

HEAD = dict(steps=10000, batch=512, hidden=1024, layers=3)


def train_head(model, frames, kind, steps, batch, hidden, layers, cache=120, seed=0, future=8):
    device = frames.device
    torch.manual_seed(2026111870+seed)
    g = torch.Generator(device=device).manual_seed(2026111871+seed)
    windows = torch.as_tensor(frames.windows(future), device=device)
    tokens, targets = [], []
    with torch.no_grad():
        for _ in range(cache):
            t0 = windows[torch.randint(0, len(windows), (512,), generator=g, device=device)]
            b = frames.batch(kind, t0, future)
            z = torch.cat((model.encode_objects(b['objects']), model.encode_objects(b['future_objects'])), 1)
            truth = frames.truth(b['xy'], b['steps'])
            keep = torch.isfinite(truth).all(-1) & torch.cat((b['present'], b['valid']), 1)
            tokens.append(z[keep])
            targets.append((rr.to_frame(torch.nan_to_num(truth), b['frame'])/model.config.position_scale)[keep])
    tokens, targets = torch.cat(tokens).float(), torch.cat(targets).float()
    head = PositionHead(model.config, hidden, layers).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(200, steps))
    for step in range(steps):
        pick = torch.randint(0, len(tokens), (batch*16,), generator=g, device=device)
        loss = (head(tokens[pick])-targets[pick]).pow(2).sum(-1).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step+1) % 2000 == 0:
            print(f'head step {step+1}: rms {model.config.position_scale*float(loss.detach())**.5:.4f} m '
                  f'({len(tokens)} tokens)', flush=True)
    return head.eval().requires_grad_(False)


def scenes(root):
    folder = Path(root)/'val'
    keys = ('episode', 't0', 'walker', 'future_pose', 'future_commands')
    return {k: np.load(folder/f'{k}.npy') for k in keys}


@torch.no_grad()
def forecast(model, head, frames, kind, data, batch=512):
    """Forecasts per F004 slot (N, 7, 16, 6, 2) in the world, and found (N, 6)."""
    device = frames.device
    at = frames.corpus.starts[data['episode'].astype(np.int64)]+data['t0'].astype(np.int64)
    n = len(at)
    out = np.zeros((n, ring_scenes.FUTURES, ring_scenes.STEPS, 6, 2), np.float32)
    found = np.zeros((n, 6), bool)
    for i in range(0, n, batch):
        rows = slice(i, min(i+batch, n))
        t0 = torch.as_tensor(at[rows], device=device)
        for f in range(ring_scenes.FUTURES):
            pose = torch.as_tensor(data['future_pose'][rows, f], device=device)
            requested = torch.as_tensor(data['future_commands'][rows, f], device=device)
            b = frames.batch(kind, t0, ring_scenes.STEPS, future_pose=pose, future_requested=requested)
            z = model.encode(b['objects'], b['robot'])
            rolled = model.rollout(z, b['actions'], model.encode_robot(b['future_robot']), ring_scenes.STEPS,
                                   b['present'][:, -1])
            world = rr.to_world(head(rolled[:, :, 1:].float())*model.config.position_scale, b['frame']).cpu().numpy()
            if f == 0:
                tokens_at_t0 = b['xy'][:, HISTORY-1].cpu().numpy()                             # (B, S, 2) world
                alive = b['present'][:, -1].cpu().numpy()
                truth = frames.corpus.walkers_xy[at[rows]]                                    # (B, 4, 2)
                assign = np.full((len(t0), 6), -1)
                for r in range(len(t0)):
                    taken = set()
                    for s, w in enumerate(data['walker'][rows][r]):
                        if w < 0:
                            continue
                        d = np.linalg.norm(np.nan_to_num(tokens_at_t0[r], nan=1e3)-truth[r, w], axis=-1)
                        d[~alive[r]] = 1e3
                        for k in taken:
                            d[k] = 1e3
                        k = int(d.argmin())
                        if d[k] < .3:
                            assign[r, s] = k
                            taken.add(k)
                found[rows] = assign >= 0
            pick = np.maximum(assign, 0)
            out[rows, f] = np.take_along_axis(world, pick[:, None, :, None], 2)
    return out, found


def score(corpus_root, scenes_root, run, keypoints=None):
    run, device = Path(run), torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.5)
    model, state = load(run/'model.pt', device)
    kind = state['kind']
    if keypoints is None and kind == 'learned':
        keypoints = str(Path(state['keypoints']).with_name('keypoints_val.npy'))
    if (run/'head_mlp.pt').exists():
        head = PositionHead(model.config, HEAD['hidden'], HEAD['layers']).to(device)
        head.load_state_dict(torch.load(run/'head_mlp.pt'))
        head.eval().requires_grad_(False)
    else:
        train_frames = Frames(corpus_root, 'train', device, keypoints=state['keypoints'], labels=True)
        head = train_head(model, train_frames, kind, **HEAD)
        torch.save(head.state_dict(), run/'head_mlp.pt')
        del train_frames
        torch.cuda.empty_cache()
    frames = Frames(corpus_root, 'val', device, keypoints=keypoints, labels=True)
    out, found = forecast(model, head, frames, kind, scenes(scenes_root))
    np.savez(run/'forecast.npz', forecast=out, found=found)
    print(f'{run.name}: found {found.sum()} of {int((scenes(scenes_root)["walker"] >= 0).sum())} matched walkers', flush=True)


def compare(corpus_root, scenes_root, tracker, learned, m5, out):
    data = ring_scenes.load(scenes_root, 'val')
    data = {k: np.asarray(v) for k, v in data.items()}
    runs = {name: np.load(Path(run)/'forecast.npz') for name, run in (('tracker tokens', tracker), ('learned tokens', learned))}
    both = runs['tracker tokens']['found'] & runs['learned tokens']['found']
    masked = dict(data, valid=data['valid'] & both[:, None, None, :])
    masked['target'] = np.where(masked['valid'][..., None], data['target'], 0.)
    m5_file = Path(out).with_name('m5_val_forecast.npy')
    if not m5_file.exists():
        from go2nav.training.object_jepa import load_model
        scenes_gpu, _ = ring_tokens.setup(scenes_root, corpus_root)
        np.save(m5_file, ring_tokens.collect(scenes_gpu, ring_tokens.m5_forecast(load_model(m5, scenes_gpu.device), scenes_gpu)))
    predictions = {'constant velocity': np.ascontiguousarray(pfm.baselines(masked)['cv']), 'M5': np.load(m5_file),
                   **{k: v['forecast'] for k, v in runs.items()}}
    rows = {name: ring_tokens.summary(masked, p) for name, p in predictions.items()}
    t, l = rows['tracker tokens'], rows['learned tokens']
    bars = {h: dict(a=l[h]['fde']['median'] <= 1.25*t[h]['fde']['median'],
                    false=l[h]['response']['false_rate'] <= t[h]['response']['false_rate']+.05,
                    slope=l[h]['response']['slope'] >= .8*t[h]['response']['slope']) for h in t}
    result = dict(population=dict(matched=int((data['walker'] >= 0).sum()), tracker_found=int(runs['tracker tokens']['found'].sum()),
                                  learned_found=int(runs['learned tokens']['found'].sum()), both=int(both.sum())),
                  rows=rows, bars=bars, passed=all(all(v.values()) for v in bars.values()))
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result['population']))
    print(ring_tokens.table(rows))
    print(json.dumps(bars), 'passed' if result['passed'] else 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--run', required=True)
    s.add_argument('--keypoints')
    c = sub.add_parser('compare')
    c.add_argument('--tracker', required=True)
    c.add_argument('--learned', required=True)
    c.add_argument('--m5', required=True)
    c.add_argument('--out', required=True)
    for command in (s, c):
        command.add_argument('--corpus', required=True)
        command.add_argument('--scenes', required=True)
    args = parser.parse_args()
    if args.command == 'score':
        score(args.corpus, args.scenes, args.run, args.keypoints)
    else:
        compare(args.corpus, args.scenes, args.tracker, args.learned, args.m5, args.out)


if __name__ == '__main__':
    main()
