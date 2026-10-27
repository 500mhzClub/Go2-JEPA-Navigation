"""The supervised ceiling of the mover probe (docs/dynamic/PLAN.md, P4): a CNN trained directly on the frames, with the
labels as targets, to localise the mover and forecast its motion. It bounds what the frames hold: a world model's probe
scores (`go2nav.eval.mover_probe`) near it mean the camera, not the world model, is the limit.

    python -m go2nav.eval.mover_ceiling ROOT --train TRAIN.npz --eval EVAL.npz --out DIR [--epochs 12] [--device cpu]

**Inputs:** the four frames t-3..t a world model's history holds (224 x 168 RGB, 12 channels) plus two coordinate
channels, on the probe's windows (the labels files of `mover_probe labels`, restricted to its usable windows), stacked
once into DIR/frames_{train,eval}.npy (uint8 memory maps). **Network** (`Ceiling`): five stride-2 convolutions with
group norm and GELU, flattened (no pooling: positions stay in the features), then a 512-unit layer. **Targets:** the
mover's position at t and its displacement to each horizon (normalised; mean squared error), as the probe's heads.
AdamW with cosine decay for a fixed number of epochs, bf16 autocast; no selection on the evaluation windows.
**Scores:** `mover_probe.score` on the validation and transfer windows (forecasts start from the true position at t,
as the probe's displacement heads do), in DIR/results.json. There is no command input, so no response test.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

import numpy as np

from go2nav.eval import mover_probe as mp

SEED, BATCH, LR, WEIGHT_DECAY, EPOCHS = 2026102702, 128, 1e-3, 1e-4, 12
HISTORY = 4


def _unit_frames(args):
    """Write one unit's windows' frames t-3..t into the stack (rows of `out`); returns the usable mask of its windows."""
    root, path, t, rows, out = args
    folder = Path(root)/path
    with np.load(folder/'steps.npz') as steps:
        ok = mp.usable(dict(block_valid=steps['block_valid']), t)
    frames = np.load(folder/'frames.npy', mmap_mode='r')
    stack = np.load(out, mmap_mode='r+')
    for r, s in zip(rows, t):
        stack[r] = frames[s-HISTORY+1:s+1]
    stack.flush()
    return ok


def stack_frames(root, data, out, workers=16):
    """DIR/frames_*.npy (W, 4, H, W, 3) uint8 and the usable mask (W,), cached beside it."""
    out, mask_path = Path(out), Path(out).with_suffix('.ok.npy')
    if mask_path.exists():
        return np.load(mask_path)
    shape = np.load(Path(root)/data['units'][0]/'frames.npy', mmap_mode='r').shape[1:]
    np.lib.format.open_memmap(out, mode='w+', dtype=np.uint8, shape=(len(data['t']), HISTORY, *shape)).flush()
    jobs = []
    for u, path in enumerate(data['units']):
        rows = np.flatnonzero(data['unit'] == u)
        if len(rows):
            jobs.append((str(root), str(path), data['t'][rows], rows, str(out)))
    ok = np.zeros(len(data['t']), bool)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('forkserver')) as pool:
        for job, unit_ok in zip(jobs, pool.map(_unit_frames, jobs, chunksize=4)):
            ok[job[3]] = unit_ok
    np.save(mask_path, ok)
    return ok


def targets(data, rows):
    """(n, 2 + 2 x len(HORIZONS)): the position at t and the displacement to each horizon."""
    return np.concatenate([data['xy_t'][rows], (data['target'][rows]-data['xy_t'][rows, None]).reshape(len(rows), -1)],
                          1).astype(np.float32)


def network(height, width, outputs):
    import torch
    import torch.nn as nn

    class Ceiling(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c = [], 3*HISTORY+2
            for width_out, kernel in ((32, 5), (64, 3), (128, 3), (256, 3), (256, 3)):
                layers += [nn.Conv2d(c, width_out, kernel, 2, kernel//2), nn.GroupNorm(8, width_out), nn.GELU()]
                c = width_out
            self.features = nn.Sequential(*layers)
            with torch.no_grad():
                n = self.features(torch.zeros(1, 3*HISTORY+2, height, width)).numel()
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 512), nn.GELU(), nn.Linear(512, outputs))
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
            self.register_buffer('coords', torch.stack((xs, ys))[None])

        def forward(self, frames):
            """frames (B, 4, H, W, 3) uint8."""
            b = len(frames)
            x = frames.permute(0, 1, 4, 2, 3).reshape(b, 3*HISTORY, *frames.shape[2:4]).float()/127.5-1
            x = torch.cat((x, self.coords.expand(b, -1, -1, -1)), 1)
            return self.head(self.features(x.contiguous(memory_format=torch.channels_last)))

    return Ceiling()


def batches(stack, rows, rng=None):
    """Batches of (rows, frames) read from the memory map, in a background thread."""
    import queue
    import threading
    order = rows if rng is None else rng.permutation(rows)
    chunks = [np.sort(order[s:s+BATCH]) for s in range(0, len(order), BATCH)]
    q = queue.Queue(maxsize=8)

    def read():
        for chunk in chunks:
            q.put((chunk, np.ascontiguousarray(stack[chunk])))
        q.put(None)
    threading.Thread(target=read, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def run(root, train, evaluation, out, *, epochs=EPOCHS, device='cpu', threads=None):
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    sets = {}
    for name, path in (('train', train), ('eval', evaluation)):
        data = dict(np.load(path))
        ok = stack_frames(root, data, out/f'frames_{name}.npy')
        sets[name] = (data, np.load(out/f'frames_{name}.npy', mmap_mode='r'), np.flatnonzero(ok))
    (dtrain, strain, rtrain), (deval, seval, reval) = sets['train'], sets['eval']
    y = torch.from_numpy(targets(dtrain, np.arange(len(dtrain['t']))))
    ymean, yscale = y[rtrain].mean(0), y[rtrain].std(0).clamp_min(1e-6)
    torch.manual_seed(SEED)
    net = network(*strain.shape[2:4], y.shape[1]).to(device).to(memory_format=torch.channels_last)
    steps = epochs*int(np.ceil(len(rtrain)/BATCH))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=.05)
    rng, log = np.random.default_rng(SEED), open(out/'train.jsonl', 'w')
    autocast = lambda: torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    for epoch in range(epochs):
        net.train()
        total, n, start = 0., 0, time.time()
        for rows, frames in batches(strain, rtrain, rng):
            with autocast():
                loss = ((net(torch.from_numpy(frames).to(device)).float()-((y[rows]-ymean)/yscale).to(device))**2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            sched.step()
            total, n = total+float(loss)*len(rows), n+len(rows)
        log.write(json.dumps(dict(epoch=epoch, loss=round(total/n, 4), samples_per_s=round(n/(time.time()-start), 1)))
                  + '\n')
        log.flush()
    net.eval()
    predicted = np.zeros((len(deval['t']), y.shape[1]), np.float32)
    with torch.inference_mode():
        for rows, frames in batches(seval, reval):
            with autocast():
                predicted[rows] = (net(torch.from_numpy(frames).to(device)).float()*yscale+ymean).cpu().numpy()
    split = mp.kinds(deval)[reval]
    results = dict(train_windows=int(len(rtrain)), eval_windows=int(len(reval)), epochs=epochs,
                   horizons_s=[0.]+[h/10 for h in mp.HORIZONS], heads=dict(cnn={}))
    for s in ('val', 'transfer'):
        rows = reval[split == s]
        entries = [dict(horizon_s=0., **mp.score(deval, rows, predicted[rows, :2], None))]
        for j, h in enumerate(mp.HORIZONS):
            forecast = deval['xy_t'][rows]+predicted[rows, 2+2*j:4+2*j]
            entries.append(dict(horizon_s=h/10, **mp.score(deval, rows, forecast, j)))
        results['heads']['cnn'][s] = entries
    (out/'results.json').write_text(json.dumps(results, indent=1)+'\n')
    torch.save(net.state_dict(), out/'ceiling.pt')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root')
    parser.add_argument('--train', required=True)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.train, args.eval, args.out, epochs=args.epochs, device=args.device,
                         threads=args.threads), indent=1))


if __name__ == '__main__':
    main()
