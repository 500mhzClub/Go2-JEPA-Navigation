"""T5: the motion readout that turns the JEPA predictor into a motion forecaster.

    python -m go2nav.training.readouts features --recordings-root DIR [--recordings-root DIR2] --out FEATURES
    python -m go2nav.training.readouts t5-0 --features FEATURES --out DIR [--seed 2026091705]
    python -m go2nav.training.readouts t5a --parent T5-0_DIR/readout.pt --features FEATURES --out DIR [--seed 2026091806]
    python -m go2nav.training.readouts t5b --parent T5a_DIR/readout.pt --features FEATURES --out DIR [--seed 2026092205]

The readout (`go2nav.models.motion.MotionReadout`) maps a current frame and a future frame (pooled, layer-normalised
V-JEPA features) to the body-frame motion between them. It is trained on TRUE future frames; at run time it reads the
predictor's predicted future, which is what makes the predictor a motion forecaster. Loss: MSE in normalised motion
units; AdamW (lr 1e-3, weight decay 1e-4), gradient clip 1.0, batch 64.

- **T5-0**: a fresh readout on the 3,518 native examples at 500 ms (normalisation fitted on their targets); 24 epochs,
  each in the order `randperm(seed + epoch)`.
- **T5a** (full heading): from T5-0; 440 batches of 32 original examples and 32 new full-heading examples (each pool
  cycled through seeded permutations).
- **T5b** (maze view, all horizons): from T5a; 440 batches of 32 original/full-heading examples and 32 maze-view
  examples, each at a seeded horizon of 100-800 ms (horizons balanced per batch); trained on the CPU, as the original.
T5b is the readout deployed in C3, and the base of the C3 decoder (T7).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from go2nav.models.motion import MotionReadout
from go2nav.training.common import FeatureStore, write_lineage

REPO = Path(__file__).resolve().parents[2]
DATA = REPO/'data/training/readouts'
LR, WEIGHT_DECAY, GRAD_CLIP, BATCH, HALF = 1e-3, 1e-4, 1.0, 64, 32
T5_0_SEED, T5_0_EPOCHS = 2026091705, 24
T5A_SEED, T5B_SEED, CONTINUATION_STEPS = 2026091806, 2026092205, 440


def jsonl(name):
    return [json.loads(line) for line in (DATA/name).read_text().splitlines() if line.strip()]


def build_features(roots, out):
    from go2nav.models.vjepa import VJepaEncoder, preprocess

    def load(name):
        path = next((Path(r)/name for r in roots if (Path(r)/name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        return preprocess(Image.open(path))
    frames = (DATA/'frames.txt').read_text().split()
    return FeatureStore.build(out, frames, load, VJepaEncoder(), pooled=True)


def cycle(n, count, seed):
    """`count` indices cycling through seeded permutations of range(n)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n) for _ in range((count+n-1)//n)])[:count]


def update(model, optimiser, features, pairs, motion, device):
    """One step on (current, future) frame-name pairs and their motions."""
    x = features.get([f for pair in pairs for f in pair]).float().reshape(len(pairs), 2, 192, 1024).to(device)
    y = (torch.as_tensor(np.asarray(motion), dtype=torch.float32, device=device)-model.target_mean)/model.target_scale
    optimiser.zero_grad(set_to_none=True)
    loss = F.mse_loss(model.normalized(x[:, 0], x[:, 1]), y)
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()
    return float(loss.detach())


def load_readout(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    model = MotionReadout(state['target_mean'], state['target_scale'])
    model.load_state_dict(state)
    return model.to(device).train()


def finish(model, out, component, seed, parents, config, data, history, started):
    checkpoint = Path(out)/'readout.pt'
    torch.save(dict(model_state_dict=model.cpu().state_dict()), checkpoint)
    write_lineage(out, component=component, seed=seed, parents=parents, config=config, data=data, checkpoint=checkpoint,
                  metrics=dict(history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


def train_t5_0(features_dir, out, *, seed=T5_0_SEED, epochs=T5_0_EPOCHS, device='cuda:0'):
    Path(out).mkdir(parents=True, exist_ok=False)
    rows, features = jsonl('t5_original.jsonl'), FeatureStore(features_dir)
    targets = np.asarray([r['motion'] for r in rows], np.float32)
    torch.manual_seed(seed)
    model = MotionReadout(targets.mean(0), targets.std(0)).to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, started = [], time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        total = 0.
        for offset in range(0, len(order), BATCH):
            part = [rows[i] for i in order[offset:offset+BATCH]]
            total += len(part)*update(model, optimiser, features, [(r['current'], r['future']) for r in part],
                                      [r['motion'] for r in part], device)
        history.append(dict(epoch=epoch, train_normalised_mse=total/len(rows)))
        print(json.dumps(history[-1]), flush=True)
    return finish(model, out, 't5_0_motion_readout', seed, [], dict(epochs=epochs, batch=BATCH, lr=LR),
                  dict(samples='t5_original.jsonl', count=len(rows)), history, started)


def train_t5a(parent, features_dir, out, *, seed=T5A_SEED, steps=CONTINUATION_STEPS, device='cuda:0'):
    Path(out).mkdir(parents=True, exist_ok=False)
    old, new, features = jsonl('t5_original.jsonl'), jsonl('t5_full_heading.jsonl'), FeatureStore(features_dir)
    torch.manual_seed(seed)
    shared = cycle(len(old), steps*HALF, seed).reshape(steps, HALF)
    extra = cycle(len(new), steps*HALF, seed+2).reshape(steps, HALF)
    model = load_readout(parent, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, total, started = [], 0., time.time()
    for n in range(steps):
        part = [old[i] for i in shared[n]]+[new[i] for i in extra[n]]
        total += update(model, optimiser, features, [(r['current'], r['future']) for r in part], [r['motion'] for r in part], device)
        if (n+1) % 55 == 0:
            history.append(dict(updates=n+1, train_normalised_mse=total/55))
            print(json.dumps(history[-1]), flush=True)
            total = 0.
    return finish(model, out, 't5a_full_heading_readout', seed, [str(parent)], dict(steps=steps, batch=BATCH, lr=LR),
                  dict(samples=['t5_original.jsonl', 't5_full_heading.jsonl']), history, started)


def train_t5b(parent, features_dir, out, *, seed=T5B_SEED, steps=CONTINUATION_STEPS, device='cpu'):
    Path(out).mkdir(parents=True, exist_ok=False)
    data, features = json.loads((DATA/'t5_maze_view.json').read_text()), FeatureStore(features_dir)
    old, maze = data['old'], data['maze']
    torch.manual_seed(seed)
    np.random.seed(seed)
    shared = cycle(len(old['pairs']), steps*HALF, seed).reshape(steps, HALF)
    other = cycle(len(maze['pairs']), steps*HALF, seed+2).reshape(steps, HALF)
    horizons = []
    for k in (10, 11):  # shared and other horizons: every horizon equally often, shuffled
        h = np.tile(np.arange(8), steps*HALF//8)
        np.random.default_rng(seed+k).shuffle(h)
        horizons.append(h.reshape(steps, HALF))
    model = load_readout(parent, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, total, started = [], 0., time.time()
    for n in range(steps):
        pairs = [old['pairs'][i][h] for i, h in zip(shared[n], horizons[0][n])] + \
                [maze['pairs'][i][h] for i, h in zip(other[n], horizons[1][n])]
        motion = [old['motions'][i][h] for i, h in zip(shared[n], horizons[0][n])] + \
                 [maze['motions'][i][h] for i, h in zip(other[n], horizons[1][n])]
        total += update(model, optimiser, features, pairs, motion, device)
        if (n+1) % 55 == 0:
            history.append(dict(updates=n+1, train_normalised_mse=total/55))
            print(json.dumps(history[-1]), flush=True)
            total = 0.
    return finish(model, out, 't5b_maze_view_readout', seed, [str(parent)], dict(steps=steps, batch=BATCH, lr=LR, device=device),
                  dict(samples='t5_maze_view.json', old=len(old['pairs']), maze=len(maze['pairs'])), history, started)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('features')
    f.add_argument('--recordings-root', action='append', required=True)
    f.add_argument('--out', required=True)
    for name, seed, needs_parent in (('t5-0', T5_0_SEED, False), ('t5a', T5A_SEED, True), ('t5b', T5B_SEED, True)):
        s = sub.add_parser(name)
        if needs_parent:
            s.add_argument('--parent', required=True)
        s.add_argument('--features', required=True)
        s.add_argument('--out', required=True)
        s.add_argument('--seed', type=int, default=seed)
    a = p.parse_args()
    if a.command == 'features':
        build_features(a.recordings_root, a.out)
    elif a.command == 't5-0':
        print(train_t5_0(a.features, a.out, seed=a.seed))
    elif a.command == 't5a':
        print(train_t5a(a.parent, a.features, a.out, seed=a.seed))
    else:
        print(train_t5b(a.parent, a.features, a.out, seed=a.seed))


if __name__ == '__main__':
    main()
