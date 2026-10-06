"""T2-T4: adapt the JEPA predictor to the Go2 camera, add balanced starts, then condition it on the horizon.

    python -m go2nav.training.native_predictor features --recordings-root DIR [--recordings-root DIR2] --out FEATURES
    python -m go2nav.training.native_predictor t2 --parent T1.pt --features FEATURES --out DIR [--seed 2026091704]
    python -m go2nav.training.native_predictor t3 --parent T2_DIR/t2.pt --features FEATURES --out DIR [--seed 2026091710]
    python -m go2nav.training.native_predictor t4 --parent T3_DIR/t3.pt --features FEATURES --out DIR [--seed 2026091802]

Data (data/training/native_predictor): examples from the Go2's own camera in training mazes (moving-action switch
family, geometry-progress family, short-pulse learning) and 48 balanced starts (the robot starting from rest with each
command). Each example has three context frames (1 s, 0.5 s, 0 s before), a target frame, the commands as the safety
limiter applied them (forward, yaw) and the normalised control history. Frames are encoded once, layer-normalised and
stored in float16 (`features`).

All stages minimise the L1 between the layer-normalised predicted and true target tokens (bf16 autocast, float32 loss),
with AdamW (lr 3e-4, weight decay 0.01) and gradient clip 1.0, batch 16.

- **T2** (native adaptation, 500 ms): from T1; 24 epochs over the 3,518 examples, each epoch in the order
  `randperm(seed + epoch)`. Saves the model and the optimiser state (T3 continues both).
- **T3** (balanced starts): from T2, model and optimiser; 8 epochs over the same order, but in every batch two seeded
  positions are replaced by balanced-start examples (each of the 48 drawn equally often).
- **T4** (horizon conditioning, 100-800 ms): from T3's weights with a fresh optimiser; the action input gains the full
  800-ms tape and the target time (zero-initialised); 1,760 batches, each with two examples per horizon, two of the
  sixteen from balanced starts.

`--action-blind` trains the control arm that sees no future commands (its target time stays), as the original did for
every stage. The defaults reproduce the shipped lineage.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from go2nav.models.predictor import DensePredictor, HorizonConditionedPredictor
from go2nav.training.common import FeatureStore, write_lineage

REPO = Path(__file__).resolve().parents[2]
DATA = REPO/'data/training/native_predictor'
BATCH, LR, WEIGHT_DECAY, GRAD_CLIP = 16, 3e-4, 0.01, 1.0
T2_SEED, T2_EPOCHS = 2026091704, 24
T3_SEED, T3_EPOCHS, T3_SWAPS = 2026091710, 8, 2
T4_SEED, T4_STEPS = 2026091802, 1760


def load_samples(name):
    return [json.loads(line) for line in (DATA/f'{name}.jsonl').read_text().splitlines() if line.strip()]


def build_features(roots, out):
    from go2nav.models.vjepa import VJepaEncoder, preprocess

    def load(name):
        path = next((Path(r)/name for r in roots if (Path(r)/name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        return preprocess(Image.open(path))
    frames = (DATA/'frames.txt').read_text().split()
    return FeatureStore.build(out, frames, load, VJepaEncoder(), normalised=True)


class Examples:
    """Tensors for a list of samples: context (B, 3, 768, 1024), target, action, control (and horizon for T4)."""

    def __init__(self, samples, features):
        if not features.normalised:
            raise ValueError('T2-T4 train on layer-normalised features')
        self.samples, self.features = samples, features

    def batch(self, indices, device):
        rows = [self.samples[i] for i in indices]
        tokens = self.features.get([f for r in rows for f in r['frames']]).float().reshape(len(rows), 4, 768, 1024).to(device)
        action = torch.tensor([r['action'] for r in rows], dtype=torch.float32, device=device)
        control = torch.tensor([r['control'] for r in rows], dtype=torch.float32, device=device)
        horizon = torch.tensor([r.get('horizon', 5) for r in rows], device=device)
        return tokens[:, :3], tokens[:, 3], action, control, horizon


def dense_l1(prediction, target):
    return F.l1_loss(F.layer_norm(prediction.float(), (1024,)), target)


def step(model, optimiser, prediction_fn):
    optimiser.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = prediction_fn()
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite loss')
    loss.backward()
    if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)):
        raise RuntimeError('non-finite gradient')
    optimiser.step()
    return float(loss.detach())


def load_predictor(path):
    model = DensePredictor()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
    return model


def train_t2(parent, features_dir, out, *, seed=T2_SEED, epochs=T2_EPOCHS, action_blind=False, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    examples = Examples(load_samples('t2_samples'), FeatureStore(features_dir))
    model = load_predictor(parent).to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, started = [], time.time()
    mask = torch.ones(BATCH, 768, dtype=torch.bool, device=device)
    for epoch in range(epochs):
        order = torch.randperm(len(examples.samples), generator=torch.Generator().manual_seed(seed+epoch))
        total = 0.
        for offset in range(0, len(order), BATCH):
            context, target, action, control, _ = examples.batch(order[offset:offset+BATCH].tolist(), device)
            action = torch.zeros_like(action) if action_blind else action
            total += len(context)*step(model, optimiser, lambda: dense_l1(model(context, action, mask[:len(context)], control=control), target))
        history.append(dict(epoch=epoch, train_l1=total/len(order)))
        print(json.dumps(history[-1]), flush=True)
    checkpoint = out/'t2.pt'
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimiser.state_dict(), epoch=epochs-1), checkpoint)
    write_lineage(out, component='t2_native_adaptation', seed=seed, parents=[str(parent)], checkpoint=checkpoint,
                  config=dict(epochs=epochs, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, action_blind=action_blind),
                  data=dict(samples='data/training/native_predictor/t2_samples.jsonl', count=len(examples.samples)),
                  metrics=dict(history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


def t3_schedule(original, new, *, seed=T3_SEED, epochs=T3_EPOCHS):
    """Per epoch, batches of original-sample indices in `randperm(seed + epoch)` order, with T3_SWAPS seeded positions
    per batch replaced by balanced-start indices (original + i), each drawn equally often (shuffled queues)."""
    rng, queue, schedule = np.random.default_rng(seed), [], []
    for epoch in range(epochs):
        order = torch.randperm(original, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        batches = []
        for offset in range(0, original, BATCH):
            batch = order[offset:offset+BATCH]
            mixed = list(batch)
            for position in rng.choice(len(batch), size=T3_SWAPS, replace=False):
                if not queue:
                    queue = rng.permutation(new).tolist()
                mixed[int(position)] = original+queue.pop()
            batches.append(mixed)
        schedule.append(batches)
    return schedule


def train_t3(parent, features_dir, out, *, seed=T3_SEED, epochs=T3_EPOCHS, action_blind=False, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    original, new = load_samples('t2_samples'), load_samples('t3_balanced_start_samples')
    examples = Examples(original+new, FeatureStore(features_dir))
    state = torch.load(parent, map_location='cpu', weights_only=False)
    model = DensePredictor()
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimiser.load_state_dict(state['optimizer_state_dict'])
    history, started = [], time.time()
    mask = torch.ones(BATCH, 768, dtype=torch.bool, device=device)
    for epoch, batches in enumerate(t3_schedule(len(original), len(new), seed=seed, epochs=epochs)):
        total, count = 0., 0
        for indices in batches:
            context, target, action, control, _ = examples.batch(indices, device)
            action = torch.zeros_like(action) if action_blind else action
            total += len(context)*step(model, optimiser, lambda: dense_l1(model(context, action, mask[:len(context)], control=control), target))
            count += len(context)
        history.append(dict(epoch=epoch, train_l1=total/count))
        print(json.dumps(history[-1]), flush=True)
    checkpoint = out/'t3.pt'
    torch.save(dict(model_state_dict=model.state_dict(), epoch=epochs-1), checkpoint)
    write_lineage(out, component='t3_balanced_start_predictor', seed=seed, parents=[str(parent)], checkpoint=checkpoint,
                  config=dict(epochs=epochs, batch=BATCH, swaps_per_batch=T3_SWAPS, lr=LR, action_blind=action_blind),
                  data=dict(samples=['t2_samples.jsonl', 't3_balanced_start_samples.jsonl'], count=len(examples.samples)),
                  metrics=dict(history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


def t4_schedule(samples, *, seed=T4_SEED, steps=T4_STEPS):
    """Batches of 16: two examples per horizon 1-8; at two rotating horizons one of the pair is a balanced start.
    Within each (balanced start?, horizon) pool, examples come from shuffled queues; each batch is then shuffled."""
    rng, pools, queues, schedule = np.random.default_rng(seed), defaultdict(list), {}, []
    for i, s in enumerate(samples):
        pools[s['source'] == 'balanced_start', s['horizon']].append(i)
    for n in range(steps):
        batch, start_horizons = [], {n % 8+1, (n+4) % 8+1}
        for h in range(1, 9):
            for draw in range(2):
                key = (h in start_horizons and draw == 0, h)
                if not queues.get(key):
                    queues[key] = rng.permutation(pools[key]).tolist()
                batch.append(queues[key].pop())
        rng.shuffle(batch)
        schedule.append(batch)
    assert all(Counter(samples[i]['horizon'] for i in b) == {h: 2 for h in range(1, 9)} for b in schedule)
    return schedule


def train_t4(parent, features_dir, out, *, seed=T4_SEED, steps=T4_STEPS, action_blind=False, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    samples = load_samples('t4_samples')
    examples = Examples(samples, FeatureStore(features_dir))
    model = HorizonConditionedPredictor(load_predictor(parent), action_blind=action_blind).to(device).train().requires_grad_(True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, total, started = [], 0., time.time()
    mask = torch.ones(BATCH, 768, dtype=torch.bool, device=device)
    for n, indices in enumerate(t4_schedule(samples, seed=seed, steps=steps), 1):
        context, target, action, control, horizon = examples.batch(indices, device)
        total += step(model, optimiser, lambda: dense_l1(model(context, action, horizon, mask, control=control), target))
        if n % 80 == 0:
            history.append(dict(updates=n, train_l1=total/80))
            print(json.dumps(history[-1]), flush=True)
            total = 0.
    checkpoint = out/'t4.pt'
    torch.save(dict(model_state_dict=model.state_dict(), updates=steps), checkpoint)
    write_lineage(out, component='t4_horizon_dense_predictor', seed=seed, parents=[str(parent)], checkpoint=checkpoint,
                  config=dict(steps=steps, batch=BATCH, lr=LR, action_blind=action_blind),
                  data=dict(samples='t4_samples.jsonl', count=len(samples)),
                  metrics=dict(history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('features')
    f.add_argument('--recordings-root', action='append', required=True)
    f.add_argument('--out', required=True)
    for name, seed in (('t2', T2_SEED), ('t3', T3_SEED), ('t4', T4_SEED)):
        s = sub.add_parser(name)
        s.add_argument('--parent', required=True)
        s.add_argument('--features', required=True)
        s.add_argument('--out', required=True)
        s.add_argument('--seed', type=int, default=seed)
        s.add_argument('--action-blind', action='store_true')
    a = p.parse_args()
    if a.command == 'features':
        build_features(a.recordings_root, a.out)
        return
    train = dict(t2=train_t2, t3=train_t3, t4=train_t4)[a.command]
    print(train(a.parent, a.features, a.out, seed=a.seed, action_blind=a.action_blind))


if __name__ == '__main__':
    main()
