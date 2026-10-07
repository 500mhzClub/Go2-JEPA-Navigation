"""T1: train the JEPA predictor from scratch on May-corpus video (the first stage of the predictor lineage).

    python -m go2nav.training.t1_temporal features --render-root DIR --out FEATURES   # encode the 18,101 frames once
    python -m go2nav.training.t1_temporal train --features FEATURES --out DIR --seed 2026080901
    python -m go2nav.training.t1_temporal evaluate CHECKPOINT --features FEATURES

**Data** (data/training/t1_temporal/rows.jsonl): 4,397 examples from 80 May-corpus scenes (3,922 train, 475 checkpoint
selection). Each has three context frames (1 s, 0.5 s and 0 s before the action), the frames 0.5 s and 1 s after, the
two 500-ms action blocks (5 commands x (forward, yaw)) and the applied-command history (normalised with
data/training/control_norm_stats.json). Frames are encoded once by the frozen V-JEPA encoder (`features`).

**Objective** (the "rgb_rollout" cell of the original factorial): predict the frame 0.5 s ahead (e1, L1 on
layer-normalised tokens), then roll forward once: the context becomes [ctx1, ctx2, prediction], the control history gains
the first action block, and the model predicts the frame 1 s ahead (e2). Loss = e1 + mean(|[p1, p2] - [y1, y2]|),
i.e. 1.5 e1 + 0.5 e2.

**Training.** A fresh predictor initialised from the seed; AdamW (lr 3e-4, weight decay 0.01), batch 4, gradient clip
1.0, bf16 autocast; 24 epochs; the epoch-21 checkpoint is kept (fixed in advance, never selected on a metric). The batch
order of each epoch is a keyed function of (seed, epoch). The shipped T1 used seed 2026080901.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from go2nav.models.predictor import DensePredictor
from go2nav.models.vjepa import normalise
from go2nav.training.common import FeatureStore, batch_plan, control_stats, write_lineage

REPO = Path(__file__).resolve().parents[2]
ROWS = REPO/'data/training/t1_temporal/rows.jsonl'
EPOCHS, CHECKPOINT_EPOCH, BATCH, LR, WEIGHT_DECAY, GRAD_CLIP = 24, 21, 4, 3e-4, 0.01, 1.0
SEED = 2026080901  # the shipped T1's seed
FRAME_KEYS = ('ctx0', 'ctx1', 'ctx2', 'y1', 'y2')


def load_rows(split):
    return [r for r in map(json.loads, ROWS.read_text().splitlines()) if r['split'] == split]


def build_features(render_root, out):
    from go2nav.models.vjepa import VJepaEncoder, preprocess_may_corpus_frame
    frames = (REPO/'data/training/t1_temporal/frames.txt').read_text().split()
    return FeatureStore.build(out, frames, lambda name: preprocess_may_corpus_frame(Image.open(Path(render_root)/name)),
                              VJepaEncoder())


def batch(rows, features, device, mean, std):
    tokens = {k: normalise(features.get([r['frames'][k] for r in rows]).float()).to(device) for k in FRAME_KEYS}
    control = torch.tensor([r['control'] for r in rows], dtype=torch.float32, device=device).reshape(len(rows), 3, 5, 2)
    return dict(context=torch.stack([tokens['ctx0'], tokens['ctx1'], tokens['ctx2']], dim=1), y1=tokens['y1'], y2=tokens['y2'],
                a1=torch.tensor([r['a1'] for r in rows], dtype=torch.float32, device=device),
                a2=torch.tensor([r['a2'] for r in rows], dtype=torch.float32, device=device),
                control=(control-mean)/std, mask=torch.ones(len(rows), 768, dtype=torch.bool, device=device))


def losses(model, b):
    """(e1, e2, loss) for one batch: one-step prediction, then one rollout step."""
    p1 = normalise(model(b['context'], b['a1'], b['mask'], control=b['control']))
    e1 = (p1-b['y1']).abs().mean()
    window = torch.stack([b['context'][:, 1], b['context'][:, 2], p1], 1)
    # The executed action becomes the newest history slot. As in the original, it is appended in raw units, while the
    # recorded history is normalised.
    control2 = torch.cat([b['control'][:, 1:], b['a1'].reshape(-1, 1, 5, 2)], 1)
    p2 = normalise(model(window, b['a2'], b['mask'], control=control2))
    e2 = (p2-b['y2']).abs().mean()
    loss = e1+torch.cat([p1, p2], 1).sub(torch.cat([b['y1'], b['y2']], 1)).abs().mean()
    return e1, e2, loss


def train(features_dir, out, *, seed, epochs=EPOCHS, checkpoint_epoch=CHECKPOINT_EPOCH, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    features, rows = FeatureStore(features_dir), load_rows('train')
    mean, std = control_stats(device)
    model = DensePredictor.seeded(seed).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    history, checkpoint, started = [], None, time.time()
    for epoch in range(epochs):
        model.train()
        totals = dict(e1=0., e2=0., loss=0.)
        plan = batch_plan(seed, epoch, len(rows), BATCH)
        for indices in plan:
            b = batch([rows[i] for i in indices], features, device, mean, std)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                e1, e2, loss = losses(model, b)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            for k, v in dict(e1=e1, e2=e2, loss=loss).items():
                totals[k] += float(v.detach())
        history.append(dict(epoch=epoch, **{k: v/len(plan) for k, v in totals.items()}))
        print(json.dumps(history[-1]), flush=True)
        if epoch == checkpoint_epoch:
            checkpoint = out/f't1_seed{seed}_epoch{epoch}.pt'
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, seed=seed, history=list(history)), checkpoint)
    metrics = evaluate(checkpoint, features_dir, device=device)
    write_lineage(out, component='t1_temporal_predictor', seed=seed, parents=[],
                  config=dict(epochs=epochs, checkpoint_epoch=checkpoint_epoch, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY,
                              grad_clip=GRAD_CLIP, precision='bf16 autocast'),
                  data=dict(rows='data/training/t1_temporal/rows.jsonl', train_rows=len(rows)), checkpoint=checkpoint,
                  metrics=dict(selection=metrics, history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


@torch.no_grad()
def evaluate(checkpoint, features_dir, *, split='checkpoint_selection', device='cuda:0'):
    """Mean e1 and e2 on a split (no gradient)."""
    device = torch.device(device)
    features, rows = FeatureStore(features_dir), load_rows(split)
    mean, std = control_stats(device)
    model = DensePredictor().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
    model.eval()
    sums = np.zeros(2)
    for start in range(0, len(rows), 8):
        part = rows[start:start+8]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            e1, e2, _ = losses(model, batch(part, features, device, mean, std))
        sums += np.array([float(e1), float(e2)])*len(part)
    return dict(split=split, rows=len(rows), e1=sums[0]/len(rows), e2=sums[1]/len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('features')
    f.add_argument('--render-root', required=True)
    f.add_argument('--out', required=True)
    t = sub.add_parser('train')
    t.add_argument('--features', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, default=SEED)
    e = sub.add_parser('evaluate')
    e.add_argument('checkpoint')
    e.add_argument('--features', required=True)
    a = p.parse_args()
    if a.command == 'features':
        build_features(a.render_root, a.out)
    elif a.command == 'train':
        print(train(a.features, a.out, seed=a.seed))
    else:
        print(json.dumps(evaluate(a.checkpoint, a.features)))


if __name__ == '__main__':
    main()
