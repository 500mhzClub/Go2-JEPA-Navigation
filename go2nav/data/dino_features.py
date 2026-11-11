"""Frozen DINOv2 patch features of corpus v2's camera frames (docs/dynamic/PLAN.md P23 items 1-2).

    python -m go2nav.data.dino_features extract --corpus CORPUS --out OUT [--train 600] [--val 100] [--seed S]

The encoder is DINOv2 ViT-S/14 (Oquab et al. 2023), loaded from the local torch hub cache and frozen: the frame
(224 x 168, the corpus's primary-camera RGB) normalised with ImageNet's statistics, and the last layer's normalised
patch tokens kept: 16 x 12 = 192 tokens of 384 a frame, row-major from the top-left, float16.

Units: `train` units of `lewm_train` and `val` of `lewm_val` drawn at random (`seed`) from those with at least one
walker, so the two splits share no layout. Per unit, OUT/<role>/<unit>.npy (frames, 192, 384) float16, written
atomically (a rerun skips what is done), and OUT/index.json: the units, their frame counts and the encoder.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

HUB = Path(os.path.expanduser('~/.cache/torch/hub'))
ENCODER = 'dinov2_vits14'
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
GRID = (12, 16)            # rows, columns of patches on a 168 x 224 frame
BATCH = 256


def load_encoder(device):
    model = torch.hub.load(str(HUB/'facebookresearch_dinov2_main'), ENCODER, source='local', pretrained=False)
    model.load_state_dict(torch.load(HUB/'checkpoints'/f'{ENCODER}_pretrain.pth', map_location='cpu'))
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode(model, frames, device):
    """Frames (N, 168, 224, 3) uint8 to patch tokens (N, 192, 384) float16."""
    out = []
    mean = torch.tensor(MEAN, device=device)[:, None, None]
    std = torch.tensor(STD, device=device)[:, None, None]
    for i in range(0, len(frames), BATCH):
        x = torch.as_tensor(np.asarray(frames[i:i+BATCH]), device=device).permute(0, 3, 1, 2).float()/255.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            tokens = model.forward_features((x-mean)/std)['x_norm_patchtokens']
        out.append(tokens.half().cpu().numpy())
    return np.concatenate(out)


def walkers_in(unit):
    with np.load(unit/'dynamic.npz') as d:
        return d['obstacles_xy'].shape[1] > 0


def select(corpus, role, count, seed):
    index = json.loads((Path(corpus)/'index_257.json').read_text())
    paths = sorted(u['path'] for u in index['units'] if u['role'] == role)
    eligible = [p for p in paths if walkers_in(Path(corpus)/p)]
    rng = np.random.default_rng([seed, role == 'lewm_val'])
    return sorted(rng.choice(eligible, min(count, len(eligible)), replace=False).tolist())


def extract(corpus, out, *, train=600, val=100, seed=2026111103, device='cuda:0'):
    corpus, out, device = Path(corpus), Path(out), torch.device(device)
    model = load_encoder(device)
    units = {'lewm_train': select(corpus, 'lewm_train', train, seed), 'lewm_val': select(corpus, 'lewm_val', val, seed)}
    frames = {}
    for role, paths in units.items():
        (out/role).mkdir(parents=True, exist_ok=True)
        for path in paths:
            name = Path(path).name
            target = out/role/f'{name}.npy'
            source = np.load(corpus/path/'frames.npy', mmap_mode='r')
            frames[path] = int(len(source))
            if target.exists():
                continue
            tokens = encode(model, source, device)
            partial = target.with_name(target.stem+'.tmp.npy')
            np.save(partial, tokens)
            partial.rename(target)
    index = dict(schema='go2nav_dino_features.v1', encoder=ENCODER, grid=GRID, dims=384, corpus=str(corpus), seed=seed,
                 units=units, frames=frames)
    (out/'index.json').write_text(json.dumps(index, indent=1)+'\n')
    return {role: (len(paths), sum(frames[p] for p in paths)) for role, paths in units.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--corpus', required=True, type=Path)
    e.add_argument('--out', required=True, type=Path)
    e.add_argument('--train', type=int, default=600)
    e.add_argument('--val', type=int, default=100)
    e.add_argument('--seed', type=int, default=2026111103)
    a = p.parse_args()
    print(json.dumps(extract(a.corpus, a.out, train=a.train, val=a.val, seed=a.seed)))


if __name__ == '__main__':
    main()
