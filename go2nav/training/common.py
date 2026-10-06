"""Shared training utilities: keyed random streams, control normalisation, feature stores and lineage manifests."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CONTROL_STATS = REPO/'data/training/control_norm_stats.json'


def stream(seed, purpose, *keys):
    """A named, stateless torch generator: identical for identical (seed, purpose, keys), independent of anything else."""
    material = '|'.join([str(seed), purpose]+[str(k) for k in keys]).encode()
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') % (2**63-1))


def batch_plan(seed, epoch, count, batch):
    """The row order of one epoch, split into batches: a pure function of (seed, epoch)."""
    order = torch.randperm(count, generator=stream(seed, 'data_order', epoch)).tolist()
    return [order[i:i+batch] for i in range(0, count, batch)]


def control_stats(device=None):
    """Mean and standard deviation of applied commands (forward, yaw), fitted on T1 training data and used by every
    later stage and at run time."""
    stats = json.loads(CONTROL_STATS.read_text())
    return tuple(torch.as_tensor(stats[k], dtype=torch.float32, device=device) for k in ('control_mean', 'control_std'))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream_:
        for block in iter(lambda: stream_.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', 'go2nav', 'configs'], cwd=REPO, capture_output=True, text=True).stdout
    return dict(commit=commit or None, uncommitted_changes=bool(dirty.strip()))


def write_lineage(out, *, component, seed, config, parents, data, checkpoint, metrics=None):
    """lineage.json beside a checkpoint: what it was trained from, with what code, seed and data."""
    record = dict(component=component, seed=seed, config=config, parents=parents, data=data,
                  checkpoint=dict(file=Path(checkpoint).name, sha256=sha256_file(checkpoint)), code=code_identity(),
                  packages={p: metadata.version(p) for p in ('torch', 'numpy')}, metrics=metrics or {})
    (Path(out)/'lineage.json').write_text(json.dumps(record, indent=1)+'\n')
    return record


class FeatureStore:
    """Frozen V-JEPA tokens of a set of frames: features.f16 (N, 768, 1024, float16) + frames.json (names) + meta.json.

    `normalised` (meta.json): tokens were layer-normalised before conversion to float16 (the T2-T5 stages' format) rather
    than stored raw (the T1 format). `pooled`: normalised tokens pooled to the 12 x 16 grid (192 tokens; the T5 format).
    Each stage reads the format it was trained with.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.paths = json.loads((self.folder/'frames.json').read_text())
        meta = self.folder/'meta.json'
        meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.normalised, self.pooled = meta.get('normalised', False), meta.get('pooled', False)
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.tokens = np.memmap(self.folder/'features.f16', dtype=np.float16, mode='r',
                                shape=(len(self.paths), 192 if self.pooled else 768, 1024))

    def get(self, paths):
        return torch.from_numpy(np.stack([self.tokens[self.index[p]] for p in paths]))

    @staticmethod
    @torch.no_grad()
    def build(folder, frame_paths, load_pixels, encoder, batch=16, normalised=False, pooled=False):
        """Encode `frame_paths` (relative names) with `encoder`; `load_pixels(name)` returns the preprocessed frame."""
        from torch.nn import functional as F
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'frames.json').write_text(json.dumps(list(frame_paths))+'\n')
        from go2nav.models.vjepa import pool_tokens
        normalised = normalised or pooled
        (folder/'meta.json').write_text(json.dumps(dict(normalised=normalised, pooled=pooled))+'\n')
        tokens = np.memmap(folder/'features.f16', dtype=np.float16, mode='w+',
                           shape=(len(frame_paths), 192 if pooled else 768, 1024))
        device = next(encoder.module.parameters()).device
        for start in range(0, len(frame_paths), batch):
            names = frame_paths[start:start+batch]
            pixels = torch.stack([load_pixels(n) for n in names]).to(device)
            out = encoder.tokens(pixels)
            out = F.layer_norm(out.float(), (1024,)) if normalised else out
            out = pool_tokens(out) if pooled else out
            tokens[start:start+len(names)] = out.half().cpu().numpy()
        tokens.flush()
        return FeatureStore(folder)
