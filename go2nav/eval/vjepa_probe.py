"""Gate L3's reference: the same true-future probe on C3's frozen V-JEPA 2.1 encoder.

    python -m go2nav.eval.vjepa_probe --cache outputs/features/t7_cache --base-readout outputs/training/t5b/readout.pt \
        --recordings-root outputs/recordings --out DIR

L3 (`go2nav.training.lewm_decoder.probe_gate`) asks what LeWM's encoder holds: C5's decoder, grown from the base
readout's `true` head and fitted with T7's recipe on the true future frame's z, scored on eval_onpolicy at 800 ms. Here
the same question for V-JEPA, with C3's side of every choice:
- `fit_s<seed>`: C3's decoder (`MotionDecoder` grown from T5b; past_frames, proj 64, hidden 896, depth 1, as deployed)
  fitted with T7's recipe (decoder_fix, 3,520 updates of 64, T7's three seeds) on the true future frame's pooled
  features instead of the T4 predictor's prediction;
- `t5b`: T5b as is on (current, true future), the counterpart of the base readout's `true` head before T7's fit.
Scores as L3: median XY error and RMS yaw error averaged over the six movement types (eval_onpolicy at 800 ms; 500 ms
also reported). `summary.json` gives each and the median of the three fits.

Stages: `future/` encodes the true future frames the T7 cache lacks (pooled, layer-normalised, as the cache); each fit
folder is written atomically; a rerun skips what is finished.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from go2nav.models.motion import MotionDecoder, load_motion_readout
from go2nav.training.common import FeatureStore, write_lineage
from go2nav.training.decoder import (BATCH, CATEGORIES, DECODER, GRAD_CLIP, LR, RECIPES, UPDATES, WEIGHT_DECAY,
                                     DecoderCache, load_contexts, metrics, sampler)
from go2nav.training.lewm_decoder import SEEDS, neighbours

RECIPE, EVAL_SET, EVAL_HORIZONS = 'decoder_fix', 'eval_onpolicy', (5, 8)


def future_name(context, h):
    return neighbours(context['frames'][2], future=8)[h-1]


def needed(contexts, recipe=RECIPE):
    """The (context, horizon) pairs a fit draws from and the evaluation reads."""
    groups = {key.split('/')[0] for key in RECIPES[recipe]}
    for i, c in enumerate(contexts):
        if c['set'] == 'train' and ('*' in groups or c['group'] in groups):
            yield from ((i, h) for h in c['horizons'])
        elif c['set'] == EVAL_SET:
            yield from ((i, h) for h in EVAL_HORIZONS)


class TrueFutures:
    """Pooled features of a context's true future frame: from the T7 cache where it holds the frame, else `future/`."""

    def __init__(self, cache, store):
        self.cache, self.store = cache, store

    def get(self, contexts, idx, horizons, device):
        out = []
        for i, h in zip(idx, horizons):
            name = future_name(contexts[i], int(h))
            row = self.cache.frame_row.get(name)
            out.append(self.cache.frames[row] if row is not None else self.store.tokens[self.store.index[name]])
        return torch.from_numpy(np.stack(out)).to(device).float()


def encode_futures(cache, contexts, roots, out, *, batch=8, device='cuda:0'):
    out = Path(out)
    if (out/'.done').exists():
        return FeatureStore(out)
    from go2nav.models.vjepa import VJepaEncoder, preprocess
    names = sorted({future_name(contexts[i], h) for i, h in needed(contexts)} - set(cache.frame_row))
    roots = [Path(r) for r in roots]

    def load(name):
        path = next((r/name for r in roots if (r/name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        return preprocess(Image.open(path))
    shutil.rmtree(out, ignore_errors=True)
    started = time.time()
    store = FeatureStore.build(out, names, load, VJepaEncoder(device), batch=batch, pooled=True)
    print(json.dumps(dict(future_frames=len(names), wall_s=round(time.time()-started))), flush=True)
    (out/'.done').touch()
    return store


def update(model, optimiser, prediction, y):
    optimiser.zero_grad(set_to_none=True)
    loss = ((prediction-(y-model.target_mean)/model.target_scale)**2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(cache, futures, model, past_frames, device):
    model.eval()
    contexts = cache.contexts
    idx_all = [i for i, c in enumerate(contexts) if c['set'] == EVAL_SET]
    cats = np.asarray([contexts[i]['category'] for i in idx_all])
    report = {}
    for h in EVAL_HORIZONS:
        preds = []
        for k in range(0, len(idx_all), 256):
            idx = idx_all[k:k+256]
            feats = cache.tensors(idx, np.full(len(idx), h), device)[0]
            true = futures.get(contexts, idx, np.full(len(idx), h), device)
            args = (feats[:, 2], true, feats[:, 1], feats[:, 0]) if past_frames else (feats[:, 2], true)
            preds.append(model(*args).cpu().numpy())
        targets = np.stack([np.asarray(contexts[i]['targets'], np.float32)[h-1] for i in idx_all])
        report[f'{EVAL_SET}@{100*h}ms'] = metrics(np.concatenate(preds), targets, cats)
    model.train()
    return report


def gate(report, horizon=8):
    """L3's numbers: median XY (mm) and RMS yaw (deg) averaged over the movement types."""
    cells = report[f'{EVAL_SET}@{100*horizon}ms']
    xy = float(np.mean([cells[c]['median_xy_mm'] for c in CATEGORIES if c in cells]))
    yaw = float(np.mean([cells[c]['rmse_yaw_deg'] for c in CATEGORIES if c in cells]))
    return dict(median_xy_mm=xy, rmse_yaw_deg=yaw, l3=xy <= 10. and yaw <= 1.5)


def fit(cache, futures, base_readout_path, out, *, seed, updates=UPDATES, device='cuda:0'):
    out = Path(out)
    if (out/'.done').exists():
        return json.loads((out/'evaluation.json').read_text())
    partial = Path(str(out)+'.tmp')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started = time.time()
    draw, cells = sampler(cache.contexts, RECIPES[RECIPE], np.random.default_rng(seed))
    torch.manual_seed(seed)
    probe = MotionDecoder(load_motion_readout(base_readout_path), **DECODER).to(device).train()
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for n in range(updates):
        idx, h = draw(BATCH)
        feats, _, _, _, y = cache.tensors(idx, h, device)
        true = futures.get(cache.contexts, idx, h, device)
        loss = update(probe, opt, probe.normalized(feats[:, 2], true, feats[:, 1], feats[:, 0]), y)
        if (n+1) % 100 == 0:
            history.append(dict(step=n+1, loss=loss))
    report = evaluate(cache, futures, probe, True, device)
    result = dict(seed=seed, recipe=RECIPE, probe=gate(report), probe_500ms=gate(report, 5), eval=report)
    torch.save(dict(readout=probe.state_dict(), readout_config=probe.config, recipe=RECIPE), partial/f'probe_s{seed}.pt')
    (partial/'evaluation.json').write_text(json.dumps(result, indent=1)+'\n')
    write_lineage(partial, component='vjepa_true_future_probe', seed=seed,
                  parents=[str(cache.folder/'meta.json'), str(base_readout_path)],
                  config=dict(recipe=RECIPE, mix=RECIPES[RECIPE], updates=updates, lr=LR, decoder=DECODER),
                  data=dict(contexts='data/training/decoder/contexts.jsonl', cells=cells),
                  checkpoint=partial/f'probe_s{seed}.pt',
                  metrics=dict(probe=result['probe'], history=history, wall_s=round(time.time()-started, 1)))
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    (out/'.done').touch()
    print(json.dumps(dict(seed=seed, probe=result['probe'], wall_s=round(time.time()-started))), flush=True)
    return result


def run(cache_folder, base_readout, roots, out, *, encode_device='cuda:0', encode_batch=8, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts()
    cache = DecoderCache(cache_folder, contexts)
    futures = TrueFutures(cache, encode_futures(cache, contexts, roots, out/'future', batch=encode_batch,
                                                device=encode_device))
    t5b = load_motion_readout(base_readout).to(device)
    summary = dict(t5b=gate(evaluate(cache, futures, t5b, False, device)))
    summary['t5b_500ms'] = gate(evaluate(cache, futures, t5b, False, device), 5)
    fits = {seed: fit(cache, futures, base_readout, out/f'fit_s{seed}', seed=seed, device=device) for seed in SEEDS}
    summary['fits'] = {str(s): r['probe'] for s, r in fits.items()}
    xy = sorted((r['probe']['median_xy_mm'], s) for s, r in fits.items())
    summary['median_fit'] = dict(seed=xy[1][1], **fits[xy[1][1]]['probe'])
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True, type=Path, help='the T7 decoder cache (go2nav.training.decoder cache)')
    p.add_argument('--base-readout', required=True, type=Path, help='T5b, the base of C3\'s decoder')
    p.add_argument('--recordings-root', action='append', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--encode-device', default='cuda:0')
    p.add_argument('--encode-batch', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    print(json.dumps(run(a.cache, a.base_readout, a.recordings_root, a.out, encode_device=a.encode_device,
                         encode_batch=a.encode_batch, device=a.device), indent=1))


if __name__ == '__main__':
    main()
