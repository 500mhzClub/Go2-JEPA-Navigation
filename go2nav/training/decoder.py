"""T7/T8: fit C3's motion decoder and the matched C4 on identical data, and evaluate them.

    python -m go2nav.training.decoder cache --predictor T4.pt --recordings-root DIR [...] --out CACHE
    python -m go2nav.training.decoder fit --cache CACHE --base-readout T5a.pt --out DIR --seed 2026093011 [--recipe stage2]
    python -m go2nav.training.decoder evaluate --cache CACHE --base-readout T5a.pt --checkpoint FIT.pt
    python -m go2nav.training.decoder choose FIT_DIR [FIT_DIR ...]

**Contexts** (data/training/decoder/contexts.jsonl): 43,265 decisions with three frames (1 s, 0.5 s, 0 s before),
the executed command tape, the applied-command history and the true motion at 100-800 ms.
- Training groups: `old` (original and full-heading recordings), `maze` (maze-view recordings), `onpolicy` (C1's own
  closed-loop decisions in the C3-v3 round), and for stage 2 `patch`/`patch_off` (decisions near/away from slip strips).
- Evaluation sets: eval_onpolicy (held-out closed-loop C1 decisions), eval_fresh_c3 (C3's own decisions),
  eval_offline (rest/turn recordings), eval_transfer, and for stage 2 eval_patch.

**Cache** (`cache`): per frame, pooled layer-normalised V-JEPA features; per (context, horizon) pair, the JEPA
predictor's predicted features for the executed tape. The predicted features depend on the predictor (T4), so a
retrained T4 needs its own cache.

**Fit** (`fit`): both models see the same stream of 64-example batches drawn by the recipe's mix of (group, movement
type) cells, each example at one of its horizons. MSE in normalised motion units; AdamW lr 3e-4, gradient clip 1.0
(C3 decoder weight decay 1e-4; C4 1e-4); 3,520 updates; no autocast.
- C3 decoder: `MotionDecoder` grown from the base readout (T5a): past_frames inputs, proj 64, hidden 896, depth 1.
- C4: `DirectMotionPredictor`, fresh, with the base readout's motion normalisation.
Recipes: `decoder_fix` (old 32, maze 16, on-policy 16 split evenly over movement types: the deployed fit) and
`stage2` (old 32, maze 16, on-policy 8, patch 6, patch_off 2).

**Choice** (`choose`): the existing rule: score = the C3 decoder's median 800-ms XY error on eval_onpolicy averaged
over the six movement types; of three seeds, deploy the median one.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from go2nav.models.motion import DirectMotionPredictor, MotionDecoder, load_motion_readout
from go2nav.models.predictor import load_horizon_predictor
from go2nav.models.vjepa import normalise, pool_tokens
from go2nav.training.common import control_stats, sha256_file, write_lineage

REPO = Path(__file__).resolve().parents[2]
CONTEXTS = REPO/'data/training/decoder/contexts.jsonl'
CATEGORIES = ('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch')
EVAL_SETS = {'eval_onpolicy': (5, 8), 'eval_fresh_c3': (8,), 'eval_offline': (5, 8), 'eval_transfer': (5, 7), 'eval_patch': (5, 8)}
EDGE_BINS = ('approach', 'entry', 'on_patch', 'exit', 'off_patch')
UPDATES, LR, WEIGHT_DECAY, GRAD_CLIP, BATCH = 3520, 3e-4, 1e-4, 1.0, 64
DECODER = dict(past_frames=True, proj=64, hidden=896, depth=1)
RECIPES = {
    'decoder_fix': {'old/*': 32, 'maze/*': 16, **{f'onpolicy/{c}': w for c, w in zip(CATEGORIES, (2.67,)*5+(2.65,))}},
    'stage2': {'old/*': 32, 'maze/*': 16, **{f'onpolicy/{c}': w for c, w in zip(CATEGORIES, (1.33,)*5+(1.35,))},
               'patch/*': 6, 'patch_off/*': 2},
}


def load_contexts():
    """All contexts. The decoder-fix contexts come first and the stage-2 ones after; a recipe's mix draws only its own
    training groups, so one cache serves both recipes with consistent indices."""
    return [json.loads(line) for line in CONTEXTS.read_text().splitlines() if line.strip()]


# ---- cache --------------------------------------------------------------------------------------------------------
class DecoderCache:
    """frames.f16 (frames, 192, 1024), pred.f16 (pairs, 192, 1024), frames.json, pairs.json, meta.json."""

    def __init__(self, folder, contexts):
        self.folder, self.contexts = Path(folder), contexts
        frames = json.loads((self.folder/'frames.json').read_text())
        pairs = json.loads((self.folder/'pairs.json').read_text())
        self.frame_row = {p: i for i, p in enumerate(frames)}
        self.pair_row = {(i, h): k for k, (i, h) in enumerate(pairs)}
        self.frames = np.memmap(self.folder/'frames.f16', dtype=np.float16, mode='r', shape=(len(frames), 192, 1024))
        self.pred = np.memmap(self.folder/'pred.f16', dtype=np.float16, mode='r', shape=(len(pairs), 192, 1024))
        self.mean, self.std = control_stats()

    def control(self, idx):
        past = torch.tensor(np.stack([np.asarray(self.contexts[i]['past'], np.float32)[:, [0, 2]].reshape(3, 5, 2) for i in idx]))
        return (past-self.mean)/self.std

    def tensors(self, idx, horizons, device):
        rows = [self.frame_row[f] for i in idx for f in self.contexts[i]['frames']]
        feats = torch.from_numpy(np.asarray(self.frames[rows])).reshape(len(idx), 3, 192, 1024).to(device).float()
        pred = torch.from_numpy(np.stack([self.pred[self.pair_row[(i, int(h))]] for i, h in zip(idx, horizons)])).to(device).float()
        tape = torch.from_numpy(np.stack([np.asarray(self.contexts[i]['tape'], np.float32)[:, [0, 2]] for i in idx])).to(device)
        y = torch.from_numpy(np.stack([np.asarray(self.contexts[i]['targets'], np.float32)[int(h)-1] for i, h in zip(idx, horizons)])).to(device)
        return feats, pred, tape, self.control(idx).to(device), y

    @staticmethod
    @torch.inference_mode()
    def build(contexts, roots, predictor_checkpoint, out, batch=16, device='cuda:0'):
        """Encode every frame (pooled) and predict every (context, horizon) pair with the given T4 predictor."""
        from go2nav.models.vjepa import VJepaEncoder, preprocess
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        frames = sorted({f for c in contexts for f in c['frames']})
        pairs = [(i, h) for i, c in enumerate(contexts) for h in c['horizons']]
        frame_row, pair_row = {p: i for i, p in enumerate(frames)}, {p: k for k, p in enumerate(pairs)}
        frame_store = np.memmap(out/'frames.f16', dtype=np.float16, mode='w+', shape=(len(frames), 192, 1024))
        pred_store = np.memmap(out/'pred.f16', dtype=np.float16, mode='w+', shape=(len(pairs), 192, 1024))
        encoder, predictor = VJepaEncoder(device), load_horizon_predictor(predictor_checkpoint).to(device)
        mean, std = control_stats(device)

        def load(name):
            path = next((Path(r)/name for r in roots if (Path(r)/name).exists()), None)
            if path is None:
                raise FileNotFoundError(name)
            return preprocess(Image.open(path))
        by_recording = defaultdict(list)  # contexts of one recording share frames: encode each once
        for i, c in enumerate(contexts):
            by_recording[c['frames'][2].rsplit('/', 1)[0]].append(i)
        started, done = time.time(), 0
        for indices in by_recording.values():
            indices.sort(key=lambda i: contexts[i]['frames'][2])
            tokens = {}
            for start in range(0, len(indices), batch):
                part = indices[start:start+batch]
                missing = sorted({f for i in part for f in contexts[i]['frames']} - set(tokens))
                for m in range(0, len(missing), batch):
                    names = missing[m:m+batch]
                    full = normalise(encoder.tokens(torch.stack([load(n) for n in names]).to(device)))
                    for name, t, p in zip(names, full, pool_tokens(full).half().cpu().numpy()):
                        tokens[name] = t
                        frame_store[frame_row[name]] = p
                keep = {f for i in indices[start+batch:start+2*batch] for f in contexts[i]['frames']}
                rows, ctx, act, ctl, hor = [], [], [], [], []
                for i in part:
                    c = contexts[i]
                    context = torch.stack([tokens[f] for f in c['frames']])
                    control = (torch.tensor(np.asarray(c['past'], np.float32)[:, [0, 2]].reshape(3, 5, 2), device=device)-mean)/std
                    action = torch.tensor(np.asarray(c['tape'], np.float32)[:, [0, 2]], device=device)
                    for h in c['horizons']:
                        rows.append(pair_row[(i, h)])
                        ctx.append(context)
                        act.append(action)
                        ctl.append(control)
                        hor.append(h)
                predicted = predictor(torch.stack(ctx), torch.stack(act), torch.tensor(hor, device=device),
                                      torch.ones(len(rows), 768, dtype=torch.bool, device=device), control=torch.stack(ctl))
                for r, p in zip(rows, pool_tokens(normalise(predicted)).half().cpu().numpy()):
                    pred_store[r] = p
                tokens = {f: t for f, t in tokens.items() if f in keep}
                done += len(part)
                if done % 2000 < len(part):
                    print(json.dumps(dict(contexts=done, total=len(contexts), wall_s=round(time.time()-started))), flush=True)
        frame_store.flush()
        pred_store.flush()
        (out/'frames.json').write_text(json.dumps(frames)+'\n')
        (out/'pairs.json').write_text(json.dumps(pairs)+'\n')
        (out/'meta.json').write_text(json.dumps(dict(predictor=str(predictor_checkpoint), predictor_sha256=sha256_file(predictor_checkpoint),
                                                     contexts=len(contexts), wall_s=round(time.time()-started)))+'\n')
        return DecoderCache(out, contexts)


# ---- fit ----------------------------------------------------------------------------------------------------------
def sampler(contexts, mix, rng):
    """draw(n) -> (context indices, horizons): a multinomial over the mix's cells, uniform within each cell."""
    cells = defaultdict(list)
    for i, c in enumerate(contexts):
        if c['set'] == 'train':
            cells[(c['group'], c['category'])].append(i)
    plan = []
    for key, share in mix.items():
        g, k = key.split('/')
        members = [i for (gg, kk), v in cells.items() if g in ('*', gg) and k in ('*', kk) for i in v]
        if members and share > 0:
            plan.append((share, np.asarray(members)))
    total = sum(s for s, _ in plan)

    def draw(n):
        counts = rng.multinomial(n, [s/total for s, _ in plan])
        idx = np.concatenate([rng.choice(m, size=k) for (_, m), k in zip(plan, counts) if k])
        return idx, np.asarray([rng.choice(contexts[i]['horizons']) for i in idx])
    return draw, {f'{g}/{k}': len(v) for (g, k), v in cells.items()}


def build_models(base_readout_path, seed, device):
    """The C3 decoder grown from the base readout, and a fresh C4: seed, then construct the base readout, the decoder and
    C4 in that order (the original order, so a seed gives the original initialisation)."""
    torch.manual_seed(seed)
    base_readout = load_motion_readout(base_readout_path)
    decoder = MotionDecoder(base_readout, **DECODER).to(device).train().requires_grad_(True)
    c4 = DirectMotionPredictor(base_readout.target_mean, base_readout.target_scale).to(device).train()
    return decoder, c4


def update(model, optimiser, prediction, y):
    optimiser.zero_grad(set_to_none=True)
    loss = ((prediction-(y-model.target_mean)/model.target_scale)**2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()
    return float(loss.detach())


def fit(cache, base_readout_path, out, *, seed, recipe='decoder_fix', updates=UPDATES, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    draw, cells = sampler(cache.contexts, RECIPES[recipe], np.random.default_rng(seed))
    decoder, c4 = build_models(base_readout_path, seed, device)
    opt3 = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt4 = torch.optim.AdamW(c4.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for n in range(updates):
        idx, h = draw(BATCH)
        feats, pred, tape, control, y = cache.tensors(idx, h, device)
        loss3 = update(decoder, opt3, decoder.normalized(feats[:, 2], pred, feats[:, 1], feats[:, 0], control), y)
        loss4 = update(c4, opt4, c4.normalized(feats, control, tape, torch.from_numpy(h).to(device)), y)
        if (n+1) % 100 == 0:
            history.append(dict(step=n+1, c3_loss=loss3, c4_loss=loss4))
    report = evaluate(cache, decoder, c4, device)
    checkpoint = out/f'fit_{recipe}_s{seed}.pt'
    torch.save(dict(readout=decoder.state_dict(), readout_config=decoder.config, c4=c4.state_dict(), variant='past_frames',
                    recipe=recipe, mix=RECIPES[recipe]), checkpoint)
    (out/'evaluation.json').write_text(json.dumps(dict(seed=seed, recipe=recipe, score_mm=score(report), eval=report), indent=1)+'\n')
    write_lineage(out, component=f't7_decoder_and_c4_{recipe}', seed=seed, parents=[str(cache.folder/'meta.json')],
                  config=dict(recipe=recipe, mix=RECIPES[recipe], updates=updates, lr=LR, decoder=DECODER),
                  data=dict(contexts='data/training/decoder/contexts.jsonl', cells=cells), checkpoint=checkpoint,
                  metrics=dict(score_mm=score(report), history=history, wall_s=round(time.time()-started, 1)))
    return checkpoint


# ---- evaluation ---------------------------------------------------------------------------------------------------
def metrics(p, t, cats):
    """Per movement type: n, median predicted/true translation ratio, median and RMS XY error (mm), RMS yaw error (deg),
    median excess translation and median true translation (mm)."""
    out = {}
    for c in ('all',)+CATEGORIES:
        m = (np.ones(len(t), bool) if c == 'all' else cats == c) & np.isfinite(t).all(axis=1)
        if not m.any():
            continue
        pt, tt = np.linalg.norm(p[m, :2], axis=1), np.linalg.norm(t[m, :2], axis=1)
        e = np.linalg.norm(p[m, :2]-t[m, :2], axis=1)
        yaw = np.degrees(np.arctan2(np.sin(p[m, 2]-t[m, 2]), np.cos(p[m, 2]-t[m, 2])))
        moving = tt >= .010
        out[c] = dict(n=int(m.sum()), median_ratio=float(np.median(pt[moving]/tt[moving])) if moving.any() else None,
                      median_xy_mm=float(np.median(e))*1000, rmse_xy_mm=float(np.sqrt(np.mean(e**2)))*1000,
                      rmse_yaw_deg=float(np.sqrt(np.mean(yaw**2))), median_excess_mm=float(np.median(pt-tt))*1000,
                      median_true_mm=float(np.median(tt))*1000)
    return out


@torch.no_grad()
def evaluate(cache, decoder, c4, device='cuda:0'):
    decoder.eval()
    c4.eval()
    report = {}
    for name, horizons in EVAL_SETS.items():
        idx_all = [i for i, c in enumerate(cache.contexts) if c['set'] == name]
        if not idx_all:
            continue
        cats = np.asarray([cache.contexts[i]['category'] for i in idx_all])
        bins = np.asarray([cache.contexts[i]['extra'].get('edge_bin', '') for i in idx_all])
        for h in horizons:
            preds = {'C3': [], 'C4': []}
            for k in range(0, len(idx_all), 256):
                idx = idx_all[k:k+256]
                feats, pred, tape, control, _ = cache.tensors(idx, np.full(len(idx), h), device)
                preds['C3'].append(decoder(feats[:, 2], pred, feats[:, 1], feats[:, 0], control).cpu().numpy())
                preds['C4'].append(c4(feats, control, tape, torch.full((len(idx),), h, device=device)).cpu().numpy())
            true = np.stack([np.asarray(cache.contexts[i]['targets'], np.float32)[h-1] for i in idx_all])
            for model, chunks in preds.items():
                p = np.concatenate(chunks)
                key = f'{name}@{100*h}ms/{model}'
                report[key] = metrics(p, true, cats)
                if name == 'eval_patch':
                    for b in EDGE_BINS:
                        if (bins == b).any():
                            report[key][f'bin:{b}'] = metrics(p[bins == b], true[bins == b], cats[bins == b])['all']
    decoder.train()
    c4.train()
    return report


def score(report, model='C3'):
    """The selection score: median 800-ms XY error on eval_onpolicy, averaged over the six movement types (mm)."""
    cells = report['eval_onpolicy@800ms/'+model]
    return float(np.mean([cells[c]['median_xy_mm'] for c in CATEGORIES if c in cells]))


def choose(fit_dirs):
    """The median-scoring fit of several seeds (the deployment rule)."""
    scored = sorted((json.loads((Path(d)/'evaluation.json').read_text())['score_mm'], str(d)) for d in fit_dirs)
    return scored[len(scored)//2], scored


def load_fit(checkpoint, base_readout, device='cuda:0'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    decoder = MotionDecoder(base_readout, **DECODER)
    decoder.load_state_dict(state['readout'])
    c4 = DirectMotionPredictor(base_readout.target_mean, base_readout.target_scale)
    c4.load_state_dict(state['c4'])
    return decoder.to(device), c4.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--predictor', required=True)
    c.add_argument('--recordings-root', action='append', required=True)
    c.add_argument('--out', required=True)
    for name in ('fit', 'evaluate'):
        s = sub.add_parser(name)
        s.add_argument('--cache', required=True)
        s.add_argument('--base-readout', required=True)
        s.add_argument('--recipe', choices=sorted(RECIPES), default='decoder_fix')
        if name == 'fit':
            s.add_argument('--out', required=True)
            s.add_argument('--seed', type=int, required=True)
        else:
            s.add_argument('--checkpoint', required=True)
    ch = sub.add_parser('choose')
    ch.add_argument('fit_dirs', nargs='+')
    a = p.parse_args()
    if a.command == 'cache':
        DecoderCache.build(load_contexts(), a.recordings_root, a.predictor, a.out)
    elif a.command == 'choose':
        print(json.dumps(choose(a.fit_dirs), indent=1))
    else:
        cache = DecoderCache(a.cache, load_contexts())
        if a.command == 'fit':
            print(fit(cache, a.base_readout, a.out, seed=a.seed, recipe=a.recipe))
        else:
            decoder, c4 = load_fit(a.checkpoint, load_motion_readout(a.base_readout))
            report = evaluate(cache, decoder, c4)
            print(json.dumps(dict(score_mm=score(report), eval=report)))


if __name__ == '__main__':
    main()
