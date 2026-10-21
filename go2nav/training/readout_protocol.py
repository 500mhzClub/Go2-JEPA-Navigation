"""The shared readout protocol: one fixed recipe that turns any latent world model into a motion forecaster, so that
arms differ in their representation and predictor only (docs/lewm/STATUS.md, readout protocol).

    python -m go2nav.training.readout_protocol windows --corpus ROOT --out PROTOCOL.json
    python -m go2nav.training.readout_protocol lewm --protocol PROTOCOL.json --corpus ROOT --world-model LEWM.pt --out DIR
    python -m go2nav.training.readout_protocol vjepa-cache --protocol PROTOCOL.json --corpus ROOT --fullres FULL_ROOT --out CACHE
    python -m go2nav.training.readout_protocol vjepa --protocol PROTOCOL.json --cache CACHE --out DIR [--c3-init fresh --c4 C4.pt]

Why: C5's decoder fitted with T7's recipe (3,520 updates of 64 on 23k on-policy decisions) forecast in-place turns near
walls 2-4 times too far and deadlocked; a readout trained on LeWM's corpus and read alone fixed it (L4 5.0 mm, 71/100
round trips). C3's decoder is T5b plus that same T7 fit, so comparing C3 with the fixed C5 would compare recipes as
well as representations. The protocol gives every arm the same labelled data and the same training.

**Data** (`windows`): from the LeWM corpus's training units (explore, mission, free; `lewm_train`), `units` units drawn
with the protocol seed; in each, the 12-frame windows (frames s..s+11, decision frame t = s+3) whose start is a
multiple of `stride` and at least 7 (t >= 10, so C3's frame 1 s back exists); per window two distinct horizons
(1-8, seeded). Labels: the body-frame motion from the simulator poses at t to t+h; the applied-command history at t
(15 x 3, the policy packet's) and the executed tape (the applied commands of the eight 100-ms ticks after t, as
recorded at t+8), as `go2nav.data.decoder_contexts` defines them for T7.

**Heads**, each arm's deployed architecture, trained on exactly those (window, horizon) pairs with one recipe
(`RECIPE`: AdamW, 20,000 updates of 512 pairs, lr 3e-4 with cosine decay, weight decay 1e-4, gradient clip 1, MSE in
normalised units) and read alone (no T7 fit):
- the decoder (reads the predicted future): C5 `LatentMotionDecoder` (fresh; normalisation fitted on the protocol
  labels) on [z_t, rollout(t+h) - z_t, z_t - z_(t-1), z_t - z_(t-3)]; C3 `MotionDecoder` as T7 builds it (grown from
  T5b, past frames, proj 64, hidden 896, depth 1) on the pooled tokens of t, t-0.5 s, t-1 s and T4's prediction;
- the direct head (no predicted future): C6 `LatentDirectPredictor`, C4 `DirectMotionPredictor` (both fresh, as T7).
C3's grown decoder starts from T5b's weights (trained on T5b's own recordings) and T5b's normalisation, which the other
heads do not get; `--c3-init fresh` trains the same architecture from a default initialisation with normalisation
fitted on the protocol labels, as C5's, for the like-for-like comparison (`--c4` reuses a C4 already trained).
The world model's own rollout uses the executed action blocks (C5) or the executed tape (C3's T4), as each predictor
receives them at run time.

Outputs: the arm's fit file in the format its forecaster loads (`models.lewm.load_heads`, `models.motion.
load_decoder_and_c4`), `evaluation.json` (T7's evaluation sets when a cache is given) and lineage. Stages are
resumable: features per unit, heads written atomically.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from go2nav.data.lewm_dataset import raise_open_file_limit
from go2nav.training.common import sha256_file, write_lineage

PROTOCOL = dict(name='readout_v1', seed=2026102001, role='lewm_train', kinds=('explore', 'mission', 'free'),
                units=None, stride=None, horizons_per_window=2, min_start=7)
RECIPE = dict(updates=20000, batch=512, lr=3e-4, weight_decay=1e-4, clip=1.)
HORIZONS = 8


def _yaw(q):
    x, y, z, w = np.asarray(q).T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


# -- protocol windows -------------------------------------------------------------------------------------------------

def make_windows(corpus, units, stride, seed=PROTOCOL['seed']):
    """The protocol: chosen units and per window (unit, start, h1, h2)."""
    root = Path(corpus)
    meta = json.loads((root/'index_12.json').read_text())['units']
    index = np.load(root/'index_12.npz')
    rng = np.random.default_rng(seed)
    pool = [i for i, u in enumerate(meta) if u['role'] == PROTOCOL['role'] and u['kind'] in PROTOCOL['kinds']]
    chosen = sorted(int(u) for u in rng.choice(pool, min(units, len(pool)), replace=False))
    rows = []
    for u in chosen:
        starts = index['start'][index['unit'] == u]
        starts = starts[(starts % stride == 0) & (starts >= PROTOCOL['min_start'])]
        for s in starts:
            h = rng.choice(np.arange(1, HORIZONS+1), PROTOCOL['horizons_per_window'], replace=False)
            rows.append((u, int(s), *sorted(int(v) for v in h)))
    return dict(PROTOCOL, units=units, stride=stride, corpus_units=[dict(index=u, path=meta[u]['path']) for u in chosen],
                windows=rows)


def labels(root, path, starts, horizons):
    """Per window: motion (n, k, 3) at its horizons, applied history at t (n, 15, 3), executed tape (n, 8, 3)."""
    with np.load(Path(root)/path/'steps.npz') as steps:
        pose, history = steps['pose'], steps['applied_history']
    t = np.asarray(starts)+3
    yaw = _yaw(pose[:, 3:7])
    c, s = np.cos(yaw[t]), np.sin(yaw[t])
    y = []
    for k in range(horizons.shape[1]):
        e = t+horizons[:, k]
        d = pose[e, :2]-pose[t, :2]
        y.append(np.stack([c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1], np.angle(np.exp(1j*(yaw[e]-yaw[t])))], 1))
    return np.stack(y, 1).astype(np.float32), history[t].astype(np.float32), history[t+8][:, -8:].astype(np.float32)


def by_unit(protocol):
    rows = np.asarray(protocol['windows'], np.int64)
    paths = {u['index']: u['path'] for u in protocol['corpus_units']}
    for u in sorted(paths):
        sel = rows[rows[:, 0] == u]
        yield u, paths[u], sel[:, 1], sel[:, 2:]


# -- training ---------------------------------------------------------------------------------------------------------

def train(model, batch_fn, n, seed, device, recipe=RECIPE):
    """Train `model` (normalised output) on pairs 0..n-1: batch_fn(indices) -> (prediction inputs..., y)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, recipe['updates'])
    model.train()
    history = []
    for step in range(recipe['updates']):
        idx = torch.randint(n, (recipe['batch'],), generator=g).numpy()
        prediction, y = batch_fn(model, idx)
        loss = ((prediction-(y-model.target_mean)/model.target_scale)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['clip'])
        opt.step()
        sched.step()
        if (step+1) % 1000 == 0:
            history.append(dict(step=step+1, loss=float(loss.detach())))
    return model.eval(), history


def atomic_save(value, path):
    tmp = Path(str(path)+'.tmp')
    torch.save(value, tmp)
    tmp.rename(path)


# -- LeWM (C5, C6) ----------------------------------------------------------------------------------------------------

@torch.inference_mode()
def lewm_features(model, root, protocol, device):
    """zh (n, 4, D), rollout at each window's horizons (n, k, D), y (n, k, 3), past (n, 15, 3), tape (n, 8, 3), h (n, k)."""
    from go2nav.models.lewm import action_blocks, control_stats, normalise_frames
    cmean, cstd = control_stats()
    out = {k: [] for k in ('zh', 'fut', 'y', 'past', 'tape', 'h')}
    for _, path, starts, horizons in by_unit(protocol):
        frames = np.load(Path(root)/path/'frames.npy', mmap_mode='r')
        with np.load(Path(root)/path/'steps.npz') as steps:
            blocks = action_blocks(steps['executed'], cmean, cstd).to(device)
        t = torch.from_numpy(starts+3).to(device)
        need = np.unique(np.concatenate([starts+3+k for k in range(-3, 1)]))
        x = torch.from_numpy(np.ascontiguousarray(frames[need])).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            zn = torch.cat([model.encode(normalise_frames(x[i:i+1024])[:, None])[:, 0].float() for i in range(0, len(x), 1024)])
        row = torch.full((int(need.max())+1,), -1, dtype=torch.long, device=device)
        row[torch.from_numpy(need).to(device)] = torch.arange(len(need), device=device)
        zh = torch.stack([zn[row[t+k]] for k in range(-3, 1)], 1)
        acts = torch.stack([blocks[t-3+k] for k in range(3+HORIZONS)], 1)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            rolled = model.rollout(zh, acts, HORIZONS).float()
        hz = torch.from_numpy(horizons).to(device)
        out['fut'].append(torch.gather(rolled, 1, (hz-1)[..., None].expand(-1, -1, rolled.shape[-1])).half().cpu())
        out['zh'].append(zh.half().cpu())
        y, past, tape = labels(root, path, starts, horizons)
        for k, v in (('y', y), ('past', past), ('tape', tape), ('h', horizons)):
            out[k].append(torch.from_numpy(np.asarray(v)))
    return {k: torch.cat(v) for k, v in out.items()}


def run_lewm(protocol_path, corpus, world_model, out, *, device='cuda:0', cache=None):
    from go2nav.models.lewm import LatentDirectPredictor, LatentMotionDecoder, control_stats, load_world_model
    raise_open_file_limit()
    out = Path(out)
    if (out/'.done').exists():
        return out/'fit.pt'
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    started = time.time()
    data = lewm_features(load_world_model(world_model, device), corpus, protocol, device)
    n, k = data['h'].shape
    print(json.dumps(dict(pairs=n*k, wall_s=round(time.time()-started))), flush=True)
    flat = data['y'].reshape(-1, 3).double()
    mean, scale = flat.mean(0).float(), flat.std(0).float()
    D = data['zh'].shape[-1]
    zh, fut, y = data['zh'].to(device), data['fut'].to(device), data['y'].to(device)
    cmean, cstd = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in control_stats())
    past = (data['past'][..., [0, 2]].to(device)-cmean)/cstd
    tape = (data['tape'][..., [0, 2]].to(device)-cmean)/cstd
    hz = data['h'].to(device)

    def pick(idx):  # pair index -> (window, slot)
        idx = torch.from_numpy(idx).to(device)
        return idx//k, idx % k
    heads, histories = {}, {}
    if (out/'c5.pt').exists():
        heads['c5'] = torch.load(out/'c5.pt', weights_only=False)
    else:
        torch.manual_seed(protocol['seed'])
        c5 = LatentMotionDecoder(D, mean, scale).to(device)
        def c5_batch(m, idx):
            w, j = pick(idx)
            return m.normalized(zh[w].float(), fut[w, j].float()), y[w, j]
        c5, histories['c5'] = train(c5, c5_batch, n*k, protocol['seed'], device)
        heads['c5'] = dict(state=c5.cpu().state_dict(), history=histories['c5'])
        atomic_save(heads['c5'], out/'c5.pt')
    if (out/'c6.pt').exists():
        heads['c6'] = torch.load(out/'c6.pt', weights_only=False)
    else:
        torch.manual_seed(protocol['seed']+1)
        c6 = LatentDirectPredictor(D, mean, scale).to(device)
        def c6_batch(m, idx):
            w, j = pick(idx)
            return m.normalized(zh[w].float(), past[w], tape[w], hz[w, j]), y[w, j]
        c6, histories['c6'] = train(c6, c6_batch, n*k, protocol['seed']+1, device)
        heads['c6'] = dict(state=c6.cpu().state_dict(), history=histories['c6'])
        atomic_save(heads['c6'], out/'c6.pt')
    fit = dict(c5=heads['c5']['state'], c6=heads['c6']['state'], latent=D, target_mean=mean, target_scale=scale,
               protocol=protocol['name'], world_model_sha256=sha256_file(world_model))
    atomic_save(fit, out/'fit.pt')
    if cache is not None:
        from go2nav.training.lewm_decoder import LatentCache, evaluate
        from go2nav.training.decoder import load_contexts, score
        from go2nav.models.lewm import load_heads
        c5m, c6m = load_heads(out/'fit.pt', device)
        report = evaluate(LatentCache(cache, load_contexts()), c5m, c6m, device)
        (out/'evaluation.json').write_text(json.dumps(dict(score_mm=score(report, 'C5'), score_c6_mm=score(report, 'C6'),
                                                           eval=report), indent=1)+'\n')
    write_lineage(out, component=f"readout_protocol_lewm_{protocol['name']}", seed=protocol['seed'],
                  parents=[str(world_model), str(protocol_path)], config=dict(protocol={k: v for k, v in protocol.items()
                  if k not in ('windows', 'corpus_units')}, recipe=RECIPE), data=dict(corpus=str(corpus), pairs=n*k),
                  checkpoint=out/'fit.pt', metrics=dict(history={h: heads[h]['history'] for h in heads},
                                                       wall_s=round(time.time()-started, 1)))
    (out/'.done').touch()
    return out/'fit.pt'


# -- V-JEPA (C3, C4) --------------------------------------------------------------------------------------------------

def vjepa_cache(protocol_path, corpus, fullres, out, *, predictor_checkpoint, batch=16, device='cuda:0'):
    """Per unit (resumable): pooled tokens (192 x 1024, layer-normalised) of every frame t-10, t-5, t of its windows,
    and T4's pooled prediction at each window's horizons under the executed tape. Frames from FULL_ROOT's re-collected
    units (`tools/readout_protocol/fullres_unit.py`, checked equal to the corpus)."""
    from PIL import Image
    from go2nav.models.predictor import load_horizon_predictor
    from go2nav.models.vjepa import VJepaEncoder, normalise, pool_tokens, preprocess
    from go2nav.training.decoder import control_stats
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    encoder, predictor = VJepaEncoder(device), load_horizon_predictor(predictor_checkpoint).to(device).eval()
    mean, std = control_stats(device)
    started = time.time()
    for u, path, starts, horizons in by_unit(protocol):
        folder = out/f'u{u:05d}'
        if (folder/'.done').exists():
            continue
        if not (Path(fullres)/path/'.verified').exists():
            raise FileNotFoundError(f'{path} not re-collected and verified in {fullres}')
        y, past, tape = labels(corpus, path, starts, horizons)
        t = starts+3
        need = np.unique(np.concatenate([t-10, t-5, t]))
        row = {int(f): i for i, f in enumerate(need)}
        shutil.rmtree(folder.with_suffix('.tmp'), ignore_errors=True)
        tmp = folder.with_suffix('.tmp')
        tmp.mkdir()
        pooled = np.lib.format.open_memmap(tmp/'frames.npy', 'w+', np.float16, (len(need), 192, 1024))
        pred = np.lib.format.open_memmap(tmp/'pred.npy', 'w+', np.float16, (len(t), horizons.shape[1], 192, 1024))
        tokens = {}
        with torch.inference_mode():
            for m in range(0, len(need), batch):
                names = need[m:m+batch]
                full = normalise(encoder.tokens(torch.stack([preprocess(Image.open(Path(fullres)/path/'full'/f'{f:04d}.png'))
                                                              for f in names]).to(device)))
                pooled[m:m+len(names)] = pool_tokens(full).half().cpu().numpy()
                for f, tk in zip(names, full):
                    tokens[int(f)] = tk
            for w0 in range(0, len(t), batch):
                ws = range(w0, min(w0+batch, len(t)))
                ctx, act, ctl, hor, slots = [], [], [], [], []
                for w in ws:
                    context = torch.stack([tokens[int(t[w]-10)], tokens[int(t[w]-5)], tokens[int(t[w])]])
                    control = (torch.tensor(past[w][:, [0, 2]].reshape(3, 5, 2), device=device)-mean)/std
                    action = torch.tensor(tape[w][:, [0, 2]], device=device)
                    for j, h in enumerate(horizons[w]):
                        ctx.append(context)
                        act.append(action)
                        ctl.append(control)
                        hor.append(int(h))
                        slots.append((w, j))
                predicted = predictor(torch.stack(ctx), torch.stack(act), torch.tensor(hor, device=device),
                                      torch.ones(len(ctx), 768, dtype=torch.bool, device=device), control=torch.stack(ctl))
                for (w, j), p in zip(slots, pool_tokens(normalise(predicted)).half().cpu().numpy()):
                    pred[w, j] = p
        pooled.flush()
        pred.flush()
        np.savez(tmp/'meta.npz', t=t, horizons=horizons, frames=need, y=y, past=past, tape=tape)
        tmp.rename(folder)
        (folder/'.done').touch()
        print(json.dumps(dict(unit=u, windows=len(t), wall_s=round(time.time()-started))), flush=True)
    (out/'meta.json').write_text(json.dumps(dict(protocol=str(protocol_path), predictor=str(predictor_checkpoint),
                                                 predictor_sha256=sha256_file(predictor_checkpoint),
                                                 wall_s=round(time.time()-started)))+'\n')


class VJepaPairs:
    """The cache's (window, horizon) pairs, gathered per batch from the per-unit memmaps."""

    def __init__(self, cache):
        self.units = []
        for folder in sorted(Path(cache).glob('u*')):
            if not (folder/'.done').exists():
                continue
            meta = np.load(folder/'meta.npz')
            row = {int(f): i for i, f in enumerate(meta['frames'])}
            t = meta['t']
            self.units.append(dict(frames=np.load(folder/'frames.npy', mmap_mode='r'), pred=np.load(folder/'pred.npy', mmap_mode='r'),
                                   ctx=np.stack([[row[int(x-10)], row[int(x-5)], row[int(x)]] for x in t]),
                                   h=meta['horizons'], y=meta['y'], past=meta['past'], tape=meta['tape']))
        k = self.units[0]['h'].shape[1]
        self.index = np.concatenate([np.stack([np.full(len(u['h'])*k, i), np.repeat(np.arange(len(u['h'])), k),
                                               np.tile(np.arange(k), len(u['h']))], 1) for i, u in enumerate(self.units)])

    def __len__(self):
        return len(self.index)

    def batch(self, idx, device):
        rows = self.index[np.sort(idx)]
        feats, pred, past, tape, h, y = [], [], [], [], [], []
        for u, w, j in rows:
            unit = self.units[u]
            feats.append(unit['frames'][unit['ctx'][w]])
            pred.append(unit['pred'][w, j])
            past.append(unit['past'][w])
            tape.append(unit['tape'][w])
            h.append(unit['h'][w, j])
            y.append(unit['y'][w, j])
        f = lambda v, dt=torch.float32: torch.from_numpy(np.stack(v)).to(device, non_blocking=True).to(dt)
        return f(feats), f(pred), np.stack(past), np.stack(tape), torch.tensor(np.asarray(h), device=device), f(y)


def run_vjepa(protocol_path, cache, out, *, base_readout, t7_cache=None, c3_init='grown', c4_from=None, device='cuda:0'):
    from go2nav.training.decoder import build_models, control_stats
    out = Path(out)
    if (out/'.done').exists():
        return out/'fit.pt'
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    pairs = VJepaPairs(cache)
    mean, std = control_stats(device)
    started = time.time()
    decoder, c4 = build_models(base_readout, protocol['seed'], device)
    if c3_init == 'fresh':
        torch.manual_seed(protocol['seed'])
        for m in decoder.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        flat = torch.from_numpy(np.concatenate([u['y'].reshape(-1, 3) for u in pairs.units]))
        decoder.target_mean.copy_(flat.mean(0).float())
        decoder.target_scale.copy_(flat.std(0).float())
    if c4_from is not None:
        shutil.copyfile(c4_from, out/'c4.pt')

    def control(past):
        return (torch.from_numpy(past[..., [0, 2]].reshape(len(past), 3, 5, 2)).to(device)-mean)/std

    def tape_n(tape):
        return torch.from_numpy(tape[..., [0, 2]]).to(device)
    histories = {}
    if (out/'c3.pt').exists():
        decoder.load_state_dict(torch.load(out/'c3.pt', weights_only=False)['state'])
    else:
        def c3_batch(m, idx):
            feats, pred, past, tape, h, y = pairs.batch(idx, device)
            return m.normalized(feats[:, 2], pred, feats[:, 1], feats[:, 0], control(past)), y
        decoder, histories['c3'] = train(decoder, c3_batch, len(pairs), protocol['seed'], device)
        atomic_save(dict(state=decoder.state_dict(), history=histories['c3']), out/'c3.pt')
    if (out/'c4.pt').exists():
        c4.load_state_dict(torch.load(out/'c4.pt', weights_only=False)['state'])
    else:
        def c4_batch(m, idx):
            feats, pred, past, tape, h, y = pairs.batch(idx, device)
            return m.normalized(feats, control(past), tape_n(tape), h), y
        c4, histories['c4'] = train(c4, c4_batch, len(pairs), protocol['seed']+1, device)
        atomic_save(dict(state=c4.state_dict(), history=histories['c4']), out/'c4.pt')
    from go2nav.training.decoder import DECODER
    fit = dict(readout=decoder.state_dict(), readout_config={k: v for k, v in DECODER.items() if k in ('proj', 'hidden', 'depth')},
               variant='past_frames', c4=c4.state_dict(), protocol=protocol['name'], c3_init=c3_init)
    atomic_save(fit, out/'fit.pt')
    if t7_cache is not None:
        from go2nav.training.decoder import DecoderCache, evaluate, load_contexts, score
        report = evaluate(DecoderCache(t7_cache, load_contexts()), decoder, c4, device)
        (out/'evaluation.json').write_text(json.dumps(dict(score_mm=score(report, 'C3'), score_c4_mm=score(report, 'C4'),
                                                           eval=report), indent=1)+'\n')
    write_lineage(out, component=f"readout_protocol_vjepa_{protocol['name']}", seed=protocol['seed'],
                  parents=[str(cache), str(base_readout), str(protocol_path)] + ([str(c4_from)] if c4_from else []),
                  config=dict(protocol={k: v for k, v in protocol.items() if k not in ('windows', 'corpus_units')}, recipe=RECIPE,
                              c3_init=c3_init),
                  data=dict(pairs=len(pairs)), checkpoint=out/'fit.pt',
                  metrics=dict(history=histories, wall_s=round(time.time()-started, 1)))
    (out/'.done').touch()
    return out/'fit.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('windows')
    w.add_argument('--corpus', required=True)
    w.add_argument('--units', type=int, required=True)
    w.add_argument('--stride', type=int, required=True)
    w.add_argument('--out', required=True, type=Path)
    lw = sub.add_parser('lewm')
    for a in ('--protocol', '--corpus', '--world-model', '--out'):
        lw.add_argument(a, required=True)
    lw.add_argument('--cache', help='the chain\'s lewm_decoder cache, for T7\'s evaluation')
    vc = sub.add_parser('vjepa-cache')
    for a in ('--protocol', '--corpus', '--fullres', '--out', '--predictor'):
        vc.add_argument(a, required=True)
    vc.add_argument('--batch', type=int, default=16)
    vj = sub.add_parser('vjepa')
    for a in ('--protocol', '--cache', '--out', '--base-readout'):
        vj.add_argument(a, required=True)
    vj.add_argument('--t7-cache', help='the T7 decoder cache, for T7\'s evaluation')
    vj.add_argument('--c3-init', choices=('grown', 'fresh'), default='grown')
    vj.add_argument('--c4', help='a trained c4.pt to reuse instead of training C4')
    a = p.parse_args()
    if a.command == 'windows':
        protocol = make_windows(a.corpus, a.units, a.stride)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(protocol)+'\n')
        print(json.dumps(dict(units=len(protocol['corpus_units']), windows=len(protocol['windows']))))
    elif a.command == 'lewm':
        print(run_lewm(a.protocol, a.corpus, a.world_model, a.out, cache=a.cache))
    elif a.command == 'vjepa-cache':
        vjepa_cache(a.protocol, a.corpus, a.fullres, a.out, predictor_checkpoint=a.predictor, batch=a.batch)
    else:
        print(run_vjepa(a.protocol, a.cache, a.out, base_readout=a.base_readout, t7_cache=a.t7_cache, c3_init=a.c3_init,
                        c4_from=a.c4))


if __name__ == '__main__':
    main()
