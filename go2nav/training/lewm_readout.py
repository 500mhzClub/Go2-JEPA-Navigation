"""The LeWM base readout: C5's decoder pretrained before T7's fit, as C3's decoder is grown from the T5b readout.

    python -m go2nav.training.lewm_readout corpus --world-model LEWM.pt --corpus ROOT --out DIR [--units 600] [--stride 1]
    python -m go2nav.training.lewm_readout contexts --world-model LEWM.pt --cache CACHE --out DIR
    python -m go2nav.training.lewm_readout package --readout DIR/readout.pt --c6-from FIT.pt --out DIR

T7 fits C5's decoder from scratch on 3,520 batches of 64 decisions; C3's decoder starts from T5b, a readout already
trained on true-future recording pairs (T5-0 -> T5a -> T5b). On the idm100 chain that difference, not the latents, was
the gap (docs/lewm/STATUS.md, cross-domain diagnostic): a head fitted on corpus windows read T7's on-policy decisions at
7.3 mm, T7's own fit at 20.8. Two sources, so the extra labelled data is measured rather than argued:
- `corpus`: windows of LeWM's corpus from the training mazes (`--units` units drawn with a fixed seed over the
  explore/mission/free kinds), labelled from the simulator poses: frames t-3..t, the body-frame motion to t+100..800 ms;
- `contexts`: only T7's own training decisions (the `decoder_fix` groups old/maze/onpolicy, all eight horizons), the
  labelled data C3 and C4 already see.
Two heads, each a `LatentMotionDecoder` with its own target normalisation: `rollout` reads the world model's rollout
under the executed actions (C5's input; grows C5's decoder) and `true` the true future frame's z (grows the L3 probe).
Each: 20,000 AdamW updates of batch 512 (lr 3e-4, cosine, weight decay 1e-4, clip 1), each example at a random horizon.
A head is written as soon as it is trained and a rerun skips it; `readout.pt` and `.done` close the folder.
`lewm_decoder fit --base-readout DIR/readout.pt` then runs T7's fit unchanged from these heads (C6 stays fresh, as C4).

**Package** (`package`): a fit file whose C5 decoder is a base readout's `rollout` head alone, read without T7's fit,
and whose C6 is another fit file's. This is the C5 baseline (docs/lewm/STATUS.md, span8w decoder fix): on the span8w
world model, T7's fit made C5 forecast in-place turns near walls 2-4 times too far and deadlock, and the corpus
readout alone did not (L4 5.0 mm, 71/100 round trips). Written as DIR/fit_base_only.pt with its lineage.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from go2nav.data.lewm_corpus import body_motion
from go2nav.models.lewm import LatentMotionDecoder, action_blocks, control_stats, load_world_model, normalise_frames
from go2nav.training.common import sha256_file, write_lineage

HORIZONS = 8
UPDATES, BATCH, LR, WEIGHT_DECAY, GRAD_CLIP = 20000, 512, 3e-4, 1e-4, 1.
SEED = 2026101801


@torch.inference_mode()
def corpus_windows(model, root, units, device, stride=1):
    """Per window: zh (n, 4, D) of frames t-3..t, true (n, 8, D) and rollout (n, 8, D) z at t+1..t+8, y (n, 8, 3)."""
    root = Path(root)
    meta = json.loads((root/'index_12.json').read_text())['units']
    index = np.load(root/'index_12.npz')
    cmean, cstd = control_stats(model.config.command_dim)
    out = {k: [] for k in ('zh', 'true', 'rollout', 'y')}
    for u in units:
        starts = index['start'][index['unit'] == u]
        starts = starts[starts % stride == 0]
        if not len(starts):
            continue
        folder = root/meta[u]['path']
        frames = np.load(folder/'frames.npy', mmap_mode='r')
        steps = np.load(folder/'steps.npz')
        zs = []
        for s in range(0, len(frames), 1024):
            x = torch.from_numpy(np.ascontiguousarray(frames[s:s+1024])).to(device, non_blocking=True)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                zs.append(model.encode(normalise_frames(x)[:, None])[:, 0].float())
        z = torch.cat(zs)
        t = starts+3
        y = body_motion(steps['pose'], t[:, None], t[:, None]+np.arange(1, HORIZONS+1))
        blocks = action_blocks(steps['executed'], cmean, cstd).to(device)
        tt = torch.from_numpy(t).to(device)
        zh = torch.stack([z[tt+k] for k in range(-3, 1)], 1)
        acts = torch.stack([blocks[tt-3+k] for k in range(3+HORIZONS)], 1)
        rolled = []
        for s in range(0, len(tt), 2048):
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                rolled.append(model.rollout(zh[s:s+2048], acts[s:s+2048], HORIZONS).float())
        out['zh'].append(zh.half().cpu())
        out['true'].append(torch.stack([z[tt+h] for h in range(1, HORIZONS+1)], 1).half().cpu())
        out['rollout'].append(torch.cat(rolled).half().cpu())
        out['y'].append(torch.from_numpy(y.astype(np.float32)))
    return {k: torch.cat(v) for k, v in out.items()}


def context_windows(cache_folder, device):
    """T7's training decisions (the decoder_fix groups) from a `lewm_decoder` cache, in the same layout."""
    from go2nav.training.decoder import load_contexts
    from go2nav.training.lewm_decoder import HISTORY, LatentCache, neighbours
    contexts = load_contexts()
    cache = LatentCache(cache_folder, contexts)
    idx = [i for i, c in enumerate(contexts) if c['set'] == 'train' and c['group'] in ('old', 'maze', 'onpolicy')]
    frame = [contexts[i]['frames'][2] for i in idx]
    z = lambda rows: torch.from_numpy(np.asarray(cache.z[np.asarray(rows).ravel()]))
    return dict(zh=z([[cache.row[f] for f in neighbours(n)] for n in frame]).reshape(len(idx), HISTORY, cache.latent),
                true=z([[cache.row[f] for f in neighbours(n, future=HORIZONS)] for n in frame]).reshape(len(idx), HORIZONS, cache.latent),
                rollout=torch.from_numpy(np.asarray(cache.rollout[idx])),
                y=torch.from_numpy(np.stack([np.asarray(contexts[i]['targets'], np.float32) for i in idx])))


def train_head(data, kind, seed, device):
    """A LatentMotionDecoder on [z_t, z(t+h) - z_t, ...] with z(t+h) the `kind` future, at random horizons."""
    zh, fut, y = (data[k].to(device) for k in ('zh', kind, 'y'))
    flat = y.reshape(-1, 3).double()
    head = LatentMotionDecoder(zh.shape[-1], flat.mean(0).float(), flat.std(0).float()).to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, UPDATES)
    g = torch.Generator(device=device).manual_seed(seed)
    history = []
    for n in range(UPDATES):
        i = torch.randint(len(zh), (BATCH,), device=device, generator=g)
        h = torch.randint(HORIZONS, (BATCH,), device=device, generator=g)
        target = (y[i, h]-head.target_mean)/head.target_scale
        loss = ((head.normalized(zh[i].float(), fut[i, h].float())-target)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if (n+1) % 1000 == 0:
            history.append(dict(step=n+1, loss=float(loss.detach())))
    return head.cpu(), history


def build(source, world_model, out, *, corpus=None, units=600, stride=1, cache=None, device='cuda:0'):
    out = Path(out)
    if (out/'.done').exists():
        return out/'readout.pt'
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(SEED)
    if source == 'corpus':
        rng = np.random.default_rng(SEED)
        meta = json.loads((Path(corpus)/'index_12.json').read_text())['units']
        pool = [i for i, u in enumerate(meta) if u['role'] == 'lewm_train' and u['kind'] in ('explore', 'mission', 'free')]
        chosen = sorted(rng.choice(pool, min(units, len(pool)), replace=False).tolist())
        data = corpus_windows(load_world_model(world_model, device), corpus, chosen, device, stride)
        provenance = dict(corpus=str(corpus), units=len(chosen), stride=stride)
    else:
        data = context_windows(cache, device)
        provenance = dict(contexts='data/training/decoder/contexts.jsonl', groups=['old', 'maze', 'onpolicy'])
    print(json.dumps(dict(source=source, windows=len(data['y']), wall_s=round(time.time()-started))), flush=True)
    histories = {}
    for k, kind in enumerate(('rollout', 'true')):
        path = out/f'{kind}.pt'
        if not path.exists():
            head, histories[kind] = train_head(data, kind, SEED+k, device)
            torch.save(dict(state=head.state_dict(), history=histories[kind]), str(path)+'.tmp')
            Path(str(path)+'.tmp').rename(path)
            print(json.dumps(dict(head=kind, final_loss=histories[kind][-1]['loss'], wall_s=round(time.time()-started))), flush=True)
        histories[kind] = torch.load(path, weights_only=False)['history']
    heads = {kind: torch.load(out/f'{kind}.pt', weights_only=False)['state'] for kind in ('rollout', 'true')}
    torch.save(dict(heads, latent=data['zh'].shape[-1], source=source, world_model_sha256=sha256_file(world_model)), out/'readout.pt')
    write_lineage(out, component=f'lewm_base_readout_{source}', seed=SEED, parents=[str(world_model)],
                  config=dict(updates=UPDATES, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, horizons=HORIZONS),
                  data=dict(provenance, windows=len(data['y'])), checkpoint=out/'readout.pt',
                  metrics=dict(history=histories, wall_s=round(time.time()-started, 1)))
    (out/'.done').touch()
    return out/'readout.pt'


def load_base(path, kind, device='cpu'):
    """The `rollout` or `true` head of a base readout."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    head = LatentMotionDecoder(state['latent'], torch.zeros(3), torch.ones(3))
    head.load_state_dict(state[kind])
    return head.to(device)


def package(readout, c6_fit, out):
    """OUT/fit_base_only.pt: C5 = the base readout's `rollout` head, C6 (and the fit file's other fields) from
    `c6_fit` (see the module docstring). Loads with `models.lewm.load_heads`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fit = torch.load(c6_fit, map_location='cpu', weights_only=False)
    base = torch.load(readout, map_location='cpu', weights_only=False)
    if base['world_model_sha256'] != fit['world_model_sha256']:
        raise ValueError('the base readout and the C6 fit read different world models')
    fit['c5'] = base['rollout']
    fit['base_readout'] = str(readout)
    fit['note'] = f'c5: the base readout {readout} rollout head alone (no T7 fit); c6: {c6_fit}'
    checkpoint = out/'fit_base_only.pt'
    torch.save(fit, checkpoint)
    write_lineage(out, component='lewm_c5_decoder_and_c6_base_only', seed=None, parents=[str(readout), str(c6_fit)],
                  config=dict(c5='base readout rollout head, no T7 fit', c6='from the C6 fit'), data={},
                  checkpoint=checkpoint)
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='source', required=True)
    for source in ('corpus', 'contexts'):
        s = sub.add_parser(source)
        s.add_argument('--world-model', required=True, type=Path)
        s.add_argument('--out', required=True, type=Path)
        if source == 'corpus':
            s.add_argument('--corpus', required=True, type=Path)
            s.add_argument('--units', type=int, default=600)
            s.add_argument('--stride', type=int, default=1, help='keep windows whose start is a multiple of this')
        else:
            s.add_argument('--cache', required=True, type=Path)
    k = sub.add_parser('package')
    k.add_argument('--readout', required=True, type=Path, help='a base readout (DIR/readout.pt)')
    k.add_argument('--c6-from', required=True, type=Path, help='the fit file whose C6 to keep')
    k.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.source == 'package':
        print(package(a.readout, a.c6_from, a.out))
    elif a.source == 'corpus':
        print(build('corpus', a.world_model, a.out, corpus=a.corpus, units=a.units, stride=a.stride))
    else:
        print(build('contexts', a.world_model, a.out, cache=a.cache))


if __name__ == '__main__':
    main()
