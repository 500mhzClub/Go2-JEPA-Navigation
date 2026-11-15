"""C001 label-free RGB slots, sequential embedding cache and paired O0/O1 training.

Each stage emits result.json awaiting a literature review. Dependent stages require
--parent-review bound to their input artifact; they never automatically interpret it.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from go2nav.models.balls_slots import RGBSlotConfig, RGBSlots, ObjectMaskedPredictor, SlotWorld, object_mask, predictor_loss
from go2nav.research import (atomic_json, budget_remaining, digest, gpu_run, require_record, require_review,
                            reviewed_artifact, require_training_slot, verify_files)
from go2nav.training.balls import to_pixels


def save(path, value):
    path = Path(path)
    torch.save(value, path.with_suffix('.partial'))
    os.replace(path.with_suffix('.partial'), path)


def generator(seed, purpose, epoch, device='cuda:0'):
    return torch.Generator(device=device).manual_seed(2026111507+seed*1000003+purpose*10007+epoch)


class Frames:
    """Only RGB and executed commands, never red/white positions or velocities."""
    def __init__(self, root, n, split, device='cuda:0'):
        folder = Path(root)/f'n{n}'/split
        self.frames = torch.from_numpy(np.load(folder/'frames.npy')).to(device)
        with np.load(folder/'labels.npz') as arrays:
            self.actions = torch.from_numpy(arrays['actions']).to(device)/3

    def batch(self, index, length):
        starts = self.frames.shape[1]-length+1
        episode, t = index//starts, index % starts
        offsets = t[:, None]+torch.arange(length, device=t.device)
        return to_pixels(self.frames[episode[:, None], offsets]), self.actions[episode[:, None], offsets[:, :-1]]

    def count(self, length):
        return len(self.frames)*(self.frames.shape[1]-length+1)


def resume(out, identity, model, optimizer, scheduler):
    path = out/'checkpoint.pt'
    if not path.exists():
        return 0, 0, math.inf, 0.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != identity:
        raise ValueError('Incompatible resume; previous config/logs preserved')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    return state['epoch'], state['batch_index'], state['best'], state.get('epoch_loss_sum', 0.)


def train_stage(root, n, out, seed, research_id, *, kind='slots', cache=None, parent_review=None, verification_review=None):
    record, research_hash = require_record(research_id)
    if not verification_review or require_review(verification_review)['status'] != 'passed':
        raise ValueError('Slot training requires a passing numerical-verification literature review')
    reviewed = require_review(verification_review)
    evidence = json.loads((Path(verification_review).parent/reviewed['result']).read_text())
    if not all(evidence.get(key, {}).get('passed') for key in ('rgb_slots', 'predictor')):
        raise ValueError('The review does not contain passing slot/predictor numerical checks')
    verified_sources = json.loads((Path(verification_review).parent/'manifest.json').read_text())['source_sha256']
    repo = Path(__file__).resolve().parents[2]
    for name in ('balls_slots', 'slots', 'lewm', 'norm'):
        source = f'go2nav/models/{name}.py'
        if digest(repo/source) != verified_sources[source]:
            raise ValueError(f'{source} changed since numerical verification')
    require_training_slot()
    protocol = record['protocol']['slot_training' if kind == 'slots' else 'predictor_training']
    out = Path(out)
    parents = [Path(root)/f'n{n}'/split/'meta.json' for split in ('train', 'val')]+[Path(verification_review)]
    if kind != 'slots':
        if cache is None or parent_review is None:
            raise ValueError('Predictors require a reviewed frozen embedding cache')
        reviewed_artifact(parent_review, Path(cache)/'index.json', 'cache_sha256')
        cache_index = json.loads((Path(cache)/'index.json').read_text())
        if cache_index['n'] != n or cache_index['seed'] != seed:
            raise ValueError('Frozen cache ball count/seed does not match predictor')
        verify_files(cache, {f'{split}.npy': value for split, value in cache_index['splits'].items()})
        verify_files(Path(root)/f'n{n}', cache_index['data_metadata'])
        parents += [Path(cache)/'index.json', Path(parent_review)]
    identity = dict(n=n, seed=seed, kind=kind, research_sha256=research_hash,
                    protocol=protocol, parents={str(p): digest(p) for p in parents})
    if (out/'checkpoint.pt').exists():
        existing = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False)
        if existing['identity'] != identity:
            raise ValueError('Incompatible resume; output left unchanged')
        del existing
    with gpu_run(out, research_id, identity, parents) as manifest:
        torch.cuda.set_per_process_memory_fraction(.80)
        torch.manual_seed(seed)
        model = RGBSlots(RGBSlotConfig(slots=n+2)).cuda() if kind == 'slots' else ObjectMaskedPredictor().cuda()
        if kind == 'slots':
            sets = {s: Frames(root, n, s) for s in ('train', 'val')}
            count, length = sets['train'].count(6), 6
        else:
            sets = {s: torch.from_numpy(np.load(Path(cache)/f'{s}.npy')).cuda() for s in ('train', 'val')}
            actions = {}
            for s in ('train', 'val'):
                with np.load(Path(root)/f'n{n}'/s/'labels.npz') as arrays:
                    actions[s] = torch.from_numpy(arrays['actions']).cuda()/3
            count, length = len(sets['train'])*(sets['train'].shape[1]-3), 4
        batch, epochs = protocol['batch'], protocol['epochs']
        batches = math.ceil(count/batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol['lr'])
        warm = max(1, int(epochs*batches*protocol.get('warmup_fraction', 0)))
        def scale(step):
            if kind != 'slots':
                return 1.
            return (step+1)/warm if step < warm else .5*(1+math.cos(math.pi*min(1., (step-warm)/(epochs*batches-warm))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        epoch0, offset, best, restored_loss = resume(out, identity, model, optimizer, scheduler)
        atomic_json(out/'config.json', identity)
        # Keep log entries only through the restored epoch/batch cursor.
        metrics = out/'metrics.jsonl'
        if metrics.exists():
            rows = [r for r in metrics.read_text().splitlines() if json.loads(r)['epoch'] < epoch0]
            metrics.write_text(''.join(r+'\n' for r in rows))

        def loss(split, index, epoch, validation=False):
            if kind == 'slots':
                pixels, _ = sets[split].batch(index, length)
                states = model(pixels, generator=generator(seed, 2, epoch))
                recon, _ = model.decode(states)
                value = (recon-pixels).square().mean()
                return value, value
            states = sets[split]
            starts = states.shape[1]-3
            e, t = index//starts, index % starts
            times = t[:, None]+torch.arange(4, device=t.device)
            z, a = states[e[:, None], times], actions[split][e[:, None], times[:, :3]]
            mask = object_mask(len(index), n+2, generator(seed, 3, epoch), z.device) if kind == 'o1' and not validation \
                else torch.zeros(len(index), n+2, dtype=torch.bool, device=z.device)
            value, terms = predictor_loss(model, z, a, mask)
            return value, terms['future']

        def checkpoint(epoch, next_batch):
            save(out/'checkpoint.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), identity=identity, epoch=epoch, batch_index=next_batch,
                 best=best, epoch_loss_sum=total if next_batch else 0.,
                 rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

        for epoch in range(epoch0, epochs):
            order = torch.randperm(count, device='cuda:0', generator=generator(seed, 1, epoch))
            model.train()
            total = restored_loss if epoch == epoch0 else 0.
            for b in range(offset if epoch == epoch0 else 0, batches):
                index = order[b*batch:(b+1)*batch]
                optimizer.zero_grad(set_to_none=True)
                value, _ = loss('train', index, epoch*batches+b)
                value.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.get('grad_clip', math.inf))
                if not torch.isfinite(value) or not torch.isfinite(grad):
                    raise FloatingPointError('Nonfinite training value/gradient; last checkpoint retained')
                optimizer.step()
                scheduler.step()
                total += float(value)*len(index)
                if (b+1) % 100 == 0:
                    checkpoint(epoch, b+1)
                    if budget_remaining(manifest) <= 0:
                        atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete', epoch=epoch))
                        return
            model.eval()
            vcount = sets['val'].count(length) if kind == 'slots' else len(sets['val'])*(sets['val'].shape[1]-3)
            validation = torch.randperm(vcount, device='cuda:0', generator=generator(seed, 4, 0))[:min(1024, vcount)]
            values = []
            with torch.no_grad():
                for idx in validation.split(32 if kind == 'slots' else batch):
                    _, future = loss('val', idx, 0, validation=True)
                    values.append((float(future), len(idx)))
            score = sum(v*c for v, c in values)/sum(c for _, c in values)
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite validation loss; cannot select checkpoint')
            if score < best:
                best = score
                save(out/'model.pt', dict(model=model.state_dict(), config=asdict(model.config) if kind == 'slots' else model.config,
                                         kind=kind, n=n, seed=seed, research_id=research_id, epoch=epoch,
                                         identity=identity,
                                         cache=None if cache is None else str(Path(cache).resolve()),
                                         cache_sha256=None if cache is None else digest(Path(cache)/'index.json')))
            row = dict(research_id=research_id, epoch=epoch, train_loss=total/count, validation=score, best=best)
            with metrics.open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            checkpoint(epoch+1, 0)
        atomic_json(out/'result.json', dict(research_id=research_id, kind=kind, n=n, seed=seed, best_validation=best,
                    checkpoint_sha256=digest(out/'model.pt'), status='awaiting_literature_review'))


def load_encoder(path, device='cuda:0'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = RGBSlots(RGBSlotConfig(**state['config']))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


def extract(root, n, checkpoint, out, research_id, parent_review):
    reviewed_artifact(parent_review, checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    require_training_slot()
    metadata = [Path(root)/f'n{n}'/s/'meta.json' for s in ('train', 'val')]
    with gpu_run(out, research_id, dict(stage='extract', n=n), [Path(checkpoint), Path(parent_review), *metadata]) as manifest:
        model, state = load_encoder(checkpoint)
        if state['n'] != n:
            raise ValueError('Encoder ball count does not match extraction data')
        for split in ('train', 'val'):
            frames = np.load(Path(root)/f'n{n}'/split/'frames.npy', mmap_mode='r')
            slots = np.lib.format.open_memmap(out/f'{split}.partial.npy', mode='w+', dtype=np.float32,
                                              shape=(*frames.shape[:2], n+2, 128))
            with torch.no_grad():
                for e in range(len(frames)):
                    if budget_remaining(manifest) <= 0:
                        slots.flush()
                        atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete'))
                        return
                    carry = None
                    for t in range(0, frames.shape[1], 10):
                        pixels = to_pixels(torch.from_numpy(np.array(frames[e:e+1, t:t+10])).cuda())
                        z = model(pixels, carry)
                        slots[e, t:t+len(z[0])] = z[0].cpu().numpy()
                        carry = z[:, -1]
            slots.flush()
            del slots
            (out/f'{split}.partial.npy').replace(out/f'{split}.npy')
        atomic_json(out/'index.json', dict(research_id=research_id, n=n, encoder=str(Path(checkpoint).resolve()),
                    encoder_sha256=digest(checkpoint), seed=state['seed'],
                    data_metadata={f'{s}/meta.json': digest(Path(root)/f'n{n}'/s/'meta.json') for s in ('train', 'val')},
                    splits={s: digest(out/f'{s}.npy') for s in ('train', 'val')}))
        atomic_json(out/'result.json', dict(research_id=research_id, cache_sha256=digest(out/'index.json'),
                    status='awaiting_literature_review'))


def load_world(checkpoint, device='cuda:0'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    index_path = Path(state['cache'])/'index.json'
    if digest(index_path) != state['cache_sha256']:
        raise ValueError('Frozen cache index no longer matches predictor parent')
    cache = json.loads(index_path.read_text())
    if digest(cache['encoder']) != cache['encoder_sha256']:
        raise ValueError('Frozen encoder no longer matches predictor parent')
    encoder, _ = load_encoder(cache['encoder'], device)
    predictor = ObjectMaskedPredictor(**state['config']).to(device)
    predictor.load_state_dict(state['model'])
    return SlotWorld(encoder, predictor).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['slots', 'extract', 'o0', 'o1'])
    for name in ('data', 'out', 'research'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--n', type=int, choices=[1, 3, 6], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint')
    p.add_argument('--cache')
    p.add_argument('--parent-review')
    p.add_argument('--verification-review')
    a = p.parse_args()
    if a.stage == 'extract':
        extract(a.data, a.n, a.checkpoint, a.out, a.research, a.parent_review)
    else:
        train_stage(a.data, a.n, a.out, a.seed, a.research, kind=a.stage, cache=a.cache, parent_review=a.parent_review,
                    verification_review=a.verification_review)


if __name__ == '__main__':
    main()
