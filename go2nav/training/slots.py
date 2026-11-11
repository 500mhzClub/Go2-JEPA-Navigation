"""Training the slot model (`models.slots`; docs/dynamic/PLAN.md P23) on cached DINOv2 features (`data.dino_features`).

    python -m go2nav.training.slots train --features FEATURES --out DIR [--config FILE] [--set key=value ...]

A batch is `batch` clips of `frames` consecutive frames, each from a unit drawn at random and a start drawn within it,
read from the float16 feature files (memory-mapped) in the training process itself (about 20 MB a step; no loader
processes forked after the GPU is opened). Adam with linear warm-up then
cosine decay, gradient clipping, bf16 autocast (SIGReg and the losses in float32). Resumable from DIR/checkpoint.pt.
Validation (`DIR/validation.jsonl`): the losses on fixed clips of the val units, and how much each slot covers (the
mean largest slot's share of the patches, and the slots' entropy over patches), the sign of slots collapsing onto
one or spreading over all. P23's measures are `eval.slot_measures`.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from go2nav.models.lewm import SIGReg
from go2nav.models.slots import SlotConfig, SlotModel, losses

MEMORY_FRACTION = .8
DEFAULTS = dict(seed=2026111106, model={}, data=dict(batch=32, frames=4),
                objective=dict(similarity_weight=.25, sigreg_weight=.05, temperature=.075, sigreg_projections=512),
                optimiser=dict(lr=1e-4, warmup=2000, steps=60000, grad_clip=1.),
                runtime=dict(log_every=100, validate_every=2500, checkpoint_every=1000, validation_clips=512))


def merged(base, extra):
    out = dict(base)
    for key, value in extra.items():
        out[key] = merged(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def load_config(path, overrides=()):
    config = merged(DEFAULTS, yaml.safe_load(Path(path).read_text()) if path else {})
    for item in overrides:
        key, _, value = item.partition('=')
        node = config
        *parents, last = key.split('.')
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return config


class Clips(torch.utils.data.IterableDataset):
    def __init__(self, root, role, frames, batch, seed):
        index = json.loads((Path(root)/'index.json').read_text())
        self.files = [Path(root)/role/f'{Path(p).name}.npy' for p in index['units'][role]]
        self.lengths = [index['frames'][p] for p in index['units'][role]]
        self.frames, self.batch, self.seed = frames, batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        weights = np.array(self.lengths, float)/sum(self.lengths)
        while True:
            out = []
            for u in rng.choice(len(self.files), self.batch, p=weights):
                start = int(rng.integers(self.lengths[u]-self.frames+1))
                out.append(np.load(self.files[u], mmap_mode='r')[start:start+self.frames])
            yield torch.from_numpy(np.stack(out))


def fixed_clips(root, role, frames, count, seed):
    index = json.loads((Path(root)/'index.json').read_text())
    units = index['units'][role]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        p = units[int(rng.integers(len(units)))]
        start = int(rng.integers(index['frames'][p]-frames+1))
        out.append(np.load(Path(root)/role/f'{Path(p).name}.npy', mmap_mode='r')[start:start+frames])
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def validate(model, clips, objective, sigreg, device, batch=64):
    model.eval()
    sums, n, largest, entropy = {}, 0, 0., 0.
    for i in range(0, len(clips), batch):
        x = clips[i:i+batch].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, terms = losses(model, x, sigreg=sigreg, similarity_weight=objective['similarity_weight'],
                              sigreg_weight=objective['sigreg_weight'], temperature=objective['temperature'])
            slots, _ = model(x)
            _, _, masks = model.decode(slots)
        masks = masks.float()
        share = masks.mean(-1)                                                    # (B, T, K) each slot's share
        largest += float(share.max(-1).values.mean())*len(x)
        p = masks/masks.sum(-1, keepdim=True)
        entropy += float(-(p*(p+1e-9).log()).sum(-1).mean()/math.log(masks.shape[-1]))*len(x)
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.)+float(v)*len(x)
        n += len(x)
    model.train()
    return dict(**{k: round(v/n, 5) for k, v in sums.items()}, largest_slot_share=round(largest/n, 4),
                slot_entropy=round(entropy/n, 4))


def schedule(warmup, total):
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    return factor


def train(features, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=1)+'\n')
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    torch.backends.cuda.enable_flash_sdp(False)              # see models.slots: the fused attention kernels fault here
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(config['seed'])
    model = SlotModel(SlotConfig(**config['model'])).to(device)
    data, objective, optim, run = config['data'], config['objective'], config['optimiser'], config['runtime']
    sigreg = SIGReg(num_proj=objective['sigreg_projections']).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=optim['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule(optim['warmup'], optim['steps']))
    step, checkpoint = 0, out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model']), optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
    loader = iter(Clips(features, 'lewm_train', data['frames'], data['batch'], config['seed']+step))   # no forked workers
    clips = fixed_clips(features, 'lewm_val', data['frames'], run['validation_clips'], config['seed'])
    log, vlog = (out/'train.jsonl').open('a'), (out/'validation.jsonl').open('a')
    generator = torch.Generator(device=device)
    sums, started, n = {}, time.time(), 0
    model.train()
    while step < optim['steps']:
        cpu = next(loader)
        x = cpu.to(device)
        generator.manual_seed(config['seed']*1_000_003+step)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total, terms = losses(model, x, sigreg=sigreg, similarity_weight=objective['similarity_weight'],
                                  sigreg_weight=objective['sigreg_weight'], temperature=objective['temperature'],
                                  generator=generator)
        if not torch.isfinite(total):
            raise FloatingPointError(f'loss {float(total)} at step {step}')
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim['grad_clip'])
        optimiser.step(), scheduler.step()
        step, n = step+1, n+1
        for key, value in terms.items():
            sums[key] = sums.get(key, 0.)+float(value)
        sums['grad_norm'] = sums.get('grad_norm', 0.)+float(norm)
        if step % run['log_every'] == 0:
            log.write(json.dumps(dict(step=step, lr=scheduler.get_last_lr()[0], s_per_step=round((time.time()-started)/n, 4),
                                      **{k: round(v/n, 5) for k, v in sums.items()}))+'\n')
            log.flush()
            sums, started, n = {}, time.time(), 0
        if step % run['validate_every'] == 0 or step == optim['steps']:
            vlog.write(json.dumps(dict(step=step, **validate(model, clips, objective, sigreg, device)))+'\n')
            vlog.flush()
        if step % run['checkpoint_every'] == 0 or step == optim['steps']:
            partial = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimiser=optimiser.state_dict(), scheduler=scheduler.state_dict(),
                            step=step, config=config), partial)
            partial.rename(checkpoint)
    torch.save(dict(model=model.state_dict(), config=model.config.to_dict(), step=step), out/'model.pt')
    return step


def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    model = SlotModel(SlotConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--features', required=True, type=Path)
    t.add_argument('--out', required=True, type=Path)
    t.add_argument('--config', type=Path)
    t.add_argument('--set', action='append', default=[])
    a = p.parse_args()
    print(train(a.features, a.out, load_config(a.config, a.set)))


if __name__ == '__main__':
    main()
