"""Stage B of the walker memory (docs/dynamic/PLAN.md, P9, step 3): the memory and the world model trained together.

    python -m go2nav.training.memory_joint --corpus ROOT --config configs/memory/joint.yaml --out outputs/memory/joint

The world model (`lewm_config`, warm-started from `init_model`) keeps its own objectives on its own batches, as
`training.lewm` trains it; each step also draws `memory.batch` sequences of `memory.length` memory steps (every second
frame, 5 Hz) from the training units with labels (`data.memory_labels`), encodes their frames with the world model in the
graph, runs the memory (warm-started from stage A, `memory.init`) and adds its binary cross-entropy, times
`memory.weight`, so the memory's gradient reaches the encoder. The memory's parameters train at `memory.lr`. Poses and
noise as in stage A (`training.memory`). Resumable as `training.lewm`: every `checkpoint_every` steps the model, the
memory, the optimiser, the scheduler and the step are written atomically, and the batches are functions of the seed and
step. Outputs: DIR/lewm.pt (the world model), DIR/memory.pt, metrics.jsonl and the world model's validation
(`training.lewm.validate`) as evaluation.json. The memory is scored after re-encoding the states with the new world
model (`training.memory cache`, then `evaluate`).
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from go2nav.data import memory_labels as ml
from go2nav.data.lewm_dataset import Corpus, WindowBatches, keyed_rng, raise_open_file_limit
from go2nav.models.lewm import LeWorldModel, SIGReg, losses, normalise_frames, save_world_model
from go2nav.models.memory import STEP_FRAMES, WalkerMemory, load_memory, save_memory
from go2nav.training import lewm as lt
from go2nav.training.memory import motions, run_memory, units

REPO = Path(__file__).resolve().parents[2]


class MemorySequences(torch.utils.data.Dataset):
    """Item k: the memory batch of step `first + k`: frames uint8 (B, L, 168, 224, 3) at every second frame, step motions
    with pose noise (B, L, 3) and labels (B, L, CELLS, CELLS)."""

    def __init__(self, root, *, seed, first, last, batch, length, pose_noise_m, yaw_noise_rad):
        self.seed, self.first, self.last, self.batch, self.length = seed, first, last, batch, length
        self.noise = np.array([pose_noise_m, pose_noise_m, yaw_noise_rad], np.float32)
        self.items = []
        for folder in units(root, 'lewm_train'):
            with np.load(folder/ml.FILE) as d:
                n = len(d['t'])
            if n >= length:
                self.items.append((folder, n))

    def __len__(self):
        return self.last-self.first

    def __getitem__(self, k):
        raise_open_file_limit()
        rng = keyed_rng(self.seed, 'memory', self.first+k)
        frames, motion, labels = [], [], []
        for _ in range(self.batch):
            folder, n = self.items[rng.integers(len(self.items))]
            start = int(rng.integers(0, n-self.length+1))
            steps = np.arange(start, start+self.length)
            frames.append(np.asarray(np.load(folder/'frames.npy', mmap_mode='r')[steps*STEP_FRAMES]))
            with np.load(folder/'steps.npz') as s:
                motion.append(motions(s['pose'])[steps])
            with np.load(folder/ml.FILE) as d:
                labels.append(d['occupied'][steps])
        motion = np.stack(motion)
        motion = motion+rng.normal(size=motion.shape).astype(np.float32)*self.noise
        return torch.from_numpy(np.stack(frames)), torch.from_numpy(motion.astype(np.float32)), torch.from_numpy(np.stack(labels))


def train(config, corpus_root, out, device='cuda:0'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = out/'config.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != config:
        raise ValueError(f'{out} was started with another config; use a new folder')
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    raise_open_file_limit()
    device = torch.device(device)
    lewm = lt.load_config(REPO/config['lewm_config'])
    for key, value in config.get('lewm_overrides', {}).items():
        node = lewm
        *parents, leaf = key.split('.')
        for p in parents:
            node = node[p]
        node[leaf] = value
    seed, data, objective, optim, runtime = (lewm[k] for k in ('seed', 'data', 'objective', 'optimiser', 'runtime'))
    seed, steps, mem = config['seed'], config['steps'], config['memory']
    torch.cuda.set_per_process_memory_fraction(runtime.get('memory_fraction', lt.MEMORY_FRACTION), device)
    mconfig = lt.model_config(lewm)
    length = mconfig.history+objective['rollout_steps']
    corpus = Corpus(corpus_root, length=length, roles=('lewm_train',), kinds=tuple(data.get('train_kinds', lt.TRAIN_KINDS)),
                    command_dim=mconfig.command_dim, movers=True, free=True)
    val_length = mconfig.history+lt.EVAL_STEPS
    validation = Corpus(corpus_root, length=val_length, roles=('lewm_val',), kinds=('explore', 'mission', 'free'),
                        command_dim=mconfig.command_dim)
    val_indices = validation.fixed(lt.VALIDATION_WINDOWS, lt.VALIDATION_SEED)
    model = LeWorldModel(mconfig).to(device).train()
    model.load_state_dict(torch.load(REPO/config['init_model'], map_location='cpu', weights_only=False)['model_state_dict'])
    memory = (load_memory(REPO/mem['init'], device) if mem.get('init') else WalkerMemory().to(device)).train()
    model.predictor.checkpoint = bool(runtime.get('checkpoint_predictor'))
    model.encoder.checkpoint_blocks = runtime.get('checkpoint_blocks', 0)
    sigreg = SIGReg(knots=objective.get('sigreg_knots', 17), num_proj=objective['sigreg_projections']).to(device)
    optimiser = torch.optim.AdamW([dict(params=model.parameters(), lr=config['lr']),
                                   dict(params=memory.parameters(), lr=mem['lr'])],
                                  weight_decay=optim['weight_decay'], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lt.schedule(config['warmup_steps'], steps))
    first = 0
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        memory.load_state_dict(state['memory'])
        optimiser.load_state_dict(state['optimiser'])
        scheduler.load_state_dict(state['scheduler'])
        first = state['step']+1
        rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line.strip()] \
            if (out/'metrics.jsonl').exists() else []
        (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step'] <= state['step']))
        print(json.dumps(dict(resumed_at=first)), flush=True)
    if runtime.get('compile_scope') == 'encoder':
        for block in model.encoder.blocks:
            block.compile(mode=runtime['compile'])
    workers = runtime['workers']
    loader = torch.utils.data.DataLoader(
        WindowBatches(corpus, seed=seed, first=first, last=steps, mazes_per_batch=data['mazes_per_batch'],
                      windows_per_maze=data['windows_per_maze'], batch_size=data['batch_size']),
        batch_size=None, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=runtime['prefetch'],
        multiprocessing_context='forkserver')
    sequences = torch.utils.data.DataLoader(
        MemorySequences(corpus_root, seed=seed, first=first, last=steps, batch=mem['batch'], length=mem['length'],
                        pose_noise_m=mem['pose_noise_m'], yaw_noise_rad=mem['yaw_noise_rad']),
        batch_size=None, shuffle=False, num_workers=max(2, workers//2), pin_memory=True, prefetch_factor=2,
        multiprocessing_context='forkserver')
    generator = torch.Generator(device=device)
    log, started, window = open(out/'metrics.jsonl', 'a'), time.time(), defaultdict(float)
    step = first-1
    for step, ((frames, actions, _, movers, free), (seq, motion, labels)) in enumerate(zip(loader, sequences), start=first):
        frames, actions = frames.to(device, non_blocking=True), actions.to(device, non_blocking=True)
        movers, free = movers.to(device, non_blocking=True), free.to(device, non_blocking=True)
        seq, motion, labels = seq.to(device, non_blocking=True), motion.to(device, non_blocking=True), labels.to(device).float()
        generator.manual_seed(lt.keyed_seed(seed, 'sigreg', step))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            terms = losses(model, normalise_frames(frames), actions, rollout_steps=objective['rollout_steps'], sigreg=sigreg,
                           lam=objective['sigreg_weight'], inverse_weight=objective.get('inverse_weight', 0.),
                           generator=generator, grid_weight=objective.get('grid_weight', 1.), movers=movers,
                           mover_weight=objective.get('mover_weight', 0.),
                           rollout_mover_weight=objective.get('rollout_mover_weight', 0.), free=free,
                           free_weight=objective.get('free_weight', 0.))
            states = model.encode(normalise_frames(seq))                          # (B, L, 13, D), in the graph
            logits = run_memory(memory, states, motion, device)
        b = mem['burn_in']
        memory_loss = F.binary_cross_entropy_with_logits(logits[:, b:].float(), labels[:, b:])
        loss = terms['loss']+mem['weight']*memory_loss
        optimiser.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            scheduler.step()
            continue
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([*model.parameters(), *memory.parameters()], optim['grad_clip'])
        optimiser.step()
        scheduler.step()
        for key, value in terms.items():
            window[key] += float(value.detach())
        window['memory'] += float(memory_loss.detach())
        window['grad_norm'] += float(grad)
        window['n'] += 1
        last = step == steps-1
        if (step+1) % runtime['log_every'] == 0 or last:
            n = window.pop('n')
            row = dict(step=step, lr=scheduler.get_last_lr()[0], s_per_step=round((time.time()-started)/n, 3),
                       **{k: v/n for k, v in window.items()})
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(row), flush=True)
            window, started = defaultdict(float), time.time()
        if (step+1) % config['validate_every'] == 0 or last:
            row = dict(step=step, validation=lt.validate(model, validation, val_indices, rollout_steps=lt.EVAL_STEPS, device=device))
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps(dict(step=step, rollout=np.round(row['validation']['rollout_by_step'], 4).tolist())), flush=True)
        if (step+1) % config['checkpoint_every'] == 0 or last:
            lt.atomic_save(dict(model=model.state_dict(), memory=memory.state_dict(), optimiser=optimiser.state_dict(),
                                scheduler=scheduler.state_dict(), step=step), checkpoint)
    log.close()
    if step != steps-1:
        raise RuntimeError(f'training stopped at step {step} of {steps}')
    save_world_model(out/'lewm.pt', model, extra=dict(step=steps, training_config=lewm, joint=config))
    save_memory(out/'memory.pt', memory, extra=dict(training=config))
    evaluation = dict(validation=lt.validate(model, validation, val_indices, rollout_steps=lt.EVAL_STEPS, device=device))
    (out/'evaluation.json').write_text(json.dumps(evaluation, indent=1)+'\n')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(train(yaml.safe_load(a.config.read_text()), a.corpus, a.out))


if __name__ == '__main__':
    main()
