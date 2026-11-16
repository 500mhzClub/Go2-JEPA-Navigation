"""Frozen F004 scientific trainer; admitted only by f004_run.py."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from f004_runtime import BASE, ROOT, gpu_device, need
from go2nav.data.paired_forecasts import Cache
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, require_record, verify_files
from go2nav.training.object_jepa import Rasters
from go2nav.training.paired_forecasts import Sampler, counts, features, forecast, objective, rate, save_checkpoint, targets


def update(model, optimizer, raw, order, rasters, arm, settings, step):
    """One complete update; CPU use is limited to the declared synthetic software fixture."""
    device = next(model.parameters()).device
    valid = torch.as_tensor(raw['valid'], device=device)
    denominator = counts(valid)
    need(int(denominator[0]) > 0, 'No valid trajectory targets in sampled batch')
    optimizer.zero_grad(set_to_none=True)
    lr = settings['lr']*rate(step, total=settings['steps'], warmup=settings['warmup_steps'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    totals = np.zeros(3)
    stride = 2*settings['microbatch_pairs']
    for start in range(0, len(order), stride):
        end = start+stride
        micro = {k:v[start:end] for k,v in raw.items()}
        inp = features(micro, rasters, device, order[start:end])
        target, mask = targets(micro, inp, order[start:end])
        prediction = forecast(model, inp, rasters, settings['rollout_steps'])
        loss, trajectory, response = objective(prediction, target, mask, arm, denominator)
        need(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        totals += [float(loss.detach()),float(trajectory),float(response)]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=totals[0],trajectory_m=totals[1],response_m=totals[2],grad_norm=float(norm),lr=lr,
                trajectory_targets=int(denominator[0]),response_targets=int(denominator[1]))


def validate(model, cache, rasters, settings):
    """Fixed first512 validation points, all7 futures; monitoring only, no checkpoint selection."""
    model.eval()
    n = min(settings['validation_points'], cache.n)
    errors = np.zeros(3)
    counted = np.zeros(3,dtype=np.int64)
    points, futures = np.repeat(np.arange(n),7), np.tile(np.arange(7),n)
    with torch.no_grad():
        for start in range(0,len(points),32):
            raw = cache.take(points[start:start+32], futures[start:start+32])
            inp = features(raw, rasters, next(model.parameters()).device)
            target, valid = targets(raw,inp)
            pred = forecast(model,inp,rasters)
            error = torch.linalg.vector_norm(pred-target,dim=-1)
            need(bool(torch.isfinite(error).all()), 'Nonfinite validation forecast')
            for i,h in enumerate((3,7,15)):
                errors[i] += float((error[:,h]*valid[:,h]).sum())
                counted[i] += int(valid[:,h].sum())
    model.train()
    return dict(points=n,endpoint_error_m=(errors/np.maximum(counted,1)).tolist(),counts=counted.tolist(),
                selection='monitoring_only_final_step_selected')


def child_admission(path):
    manifest = json.loads(Path(path).read_text())
    record,rh = require_record('F004')
    need(manifest['record_sha256'] == rh and manifest['settings'] == record['protocol']['training'], 'Wrong scientific protocol')
    need(manifest['model'] == record['protocol']['model'], 'Wrong model configuration')
    need(manifest['arm'] in record['protocol']['arms'] and manifest['seed'] == 0, 'Wrong training identity')
    parent = psutil.Process(os.getppid())
    need(parent.pid == manifest['supervisor_pid'] and parent.create_time() == manifest['supervisor_created'], 'Not the admitted supervisor child')
    need(Path(parent.cwd()).resolve() == ROOT.resolve() and manifest['cwd'] == str(ROOT), 'Wrong source checkout')
    rows = json.loads(Path(manifest['ledger']).read_text())
    matches = [r for r in rows if r['id'] == manifest['interval']['id']]
    need(matches == [manifest['interval']] and matches[0]['end'] is None, 'Unadmitted elapsed interval')
    verify_files(ROOT,manifest['inputs_sha256'])
    verify_files(ROOT,manifest['source_sha256'])
    return manifest, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--admission',required=True)
    args = ap.parse_args()
    admission,record = child_admission(args.admission)
    out = Path(admission['science_out'])
    need(not out.exists(), 'Refusing existing scientific output')
    device = gpu_device()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = record['protocol']['training']
    train, val = Cache(BASE/'data','train'), Cache(BASE/'data','val')
    mazes = sorted(set(train.arrays['maze'].tolist()+val.arrays['maze'].tolist()))
    rasters = Rasters(ROOT/'outputs/data/ring_corpus_v1/objects/rasters',mazes,device)
    model = ObjectJEPA(ObjectConfig(**record['protocol']['model'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    sampler = Sampler(train.n,0,admission['arm'])
    binding = dict(record_sha256=admission['record_sha256'],arm=admission['arm'],seed=0,
                   inputs_sha256=admission['inputs_sha256'],source_sha256=admission['source_sha256'],
                   model=record['protocol']['model'],settings=settings)
    out.mkdir()
    manifest = dict(admission, device=torch.cuda.get_device_name(device),torch_version=torch.__version__,
                    hip_version=torch.version.hip,status='running',started_at=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'manifest.json',manifest)
    began = time.monotonic()
    # Step0 fixes the actual initialization before any validation or updates.
    save_checkpoint(out/'initial.pt',model,optimizer,sampler,0,binding)
    with (out/'training.jsonl').open('x') as log, (out/'validation.jsonl').open('x') as val_log, sdpa_kernel(SDPBackend.MATH):
        for step in range(settings['steps']):
            points,futures,order = sampler.draw(settings['pairs_per_batch'])
            raw = train.take(points,futures)
            values = update(model,optimizer,raw,order,rasters,admission['arm'],settings,step)
            completed = step+1
            atomic_json(out/'progress.json',dict(step=completed,at=time.time(),elapsed_seconds=time.monotonic()-began))
            if completed % settings['log_every'] == 0 or completed == 1:
                row = dict(step=completed,elapsed_seconds=time.monotonic()-began,**values)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(row,allow_nan=False),flush=True)
            if completed % settings['checkpoint_every'] == 0:
                save_checkpoint(out/f'checkpoint_{completed:05d}.pt',model,optimizer,sampler,completed,binding)
            if completed % settings['validation_every'] == 0:
                row = dict(step=completed,**validate(model,val,rasters,settings))
                val_log.write(json.dumps(row,allow_nan=False)+'\n'); val_log.flush()
                print(json.dumps(dict(validation=row),allow_nan=False),flush=True)
    # Checkpoint20k includes the final validation BatchNorm evaluation-mode round trip.
    save_checkpoint(out/'model.pt',model,optimizer,sampler,settings['steps'],binding)
    atomic_json(out/'result.json',dict(research_id='F004',phase='training',status='awaiting_literature_review',
                run_identity=dict(arm=admission['arm'],seed=0),record_sha256=admission['record_sha256'],
                final_step=settings['steps'],checkpoint_sha256=digest(out/'model.pt'),
                initial_sha256=digest(out/'initial.pt'),training_log_sha256=digest(out/'training.jsonl'),
                validation_log_sha256=digest(out/'validation.jsonl'),binding=binding,
                elapsed_seconds=time.monotonic()-began,selection='final_step_only',
                next_step='Verify terminal checkpoint/logs and literature-review; qualified physical evaluation and trained numerical checks remain required.'))


if __name__ == '__main__':
    main()
