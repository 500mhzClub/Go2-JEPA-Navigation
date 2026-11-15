"""A001: non-destructive checkpoint diagnostics and train-only BN calibration.

python -m go2nav.eval.balls_inference --checkpoint RUN/checkpoint.pt --data DATA --out NEW --research A001
All model execution uses the dedicated GPU except the explicit CPU parity reference.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from go2nav.data.balls import load as load_data
from go2nav.research import atomic_json, budget_remaining, digest, gpu_run, require_record, require_training_slot
from go2nav.training.balls import load, to_pixels


def windows(root, n, split, count, seed, batch=64, device='cuda:0'):
    frames, labels = load_data(root, n, split)
    rng = np.random.default_rng(seed)
    episodes = rng.integers(len(frames), size=count)
    starts = rng.integers(frames.shape[1]-3, size=count)
    for i in range(0, count, batch):
        e, t = episodes[i:i+batch, None], starts[i:i+batch, None]+np.arange(4)
        pixels = to_pixels(torch.from_numpy(np.array(frames[e, t])).to(device))
        actions = torch.from_numpy(np.array(labels['actions'][e, t[:, :3]])).to(device)/3
        yield pixels, actions


def outputs(model, pixels, actions):
    dtype = next(model.parameters()).dtype
    pixels, actions = pixels.to(dtype), actions.to(dtype)
    z = model.encode(pixels)
    return z, model.predict(z[:, :3], actions)


def finite_state(model):
    return [name for name, x in model.state_dict().items() if not torch.isfinite(x).all()]


def difference(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    return dict(passed=bool(torch.allclose(a, b, rtol=1e-4, atol=1e-5)),
                max_abs=float((a-b).abs().max()), rms=float((a-b).square().mean().sqrt()))


@torch.no_grad()
def batch_modes(model, pixels, actions):
    original = copy.deepcopy(model.state_dict())
    result = {}
    for mode in ('saved', 'encoder_batch', 'predictor_batch', 'both_batch'):
        model.load_state_dict(original)
        model.eval()
        for root, active in ((model.projector, mode in ('encoder_batch', 'both_batch')),
                             (model.pred_proj, mode in ('predictor_batch', 'both_batch'))):
            for module in root.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.train(active)
        z, pred = outputs(model, pixels, actions)
        result[mode] = dict(pred_mse=float((pred-z[:, 1:]).square().mean()),
                            latent_std=float(z.flatten(0, 1).std(0).mean()),
                            finite=bool(torch.isfinite(z).all() and torch.isfinite(pred).all()))
    model.load_state_dict(original)
    model.eval()
    return result


@torch.no_grad()
def invariance(model, pixels, actions):
    model.eval()
    reference = outputs(model, pixels, actions)
    repeat = outputs(model, pixels, actions)
    single = [outputs(model, p[None], a[None]) for p, a in zip(pixels, actions)]
    single = tuple(torch.cat([s[i] for s in single]) for i in range(2))
    permutation = torch.arange(len(pixels)-1, -1, -1, device=pixels.device)
    permuted = outputs(model, pixels[permutation], actions[permutation])
    return {label: {name: difference(reference[i], values[i]) for i, name in enumerate(('encoded', 'predicted'))}
            for label, values in (('repeat', repeat), ('single', single),
                                  ('permuted', tuple(v[permutation] for v in permuted)))}


@torch.no_grad()
def calibrate(model, batches):
    """Exact pooled unbiased moments, in dependency order; no parameter updates.

    batches is a factory so the predictor sees the same train windows after the
    encoder's population moments have been fixed. No dropout or BN train mode.
    """
    model.eval()
    changes = {}
    for prefix in ('projector', 'pred_proj'):
        modules = [(name, module) for name, module in getattr(model, prefix).named_modules()
                   if isinstance(module, nn.BatchNorm1d)]
        if len(modules) != 1:
            raise ValueError('A001 expects exactly one BN per projector; revise protocol for another architecture')
        name, module = modules[0]
        count, mean, m2 = 0, None, None

        def accumulate(_, args):
            nonlocal count, mean, m2
            x = args[0].detach().double().cpu()
            size, mu = len(x), x.mean(0)
            varsum = (x-mu).square().sum(0)
            if count == 0:
                mean, m2 = mu, varsum
            else:
                delta = mu-mean
                m2 += varsum+delta.square()*count*size/(count+size)
                mean += delta*size/(count+size)
            count += size

        hook = module.register_forward_pre_hook(accumulate)
        try:
            for pixels, actions in batches():
                dtype = next(model.parameters()).dtype
                pixels, actions = pixels.to(dtype), actions.to(dtype)
                z = model.encode(pixels)
                if prefix == 'pred_proj':
                    model.predict(z[:, :3], actions)
        finally:
            hook.remove()
        if count < 2 or not torch.isfinite(m2).all():
            raise ValueError('Invalid calibration moments')
        variance = m2/(count-1)
        changes[f'{prefix}.{name}'] = dict(count=count,
            mean_change=difference(module.running_mean, mean.float()),
            variance_change=difference(module.running_var, variance.float()))
        module.running_mean.copy_(mean)
        module.running_var.copy_(variance)
        module.num_batches_tracked.zero_()
    return changes


def parity(model, pixels, actions):
    """Compare a frozen-weight eval forward/backward, including every trainable tensor."""
    cpu = copy.deepcopy(model).cpu().eval()
    model.eval()
    results, gradients = [], []
    for current, p, a in ((cpu, pixels.cpu(), actions.cpu()), (model, pixels, actions)):
        current.zero_grad(set_to_none=True)
        z, pred = outputs(current, p, a)
        loss = (pred-z[:, 1:]).square().mean()
        loss.backward()
        results.append((z.detach(), pred.detach()))
        gradients.append({name: value.grad.detach().cpu().clone() for name, value in current.named_parameters()
                          if value.grad is not None})
    forward = {name: difference(results[0][i], results[1][i]) for i, name in enumerate(('encoded', 'predicted'))}
    grads = {name: difference(gradients[0][name], gradients[1][name]) for name in gradients[0]}
    nonfinite = [name for name in gradients[0] if not all(torch.isfinite(g[name]).all() for g in gradients)]
    model.zero_grad(set_to_none=True)
    return dict(forward=forward, gradients=grads, nonfinite_gradients=nonfinite,
                passed=not nonfinite and all(v['passed'] for v in [*forward.values(), *grads.values()]))


def run(checkpoint, root, out, research_id):
    record, _ = require_record(research_id)
    cfg = record['protocol']
    if research_id == 'A003':
        require_training_slot()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Opening before copying pins an inode even while the trainer atomically replaces the path.
    snapshot = out/'source.pt'
    with Path(checkpoint).open('rb') as source, snapshot.open('wb') as dest:
        shutil.copyfileobj(source, dest)
    with gpu_run(out, research_id, cfg, [snapshot]) as manifest:
        torch.cuda.set_per_process_memory_fraction(.80 if research_id == 'A003' else .10)
        model, head, state = load(snapshot, 'cuda:0')
        if cfg.get('inference_dtype') == 'float64':
            model.double()
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        n = state['n']
        invalid = finite_state(model)+([] if head is None else ['head.'+name for name in finite_state(head)])
        result = dict(research_id=research_id, source_sha256=digest(snapshot), step=state.get('step'), n=n,
                      kind=state['kind'], invalid_state=invalid, before={}, after={})
        if result['invalid_state']:
            atomic_json(out/'result.json', result)
            raise RuntimeError('Nonfinite checkpoint state; preserve and investigate affected run')
        for split in ('train', 'val'):
            p, a = next(windows(root, n, split, cfg['diagnostic_windows'], cfg['sampling_seed']))
            result['before'][split] = batch_modes(model, p, a)
        p, a = p[:8], a[:8]
        result['before']['invariance'] = invariance(model, p, a)
        result['before']['parity'] = parity(model, p[:4], a[:4])
        original_parameters = {name: v.detach().cpu().clone() for name, v in model.named_parameters()}
        result['calibration'] = calibrate(model, lambda: windows(root, n, 'train', cfg['calibration_windows'],
                                         cfg['calibration_seed'], cfg['calibration_batch']))
        result['weights_unchanged'] = all(torch.equal(v.cpu(), original_parameters[name])
                                          for name, v in model.named_parameters())
        for split in ('train', 'val'):
            p, a = next(windows(root, n, split, cfg['diagnostic_windows'], cfg['sampling_seed']))
            result['after'][split] = batch_modes(model, p, a)
        result['after']['invariance'] = invariance(model, p[:8], a[:8])
        result['after']['parity'] = parity(model, p[:4], a[:4])
        # Export inference only: a calibrated checkpoint must not masquerade as a resumable trainer state.
        derived = {k: state[k] for k in ('config', 'kind', 'n', 'train', 'head_state_dict')}
        derived.update(model_state_dict=model.cpu().state_dict(), step=state.get('step'),
                       research_id=research_id, calibration=result['calibration'], source_sha256=digest(snapshot),
                       inference_only=True, inference_dtype=cfg.get('inference_dtype', 'float32'),
                       attention_backend='math')
        torch.save(derived, out/'calibrated.pt.partial')
        (out/'calibrated.pt.partial').replace(out/'calibrated.pt')
        result['calibrated_sha256'] = digest(out/'calibrated.pt')
        result['budget_remaining_hours'] = budget_remaining(manifest)
        result['status'] = 'awaiting_literature_review'
        atomic_json(out/'result.json', result)
        print(json.dumps({k: result[k] for k in ('n', 'kind', 'step', 'weights_unchanged', 'status')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--research', required=True)
    args = p.parse_args()
    run(args.checkpoint, args.data, args.out, args.research)


if __name__ == '__main__':
    main()
