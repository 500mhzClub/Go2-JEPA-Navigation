"""C003: bounded GPU optimizer recovery and updated-weight numerical checks."""
import argparse
import copy
import math
from pathlib import Path
import time

import torch

from go2nav.eval.balls_inference import difference
from go2nav.models.balls_slots import RGBSlots, ObjectMaskedPredictor, object_mask, predictor_loss
from go2nav.research import atomic_json, budget_remaining, digest, gpu_run, require_record
from go2nav.training.balls_slots import generator, resume, save


def mismatches(expected, actual, path='', *, rtol=1e-6, atol=1e-7):
    """Compare every nested checkpoint field, retaining the paths of all failures."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.shape != actual.shape or expected.dtype != actual.dtype:
            return [path]
        a, b = expected.detach().cpu(), actual.detach().cpu()
        if a.is_floating_point():
            same = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and
                        torch.allclose(a, b, rtol=rtol, atol=atol))
        else:
            same = torch.equal(a, b)
        return [] if same else [path]
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            return [path+'/keys']
        return [bad for k in expected for bad in mismatches(expected[k], actual[k], f'{path}/{k}', rtol=rtol, atol=atol)]
    if isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            return [path+'/length_or_type']
        return [bad for i, value in enumerate(expected)
                for bad in mismatches(value, actual[i], f'{path}/{i}', rtol=rtol, atol=atol)]
    if isinstance(expected, float):
        return [] if isinstance(actual, (float, int)) and math.isfinite(expected) and math.isfinite(actual) \
            and math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol) else [path]
    return [] if expected == actual else [path]


def snapshot(model, optimizer, scheduler):
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        return copy.deepcopy(value)
    return cpu(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))


def components(kind):
    model = (RGBSlots() if kind == 'rgb_slots' else ObjectMaskedPredictor()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if kind == 'rgb_slots' else 5e-4)
    def scale(step):
        if kind != 'rgb_slots':
            return 1.
        return (step+1)/2 if step < 2 else .5*(1+math.cos(math.pi*min(1., (step-2)/4)))
    return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def update(kind, model, optimizer, scheduler, step, check_budget):
    check_budget()
    draws = dict(cpu=torch.rand(7), gpu=torch.rand(7, device='cuda:0').cpu())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if kind == 'rgb_slots':
        pixels = torch.rand(2, 3, 3, 64, 64, device='cuda:0')*2-1
        z = model(pixels, generator=generator(0, 2, step))
        recon, _ = model.decode(z)
        value = (recon-pixels).square().mean()
    else:
        states, actions = torch.randn(2, 4, 5, 128, device='cuda:0'), torch.randn(2, 3, 2, device='cuda:0')
        mask = object_mask(2, 5, generator(0, 3, step), 'cuda:0')
        draws['mask'] = mask.cpu()
        value, _ = predictor_loss(model, states, actions, mask)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .05 if kind == 'rgb_slots' else math.inf)
    if not torch.isfinite(value) or not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite diagnostic update')
    optimizer.step()
    scheduler.step()
    return dict(loss=float(value.detach()), draws=draws)


def inputs(kind):
    rng = torch.Generator().manual_seed(2026111515)
    if kind == 'rgb_slots':
        return [torch.rand(2, 3, 3, 64, 64, generator=rng)*2-1]
    return [torch.randn(2, 4, 5, 128, generator=rng), torch.randn(2, 3, 2, generator=rng),
            torch.tensor([[False, True, False, False, False], [False, False, False, True, True]])]


def evaluate(kind, model, data):
    if kind == 'rgb_slots':
        z = model(data[0])
        recon, masks = model.decode(z)
        return (recon-data[0]).square().mean(), dict(slots=z, reconstruction=recon, masks=masks)
    value, _ = predictor_loss(model, *data)
    return value, dict(predicted=model(data[0][:, :3], data[1], data[2])[0])


def numerics(kind, model, check_budget):
    data, outputs, gradients, missing = inputs(kind), [], [], {}
    for device in ('cpu', 'cuda:0'):
        check_budget()
        current = copy.deepcopy(model).to(device).eval()
        current.zero_grad(set_to_none=True)
        loss, values = evaluate(kind, current, [v.to(device) for v in data])
        loss.backward()
        outputs.append({k: v.detach().cpu() for k, v in values.items()})
        gradients.append({k: p.grad.detach().cpu().clone() for k, p in current.named_parameters() if p.grad is not None})
        missing[device] = [k for k, p in current.named_parameters() if p.requires_grad and p.grad is None]
        del current, loss, values
    result = dict(outputs={k: difference(outputs[0][k], outputs[1][k]) for k in outputs[0]},
                  gradients={k: difference(gradients[0][k], gradients[1][k])
                             for k in gradients[0].keys() & gradients[1].keys()}, missing_gradients=missing)
    model.eval()
    gpu = [v.cuda() for v in data]
    with torch.no_grad():
        _, batched = evaluate(kind, model, gpu)
        _, repeated = evaluate(kind, model, gpu)
        _, reversed_ = evaluate(kind, model, [v.flip(0) for v in gpu])
        separate = [evaluate(kind, model, [v[i:i+1] for v in gpu])[1] for i in range(2)]
    result['grouping'] = {k: dict(repeat=difference(batched[k], repeated[k]),
                                  reversed=difference(batched[k], reversed_[k].flip(0)),
                                  separate=difference(batched[k], torch.cat([row[k] for row in separate])))
                          for k in batched}
    checks = [*result['outputs'].values(), *result['gradients'].values(),
              *(v for row in result['grouping'].values() for v in row.values())]
    result['passed'] = not any(missing.values()) and gradients[0].keys() == gradients[1].keys() \
        and all(v['passed'] for v in checks)
    return result


def check_model(kind, out, seed, check_budget):
    torch.manual_seed(seed)
    model, optimizer, scheduler = components(kind)
    first = update(kind, model, optimizer, scheduler, 0, check_budget)
    identity = dict(research_id='C003', kind=kind, seed=seed, diagnostic=True)
    out.mkdir()
    checkpoint = out/'checkpoint.pt'
    save(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         identity=identity, epoch=0, batch_index=1, best=.75, epoch_loss_sum=first['loss']*2,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    checkpoint_hash = digest(checkpoint)
    expected = [update(kind, model, optimizer, scheduler, step, check_budget) for step in (1, 2)]
    final = snapshot(model, optimizer, scheduler)
    del model, optimizer, scheduler
    model, optimizer, scheduler = components(kind)
    cursor = resume(out, identity, model, optimizer, scheduler)
    actual = [update(kind, model, optimizer, scheduler, step, check_budget) for step in (1, 2)]
    failures = dict(state=mismatches(final, snapshot(model, optimizer, scheduler)),
                    cursor=mismatches((0, 1, .75, first['loss']*2), cursor),
                    losses=mismatches([r['loss'] for r in expected], [r['loss'] for r in actual]),
                    rng=mismatches([r['draws'] for r in expected], [r['draws'] for r in actual], rtol=0, atol=0))
    recovery = dict(passed=not any(failures.values()) and digest(checkpoint) == checkpoint_hash,
                    differences=failures, checkpoint_sha256=checkpoint_hash,
                    checkpoint_unchanged=digest(checkpoint) == checkpoint_hash,
                    continuous_losses=[first['loss'], *[r['loss'] for r in expected]],
                    resumed_losses=[r['loss'] for r in actual])
    result = dict(recovery=recovery, numerics=numerics(kind, model, check_budget))
    result['passed'] = result['recovery']['passed'] and result['numerics']['passed']
    return result


def run(out, research_id):
    record, _ = require_record(research_id)
    if record['id'] != 'C003':
        raise ValueError('This frozen diagnostic requires C003')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with gpu_run(out, research_id, record['protocol']) as manifest:
        torch.cuda.set_per_process_memory_fraction(.10)
        start = time.monotonic()
        def check_budget():
            if budget_remaining(manifest) <= 0 or time.monotonic()-start >= record['resources']['gpu_hours_max']*3600:
                raise TimeoutError('Diagnostic time allowance exhausted')
        result = dict(research_id=research_id, status='awaiting_literature_review')
        try:
            for i, kind in enumerate(('rgb_slots', 'predictor')):
                result[kind] = check_model(kind, out/kind, record['protocol']['seed']+i, check_budget)
                atomic_json(out/'result.json', result)
                print({kind: result[kind]['passed']}, flush=True)
                torch.cuda.empty_cache()
        except TimeoutError:
            result['status'] = 'budget_incomplete'
            atomic_json(out/'result.json', result)
        except BaseException as exc:
            result.update(status='technical_blocked', error=repr(exc))
            atomic_json(out/'result.json', result)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--research', required=True)
    a = p.parse_args()
    run(a.out, a.research)


if __name__ == '__main__':
    main()
