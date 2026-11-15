"""C001's predeclared small-batch CPU/GPU forward/backward verification."""
import argparse
import copy
from pathlib import Path

import torch

from go2nav.eval.balls_inference import difference
from go2nav.models.balls_slots import RGBSlots, ObjectMaskedPredictor, predictor_loss
from go2nav.research import atomic_json, gpu_run


def compare(model, inputs, forward):
    outputs, gradients = [], []
    for device in ('cpu', 'cuda:0'):
        current = copy.deepcopy(model).to(device).eval()
        value, result = forward(current, *[x.to(device) for x in inputs])
        value.backward()
        outputs.append(result.detach().cpu())
        gradients.append({k: p.grad.detach().cpu() for k, p in current.named_parameters() if p.grad is not None})
    output = difference(*outputs)
    grad = {k: difference(gradients[0][k], gradients[1][k]) for k in gradients[0]}
    return dict(output=output, gradients=grad, passed=output['passed'] and all(v['passed'] for v in grad.values()))


def run(out, research_id):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with gpu_run(out, research_id, dict(stage='numerical_verification', batch=2, frames=3)):
        torch.cuda.set_per_process_memory_fraction(.10)
        torch.manual_seed(2026111509)
        pixels = torch.rand(2, 3, 3, 64, 64)*2-1
        def rgb(model, x):
            z = model(x)
            recon, _ = model.decode(z)
            return (recon-x).square().mean(), z
        result = dict(research_id=research_id, rgb_slots=compare(RGBSlots(), [pixels], rgb))
        states, actions = torch.randn(2, 4, 5, 128), torch.randn(2, 3, 2)
        mask = torch.tensor([[False, True, False, False, False], [False, False, False, True, True]])
        def predicted(model, z, a, m):
            loss, _ = predictor_loss(model, z, a, m)
            return loss, model(z[:, :3], a, m)[0]
        result['predictor'] = compare(ObjectMaskedPredictor(), [states, actions, mask], predicted)
        result['status'] = 'awaiting_literature_review'
        atomic_json(out/'result.json', result)
        print({k: v['passed'] for k, v in result.items() if isinstance(v, dict)}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--research', required=True)
    a = p.parse_args()
    run(a.out, a.research)


if __name__ == '__main__':
    main()
