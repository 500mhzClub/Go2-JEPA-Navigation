"""The GPU check before the grid LeWM trains: each case in its own process (a fault kills only that case), three
identical backward passes each; a case passes when every gradient is finite and the passes agree. Cases: the full
loss of go2_v2 and of go2_v2_grid on a real batch, then the grid model's parts (encoder and pooling, predictor,
SIGReg on the grid). Prints one JSON line per case; exit 0 only when go2_v2_grid's full loss passes."""
import json
import subprocess
import sys

import torch

ROOT = '/mnt/workspace_drive/go2nav_lewm_v2'


def run(case):
    from go2nav.data.lewm_dataset import Corpus
    from go2nav.models.lewm import LeWorldModel, SIGReg, losses, normalise_frames
    from go2nav.training.lewm import load_config, model_config
    dev = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.85, dev)
    name = 'go2_v2' if case == 'base' else 'go2_v2_grid'
    model = LeWorldModel.seeded(0, model_config(load_config(f'configs/lewm/{name}.yaml'))).to(dev).train()
    corpus = Corpus(ROOT, length=12, roles=('lewm_train',), command_dim=3)
    frames, actions, _ = corpus.batch(corpus.fixed(32, 0))
    frames, actions = frames.to(dev), actions.to(dev)
    g = torch.Generator(device=dev)
    def loss():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pixels = normalise_frames(frames)
            if case in ('base', 'grid'):
                return losses(model, pixels, actions, rollout_steps=8, sigreg=SIGReg().to(dev), lam=.08,
                              inverse_weight=100., generator=g.manual_seed(0))['loss']
            if case == 'encode':
                return model.encode(pixels).float().square().mean()
            z = torch.randn(32, 12, 13, 192, device=dev, generator=g.manual_seed(1), requires_grad=False)
            if case == 'predict':
                return model.rollout(z[:, :4], actions, 8).float().square().mean()
            return SIGReg().to(dev)(z.flatten(1, 2).transpose(0, 1).requires_grad_(), g.manual_seed(2))
    norms, finite = [], True
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite &= bool(torch.isfinite(value)) and all(bool(torch.isfinite(x).all()) for x in grads)
        norms.append(float(torch.norm(torch.stack([x.float().norm() for x in grads]))) if grads else float(value))
    torch.cuda.synchronize()
    agree = max(norms)-min(norms) <= 1e-3*max(abs(n) for n in norms)
    print(json.dumps(dict(case=case, finite=finite, agree=agree, norms=norms)), flush=True)
    sys.exit(0 if finite and agree else 1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1])
    ok = {}
    for case in ('base', 'grid', 'encode', 'predict', 'sigreg'):
        done = subprocess.run([sys.executable, '-W', 'ignore', __file__, case], capture_output=True, text=True)
        line = next((l for l in done.stdout.splitlines() if l.startswith('{')), None)
        ok[case] = done.returncode == 0
        print(line or json.dumps(dict(case=case, crashed=done.returncode, error=done.stderr.strip().splitlines()[-1:])),
              flush=True)
    sys.exit(0 if ok['grid'] else 1)
