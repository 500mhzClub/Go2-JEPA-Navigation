"""A008 frozen CPU reference; no fitting or model updates."""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from go2nav.data.balls import load as load_data
from go2nav.training.balls import load, to_pixels
from go2nav.research import atomic_json, digest, provenance, require_record, require_review
import go2nav.research
assert Path(go2nav.research.__file__).resolve() == Path.cwd()/'go2nav/research.py'

require_record('A008')
require_review('outputs/gated/A002/review.json')
parent = json.loads(Path('outputs/gated/A002/result.json').read_text())
out = Path('outputs/gated/A008/geometry')
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
assert not torch.cuda.is_initialized()
frames, labels = load_data('outputs/data/balls', 3, 'val')
rng = np.random.default_rng(2026111519)
episodes = np.repeat(np.arange(len(frames)), 5)
times = rng.integers(frames.shape[1], size=len(episodes))
left = rng.integers(len(episodes), size=2000)
right = rng.integers(len(episodes), size=2000)
same = episodes[left] == episodes[right]
while same.any():
    right[same] = rng.integers(len(episodes), size=int(same.sum()))
    same = episodes[left] == episodes[right]
red = labels['red'][episodes, times].astype(np.float64)
white = labels['white'][episodes, times].astype(np.float64)
red_distance = np.linalg.norm(red[left]-red[right], axis=-1)
white_distance = []
for a, b in zip(white[left], white[right]):
    cost = ((a[:, None]-b[None])**2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    white_distance.append(np.sqrt(cost[rows, cols].sum()))
white_distance = np.asarray(white_distance)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
results, inputs = {}, [Path(__file__)]

with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    for i, kind in enumerate(('l0', 'l1')):
        name = f'{kind}_n3_s0'
        checkpoint = Path('outputs/gated/A001')/name/'calibrated.pt'
        checkpoint_hash = digest(checkpoint)
        assert parent['cases'][name]['source_sha256'] == checkpoint_hash
        assert parent['cases'][name]['modes']['float64_math']['passed']
        model, _, state = load(checkpoint, 'cpu')
        assert state['step'] == 40000 and state['n'] == 3 and state['kind'] == kind
        model = model.double().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        def encode(indices):
            pixels = to_pixels(torch.from_numpy(np.array(frames[episodes[indices], times[indices]]))).double()
            return model.encode(pixels[:, None])[:, 0]
        z = []
        for start in range(0, len(episodes), 32):
            assert time.monotonic()-started < 180, 'A008 CPU time cap'
            z.append(encode(np.arange(start, min(start+32, len(episodes)))))
        z = torch.cat(z)
        singles = torch.cat([encode(np.array([j])) for j in range(8)])
        reverse = encode(np.arange(7, -1, -1)).flip(0)
        torch.testing.assert_close(z[:8], singles, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(z[:8], reverse, rtol=1e-4, atol=1e-5)
        assert torch.isfinite(z).all()
        latent = np.linalg.norm(z.numpy()[left]-z.numpy()[right], axis=-1)
        assert np.isfinite(latent).all()
        unchanged = digest(checkpoint) == checkpoint_hash and all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        assert unchanged and not torch.cuda.is_initialized()
        results[kind] = dict(step=40000, source_sha256=checkpoint_hash,
            red_spearman=float(spearmanr(latent, red_distance).statistic),
            white_set_spearman=float(spearmanr(latent, white_distance).statistic),
            mean_latent_std=float(z.std(0).mean()), mean_latent_pair_distance=float(latent.mean()),
            grouping_passed=True, single_max_abs=float((z[:8]-singles).abs().max()),
            unchanged_weights_and_checkpoint=unchanged)
        np.savez_compressed(out/f'{kind}_cases.npz', episodes=episodes, times=times, left=left, right=right,
                            latent=z.numpy(), latent_distance=latent, red_distance=red_distance,
                            white_distance=white_distance)
        results[kind]['cases_sha256'] = digest(out/f'{kind}_cases.npz')
        for j, (physical, label, rho) in enumerate(((red_distance, 'Red ball', results[kind]['red_spearman']),
                                                  (white_distance, 'White set (matched)', results[kind]['white_set_spearman']))):
            ax = axes[i, j]
            ax.hexbin(physical, latent, gridsize=30, mincnt=1, cmap='viridis')
            ax.set_title(f'{kind.upper()} / {label}: Spearman rho={rho:.3f}')
            ax.set_xlabel('Physical position distance (pixels)')
            ax.set_ylabel('Latent L2 distance')
        inputs.append(checkpoint)
fig.suptitle('Matched step 40,000: validation geometry, not a planning score')
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(out/'latent_geometry.png', dpi=180)
fig.savefig(out/'latent_geometry.pdf')
atomic_json(out/'manifest.json', provenance('A008', dict(stage='frozen_cpu_reference'), inputs))
atomic_json(out/'result.json', dict(research_id='A008', status='awaiting_literature_review',
    observations=len(episodes), episodes=len(np.unique(episodes)), pairs=2000, cases=results,
    elapsed_seconds=time.monotonic()-started, gpu_seconds=0, cuda_initialized=False,
    limitation='Dependent pairs; no independent-pair significance claim or information-absence proof',
    figure_sha256=digest(out/'latent_geometry.png')))
print(json.dumps(results, indent=2))
