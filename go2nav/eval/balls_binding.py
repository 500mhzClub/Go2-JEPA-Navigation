"""C002: frozen RGB slot-mask binding, with labels used only for scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data.balls import load as load_data
from go2nav.research import (atomic_json, budget_remaining, digest, gpu_run, require_record,
                            require_training_slot, reviewed_artifact)
from go2nav.sim.balls import RADIUS, SIZE
from go2nav.training.balls import to_pixels
from go2nav.eval.slots_inference import load_reviewed_encoder


def discs(red, white):
    """Full geometric discs and visible hard-painter masks, ordered red then white.

    Pixel centres/radius follow sim.balls.render. White discs are painted in
    index order and red last. The full discs remain the primary scoring targets.
    """
    positions = np.concatenate((np.asarray(red)[..., None, :], white), -2)
    y, x = np.mgrid[:SIZE, :SIZE]+.5
    full = ((positions[..., 0, None, None]-x)**2+(positions[..., 1, None, None]-y)**2) <= RADIUS**2
    visible = np.zeros_like(full)
    covered = np.zeros(full.shape[:-3]+(SIZE, SIZE), bool)
    for k in [0, *range(full.shape[-3]-1, 0, -1)]:
        visible[..., k, :, :] = full[..., k, :, :] & ~covered
        covered |= full[..., k, :, :]
    return full, visible


def measure(masks, red, white):
    """Per-frame best-IoU binding includes the entire predicted mask's area."""
    masks = np.asarray(masks)
    if not np.isfinite(masks).all():
        raise FloatingPointError('Nonfinite decoder masks')
    full, visible = discs(red, white)
    owner = masks.argmax(-3)
    predicted = owner[..., None, :, :] == np.arange(masks.shape[-3])[:, None, None]
    intersection = (full[..., :, None, :, :] & predicted[..., None, :, :, :]).sum((-1, -2))
    union = full.sum((-1, -2))[..., :, None]+predicted.sum((-1, -2))[..., None, :]-intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    best = iou.argmax(-1)
    value = np.take_along_axis(iou, best[..., None], -1)[..., 0]
    reused = (best[..., :, None] == best[..., None, :]).sum(-1) > 1
    return dict(iou=value, bound=value >= .5, best_slot=best, reused=reused,
                visible_fraction=visible.sum((-1, -2))/full.sum((-1, -2)))


def summarize(rows):
    report = {}
    for name, selected in (('all', rows), ('red', [r for r in rows if r['ball'] == 0]),
                           ('white', [r for r in rows if r['ball'] > 0]),
                           ('at_least_90_percent_visible', [r for r in rows if r['visible_fraction'] >= .9])):
        if not selected:
            report[name] = dict(count=0, bound_share=None, bound_ci95=None, mean_best_iou=None)
            continue
        episodes = sorted({r['episode'] for r in rows})
        positions = {e: i for i, e in enumerate(episodes)}
        sums, counts = np.zeros(len(episodes)), np.zeros(len(episodes))
        for row in selected:
            i = positions[row['episode']]
            sums[i] += row['bound']
            counts[i] += 1
        rng = np.random.default_rng(2026111512)
        choices = rng.integers(len(episodes), size=(2000, len(episodes)))
        denominator = counts[choices].sum(-1)
        samples = sums[choices].sum(-1)[denominator > 0]/denominator[denominator > 0]
        report[name] = dict(count=len(selected), bound_share=float(sums.sum()/counts.sum()),
            bound_ci95=np.quantile(samples, [.025, .975]).tolist(),
            mean_best_iou=float(np.mean([r['iou'] for r in selected])),
            mean_visible_fraction=float(np.mean([r['visible_fraction'] for r in selected])),
            best_slot_reused_share=float(np.mean([r['reused'] for r in selected])))
    return report


@torch.no_grad()
def evaluate(root, n, checkpoint, out, research_id, parent_review):
    record, _ = require_record(research_id)
    cfg = record['protocol']['binding']
    reviewed_artifact(parent_review, checkpoint)
    require_training_slot()
    frames, labels = load_data(root, n, 'test')
    if len(frames) != cfg['episodes']:
        raise ValueError('Binding test split differs from the preregistered episode count')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = [Path(checkpoint), Path(parent_review), Path(root)/f'n{n}'/'test'/'meta.json']
    with gpu_run(out, research_id, dict(stage='binding', n=n, protocol=cfg), inputs) as manifest:
        torch.cuda.set_per_process_memory_fraction(.80)
        model, state = load_reviewed_encoder(checkpoint, parent_review)
        if state['kind'] != 'slots' or state['n'] != n:
            raise ValueError('Binding needs the selected RGB encoder of this ball count')
        episodes = np.repeat(np.arange(len(frames)), cfg['windows_per_episode'])
        rng = np.random.default_rng([cfg['seed'], n, True])
        starts = rng.integers(2, frames.shape[1]-4, size=len(episodes))
        rows = []
        with (out/'cases.jsonl').open('w') as log:
            for i in range(0, len(episodes), 32):
                if budget_remaining(manifest) <= 0:
                    atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete', cases=len(rows)))
                    return
                e, t = episodes[i:i+32], starts[i:i+32]
                pixels = to_pixels(torch.from_numpy(np.array(frames[e[:, None], t[:, None]+np.arange(-2, 1)])).cuda())
                z = model(pixels.to(dtype=next(model.parameters()).dtype))
                _, masks = model.decode(z[:, -1:])
                values = measure(masks[:, 0].cpu().numpy(), labels['red'][e, t], labels['white'][e, t])
                for b in range(len(e)):
                    for k in range(n+1):
                        row = dict(episode=int(e[b]), frame=int(t[b]), ball=k,
                                   **{name: value[b, k].item() for name, value in values.items()})
                        rows.append(row)
                        log.write(json.dumps(row)+'\n')
        atomic_json(out/'result.json', dict(research_id=research_id, kind='slots', n=n, seed=state['seed'],
                    encoder_sha256=digest(checkpoint), cases_sha256=digest(out/'cases.jsonl'),
                    measures=summarize(rows), status='awaiting_literature_review'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'out', 'research', 'parent-review'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--n', type=int, choices=[1, 3, 6], required=True)
    a = p.parse_args()
    evaluate(a.data, a.n, a.checkpoint, a.out, a.research, a.parent_review)


if __name__ == '__main__':
    main()
