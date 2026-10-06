"""Consolidate the T1 (temporal predictor) training rows from the LeWMQuad-v3 caches into one file (porting tool).

The original T1 data is spread over five layers (temporal rows, two-step rows, proprio rows, a canonical cache map and
a factorial manifest), with three index spaces. This tool resolves every factorial row once and writes
data/training/t1_temporal/rows.jsonl, one row per training or checkpoint-selection example:

    id, split, family, episode_cluster,
    frames: {ctx0, ctx1, ctx2, y1, y2}  paths relative to the May-corpus render root (render_textured_v03)
             ctx0/ctx1/ctx2: the frames 1 s, 0.5 s and 0 s before the action; y1: 0.5 s after; y2: 1 s after
    a1, a2: the first and second 500-ms action blocks (5 commands x (forward, yaw), as recorded)
    control: the applied-command history, (3, 5, 2), raw (normalised at training time)

Every resolved row is checked against the factorial row by its pair hash. Rows are written in factorial order.

Usage: python tools/porting/export_t1_dataset.py [--cache ~/.cache/lewm_go2_temporal_v03]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/training/t1_temporal'
RENDER_ROOT_MARKER = 'render_textured_v03/'


def relative(path):
    return path.split(RENDER_ROOT_MARKER, 1)[1]


def main(cache):
    cache = Path(cache).expanduser()
    temporal = [json.loads(l) for l in (cache/'temporal_rows.jsonl').read_text().splitlines() if l.strip()]
    two_step = [json.loads(l) for l in (cache/'two_step/two_step_rows.jsonl').read_text().splitlines() if l.strip()]
    proprio = [json.loads(l) for l in (cache/'proprio_v1/proprio_rows.jsonl').read_text().splitlines() if l.strip()]
    factorial = json.loads((cache/'proprio_v1/factorial_manifest.json').read_text())
    by_split = {split: [r for r in temporal if r['role'] == split] for split in ('train', 'checkpoint_selection')}
    two_by_split = {split: [r for r in two_step if r['role'] == split] for split in ('train', 'checkpoint_selection')}
    rows = []
    for entry in factorial['rows']:
        split = 'train' if entry['split'] == 'train' else 'checkpoint_selection'
        index = entry['cache_index']
        base = by_split[split][index] if index < len(by_split[split]) else None
        if base is None or base['pair_sha256'] != entry['pair_sha256']:
            base = temporal[index]  # an absolute index into the whole file
        step2 = two_by_split[split][entry['step2_cache_index']]
        source = proprio[entry['manifest_row_index']]
        if not (base['pair_sha256'] == step2['pair_sha256'] == source['pair_sha256'] == entry['pair_sha256']):
            raise ValueError(f"row {entry['stable_row_id']}: pair hashes disagree across layers")
        offsets = {f['offset']: f['path'] for f in base['frames']}
        rows.append(dict(
            id=entry['stable_row_id'], split=split, family=entry['family'], episode_cluster=entry['episode_cluster'],
            frames=dict(ctx0=relative(offsets[-480]), ctx1=relative(offsets[-240]), ctx2=relative(offsets[0]),
                        y1=relative(base['target_path']), y2=relative(step2['step2_path'])),
            a1=source['action_blocks'][0], a2=source['action_blocks'][min(1, len(source['action_blocks'])-1)],
            control=source['control']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    frames = sorted({p for r in rows for p in r['frames'].values()})
    (OUT/'frames.txt').write_text('\n'.join(frames)+'\n')
    print(dict(rows=len(rows), train=sum(r['split'] == 'train' for r in rows), frames=len(frames)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='~/.cache/lewm_go2_temporal_v03')
    main(p.parse_args().cache)
