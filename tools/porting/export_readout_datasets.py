"""Export the T5 (motion readout) training data from the LeWMQuad-v3 run folders (porting tool).

Writes data/training/readouts/:
- t5_original.jsonl: T5-0, 3,518 (current frame, frame 500 ms later, true body-frame motion) examples. The targets are
  recomputed with the original code (frozen snapshot) and checked against the hash T5-0 recorded.
- t5_full_heading.jsonl: T5a's new examples (wider headings, visible-robot rendering), same form.
- t5_maze_view.json: T5b's data: `old` contexts (original and full-heading, at 8 horizons) and `maze` contexts (the
  maze-view collection, at 8 horizons); per context 8 (current, future) frame pairs and 8 motions.
- frames.txt: every frame, named <recording folder>/<trial>/rgb_NNNN.png.

Usage: GO2NAV_SNAPSHOT=/path/to/snapshot python tools/porting/export_readout_datasets.py  (programme environment)
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import original

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/training/readouts'
MARKER = 'navigation_development_artifacts_v1/'
R = original.path('recovery', '.generated/navigation_development_artifacts_v1')
W = original.path('workspace', '.generated/navigation_development_artifacts_v1')
ST = original.path('steam', 'navigation_development_artifacts_v1')


def name(path):
    return str(path).split(MARKER, 1)[1]


def main():
    snapshot = Path(os.environ['GO2NAV_SNAPSHOT'])
    sys.path[:0] = [str(snapshot), str(snapshot/'lewm_genesis'), str(snapshot/'lewm_worlds')]
    os.chdir(snapshot)
    from scripts import train_go2_dense_visual_motion_readout_development as original
    samples, targets = original.training_data()
    plan = json.loads((ST/'go2_dense_visual_motion_readout_v1_attempt_001/plan.json').read_text())
    if hashlib.sha256(targets.tobytes()).hexdigest() != plan['target_sha256']:
        raise ValueError('T5-0 targets differ from the recorded ones')
    t2_frames = json.loads((ST/'go2_frozen_vjepa_native_adaptation_v1_attempt_001/frame_paths.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    t5_original = [dict(current=name(t2_frames[s['frames'][2]]), future=name(t2_frames[s['frames'][3]]), motion=t.tolist())
                   for s, t in zip(samples, targets)]
    heading = json.loads((W/'go2_full_heading_training_v1_attempt_001/samples.json').read_text())
    t5_heading = [dict(current=name(r['current_rgb']), future=name(r['future_rgb']), motion=r['motion']) for r in heading]
    maze_folder = R/'go2_maze_view_readout_v1_attempt_003'
    data = json.loads((maze_folder/'samples.json').read_text())
    paths = [name(p) for p in json.loads((maze_folder/'frame_paths.json').read_text())]
    maze_view = {group: dict(pairs=[[[paths[a], paths[b]] for a, b in ctx] for ctx in data[f'{group}_pairs']],
                             motions=data[f'{group}_targets']) for group in ('old', 'maze')}
    (OUT/'t5_original.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in t5_original))
    (OUT/'t5_full_heading.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in t5_heading))
    (OUT/'t5_maze_view.json').write_text(json.dumps(maze_view)+'\n')
    frames = {r[k] for rows in (t5_original, t5_heading) for r in rows for k in ('current', 'future')}
    frames |= {f for g in maze_view.values() for ctx in g['pairs'] for pair in ctx for f in pair}
    (OUT/'frames.txt').write_text('\n'.join(sorted(frames))+'\n')
    print(dict(original=len(t5_original), heading=len(t5_heading), old=len(maze_view['old']['pairs']),
               maze=len(maze_view['maze']['pairs']), frames=len(frames)))


if __name__ == '__main__':
    main()
