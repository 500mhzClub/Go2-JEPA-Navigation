"""Export the T7/T8 (C3 decoder and C4) contexts from the LeWMQuad-v3 stage-2 feature cache (porting tool).

The stage-2 cache holds every context the decoder fits used: the decoder fix's 34,745 (training groups old, maze and
on-policy; evaluation sets eval_onpolicy, eval_offline, eval_transfer, eval_fresh_c3) followed by the 8,520 stage-2
contexts (training groups patch and patch_off; evaluation set eval_patch). This writes data/training/decoder/:
- contexts.jsonl: per context its set, group, movement category, the three frames (1 s, 0.5 s, 0 s before the
  decision), the executed command tape (8 x [vx, vy, yaw]), the applied-command history (15 x 3), the true motion at
  100-800 ms (8 x 3; NaN where unknown), its horizons (two random ones for training, fixed for evaluation), the stage
  (decoder_fix or stage2) and any extra fields (edge bin, run, ...);
- frames.txt: every frame, named by its path below navigation_development_artifacts_v1/.

Usage: python tools/porting/export_decoder_dataset.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/training/decoder'
CACHE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
             'go2_navigation_capability_v1_attempt_001/stage2_feature_cache_v1')
MARKER = 'navigation_development_artifacts_v1/'
DECODER_FIX_CONTEXTS = 34745
KEPT = ('set', 'group', 'category', 'tape', 'past', 'targets', 'horizons')


def main():
    items = json.loads((CACHE/'items.json').read_text())
    rows = json.loads((CACHE/'frame_rows.json').read_text())
    if any(MARKER not in p for p in rows):
        raise ValueError('every frame must lie under navigation_development_artifacts_v1')
    names = [p.split(MARKER, 1)[1] for p in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'contexts.jsonl').open('w') as stream:
        for i, item in enumerate(items):
            row = {k: item.get(k) for k in KEPT}
            row['frames'] = [names[r] for r in item['frame_rows']]
            row['stage'] = 'decoder_fix' if i < DECODER_FIX_CONTEXTS else 'stage2'
            row['extra'] = {k: v for k, v in item.items() if k not in KEPT+('frame_rows', 'source', 'frame')}
            stream.write(json.dumps(row)+'\n')
    (OUT/'frames.txt').write_text('\n'.join(sorted(set(names)))+'\n')
    print(dict(contexts=len(items), frames=len(set(names))))


if __name__ == '__main__':
    main()
