"""Export the T2-T4 (native predictor) training samples from the LeWMQuad-v3 run folders (porting tool).

Each original stage saved its exact samples and frame list beside its checkpoint. This tool rewrites them as
data/training/native_predictor/:
- t2_samples.jsonl: 3,518 one-step (500-ms) examples from the native Go2 recordings;
- t3_balanced_start_samples.jsonl: the 48 balanced-start examples T3 adds (T3's samples are T2's plus these);
- t4_samples.jsonl: 29,008 horizon examples (100-800 ms), including the balanced starts;
- frames.txt: every frame used, named <recording folder>/<trial>/rgb_NNNN.png (the folder layout under the native
  recordings root, i.e. the original navigation_development_artifacts_v1 tree).
Each sample names its four frames (contexts 1 s, 0.5 s, 0 s before; the target), the commands (as trained: forward and
yaw) and the normalised control history. The original schedules are not exported: they are regenerated from the seeds
(see go2nav/training/native_predictor.py) and checked against the originals by tests.

Usage: python tools/porting/export_native_predictor_datasets.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'data/training/native_predictor'
T2 = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_vjepa_native_adaptation_v1_attempt_001')
W = Path('/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
T3, T4 = W/'go2_balanced_start_predictor_v1_attempt_001', W/'go2_horizon_dense_predictor_v1_attempt_001'
MARKER = 'navigation_development_artifacts_v1/'


def name(path):
    return path.split(MARKER, 1)[1]


def load(folder):
    return json.loads((folder/'samples.json').read_text()), [name(p) for p in json.loads((folder/'frame_paths.json').read_text())]


def rows(samples, frames, keys):
    out = []
    for s in samples:
        row = {k: s[k] for k in keys if k in s}
        row['frames'] = [frames[i] for i in s['frames']]
        out.append(row)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t2, t2_frames = load(T2)
    t3, t3_frames = load(T3)
    t4, t4_frames = load(T4)
    if t3[:len(t2)] != t2 or t3_frames[:len(t2_frames)] != t2_frames:
        raise ValueError("T3's samples must extend T2's")
    keys = ('sample_id', 'source', 'trial', 'frame', 'horizon', 'action', 'control')
    datasets = dict(t2_samples=rows(t2, t2_frames, keys), t3_balanced_start_samples=rows(t3[len(t2):], t3_frames, keys),
                    t4_samples=rows(t4, t4_frames, keys))
    for file, data in datasets.items():
        (OUT/f'{file}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    frames = sorted({f for data in datasets.values() for r in data for f in r['frames']})
    (OUT/'frames.txt').write_text('\n'.join(frames)+'\n')
    print({k: len(v) for k, v in datasets.items()}, 'frames', len(frames))


if __name__ == '__main__':
    main()
