"""Fit the C3 decoder and C4 with the ported code on the ORIGINAL feature cache, and compare with the deployed fit.

The original stage-2 cache (LeWMQuad-v3, stage2_feature_cache_v1) holds the decoder fix's contexts first, with the same
frame and pair rows the original fits used. This adapter reads its arrays in place, runs `go2nav.training.decoder.fit`
with the deployed seed and recipe, and prints the selection score and per-type errors beside the original record.

Usage: python tools/porting/check_decoder_fit_against_original.py --out DIR [--recipe stage2]  (stage2: seed 2026093012)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from go2nav.training import decoder as d
from go2nav.training.common import control_stats
import original

ROOT = original.path('recovery', '.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
CACHE = ROOT/'stage2_feature_cache_v1'
ORIGINAL_FIT = ROOT/'dev_decoder_fits/p3_large_past_frames_s2026093011.json'
T5A = original.path('workspace', '.generated/navigation_development_artifacts_v1/go2_full_heading_readout_v1_attempt_001/mixed_data_final.pt')
MARKER = 'navigation_development_artifacts_v1/'


class OriginalCache(d.DecoderCache):
    def __init__(self, contexts):
        self.folder, self.contexts = CACHE, contexts
        frames = [p.split(MARKER, 1)[1] for p in json.loads((CACHE/'frame_rows.json').read_text())]
        self.frame_row = {p: i for i, p in enumerate(frames)}
        self.pair_row = {(i, h): k for k, (i, h) in enumerate(json.loads((CACHE/'pairs.json').read_text()))}
        self.frames = np.load(CACHE/'frames.f16.npy', mmap_mode='r')
        self.pred = np.load(CACHE/'pred.f16.npy', mmap_mode='r')
        self.mean, self.std = control_stats()


STAGE2_FIT = ROOT/'stage2_decoder_fits/stage2_p3_large_past_frames_s2026093012.json'


def main(out, recipe):
    cache = OriginalCache(d.load_contexts())
    seed, record = (2026093011, ORIGINAL_FIT) if recipe == 'decoder_fix' else (2026093012, STAGE2_FIT)
    checkpoint = d.fit(cache, T5A, out, seed=seed, recipe=recipe)
    ported = json.loads((Path(out)/'evaluation.json').read_text())
    original = json.loads(record.read_text())
    print('selection score (mm): ported', round(ported['score_mm'], 3))
    keys = ('eval_onpolicy@800ms/C3', 'eval_onpolicy@800ms/C4', 'eval_transfer@700ms/C3', 'eval_offline@800ms/C4')
    if recipe == 'stage2':
        keys += ('eval_patch@800ms/C3', 'eval_patch@800ms/C4')
    for key in keys:
        for c in ('all',)+d.CATEGORIES+tuple(f'bin:{b}' for b in d.EDGE_BINS):
            a, b = original['eval'][key].get(c), ported['eval'][key].get(c)
            if a and b:
                print(f"{key:26s} {c:11s} median xy {a['median_xy_mm']:7.3f} -> {b['median_xy_mm']:7.3f}   ratio {a['median_ratio']} -> {b['median_ratio']}")
    print(checkpoint)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--recipe', choices=['decoder_fix', 'stage2'], default='decoder_fix')
    a = p.parse_args()
    main(a.out, a.recipe)
