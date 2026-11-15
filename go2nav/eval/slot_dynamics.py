"""D001: paired offline prediction with recorded engineered/slot observations.

No tracker is rerun. Every initially visible walker yields a row, including misses.
Legacy future-visible and after-occlusion cohorts are reported separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from go2nav.eval.mover_probe import in_view
from go2nav.research import (atomic_json, budget_remaining, digest, gpu_run, require_record,
                            reviewed_artifact, require_training_slot, verify_files)

HORIZONS = (16, 32, 64)
ARMS = ('m5_engineered', 'm5_slots', 's1_slots', 'cv_slots', 'persistence_slots')


def match(xy, seen, truth, visible, threshold=.3):
    """One-to-one observations within 0.3m; labels are used only here for scoring."""
    slots, walkers = np.flatnonzero(seen & np.isfinite(xy).all(-1)), np.asarray(visible, int)
    if not len(slots) or not len(walkers):
        return {}
    costs = np.linalg.norm(xy[slots, None]-truth[None, walkers], axis=-1)
    rows, cols = linear_sum_assignment(np.where(costs <= threshold, costs, 1e6))
    return {int(walkers[j]): int(slots[i]) for i, j in zip(rows, cols) if costs[i, j] <= threshold}


def percentiles(values):
    if not values:
        return dict(count=0, p50=None, p90=None, p95=None)
    q = np.quantile(values, [.5, .9, .95])
    return dict(count=len(values), **dict(zip(('p50', 'p90', 'p95'), map(float, q))))


def summarize(rows, detections):
    report = dict(detection={}, prediction={})
    for source in ('engineered', 'slots'):
        sel = [r for r in detections if r['source'] == source]
        counts = {k: sum(r[k] for r in sel) for k in ('eligible', 'found', 'predictions', 'switches', 'fragments',
                                                     'near', 'near_in_view')}
        report['detection'][source] = counts | dict(recall=counts['found']/max(1, counts['eligible']),
             precision=counts['found']/max(1, counts['predictions']),
             in_view_share_of_near=counts['near_in_view']/max(1, counts['near']))
    for cohort in ('initially_visible', 'legacy_visible', 'after_occlusion'):
        report['prediction'][cohort] = {}
        for horizon in HORIZONS:
            sel = [r for r in rows if r['horizon'] == horizon and
                   (cohort == 'initially_visible' or (r['future_visible'] and r['future_near'] if cohort == 'legacy_visible'
                                                    else not r['future_visible']))]
            for waiting in (False, True):
                subset = [r for r in sel if r['waiting']] if waiting else sel
                common = [r for r in subset if all(r['error'][a] is not None for a in ARMS)]
                part = dict(eligible=len(subset), common_count=len(common), arms={})
                for arm in ARMS:
                    found = [r['error'][arm] for r in subset if r['error'][arm] is not None]
                    part['arms'][arm] = dict(coverage=len(found)/max(1, len(subset)),
                        conditional=percentiles(found), common=percentiles([r['error'][arm] for r in common]))
                report['prediction'][cohort][f'{horizon/10:.1f}s'+('_waiting' if waiting else '')] = part
    recall = report['detection']['slots']['recall']/max(report['detection']['engineered']['recall'], 1e-12)
    def median(arm, h):
        return report['prediction']['legacy_visible'][h]['arms'][arm]['common']['p50']
    values = [median(a, h) for a in ARMS for h in ('3.2s', '6.4s')]
    valid = all(v is not None for v in values)
    m5, s1, ceiling = median('m5_slots', '3.2s'), median('s1_slots', '3.2s'), median('m5_engineered', '3.2s')
    passing_arms = [a for a in ('m5_slots', 's1_slots') if valid and median(a, '3.2s') <= 1.5*ceiling
                    and all(median(a, h) < median('cv_slots', h) for h in ('3.2s', '6.4s'))]
    report['gates'] = dict(found_ratio=recall, found_pass=recall >= .9,
                           prediction_passing_arms=passing_arms, passed=recall >= .9 and bool(passing_arms))
    return report


def evaluate(corpus, m5, s1, mazes, out, research_id, tracks_review, s1_review):
    require_record(research_id)
    corpus, out = Path(corpus), Path(out)
    reviewed_artifact(tracks_review, corpus/'tracks_index.json', 'tracks_sha256')
    verify_files(corpus, json.loads((corpus/'tracks_index.json').read_text())['files'])
    reviewed_artifact(s1_review, s1)
    require_training_slot()
    out.mkdir(parents=True, exist_ok=False)
    from go2nav.eval.mission import episode_spec
    from go2nav.planning.fields import Fields
    from go2nav.planning.object_model import ObjectModel
    from go2nav.planning.sampling import carried
    from go2nav.sim.kinematic import Geometry
    rows, detections = [], []
    with gpu_run(out, research_id, dict(stage='offline'), [Path(m5), Path(s1), corpus/'tracks_index.json']) as manifest:
        for meta_path in sorted((corpus/'val').glob('*.json')):
            if budget_remaining(manifest) <= 0:
                atomic_json(out/'result.json', dict(research_id=research_id, status='budget_incomplete', cases=len(rows)))
                return
            meta = json.loads(meta_path.read_text())
            with np.load(meta_path.with_suffix('.npz')) as arrays:
                pose, requested = arrays['pose'], arrays['requested']
                truth, waiting, sight = arrays['walkers_xy'], arrays['waiting'], arrays['in_sight']
            tracks = {key: dict(np.load(corpus/folder/'val'/meta_path.with_suffix('.npz').name))
                      for key, folder in (('engineered', 'engineered'), ('slots', 'objects'))}
            spec, _ = episode_spec(meta['set'], meta['maze'], meta['home_episode'], root=Path(mazes))
            geometry = Geometry(spec)
            fields = Fields(geometry)
            models = {key: ObjectModel(path, device='cuda:0') for key, path in
                      (('m5_engineered', m5), ('m5_slots', m5), ('s1_slots', s1))}
            for model in models.values():
                model.bind(fields)
            previous = {key: {} for key in tracks}
            ever = {key: set() for key in tracks}
            for t in range(len(pose)):
                visible = [j for _, j in in_view(pose[t], truth[t], geometry.visible)]
                mappings = {}
                for key, values in tracks.items():
                    predicted_visible = [j for _, j in in_view(pose[t], values['xy'][t], geometry.visible)
                                         if values['seen'][t, j]]
                    eligible_predictions = np.zeros_like(values['seen'][t])
                    eligible_predictions[predicted_visible] = True
                    mappings[key] = match(values['xy'][t], eligible_predictions, truth[t], visible)
                    current = {j: int(values['id'][t, k]) for j, k in mappings[key].items()}
                    near = np.isfinite(truth[t]).all(-1) & (np.linalg.norm(truth[t]-pose[t, :2], axis=-1) <= 3)
                    detections.append(dict(episode=meta['index'], frame=t, source=key, eligible=len(visible),
                        found=len(current), predictions=len(predicted_visible),
                        switches=sum(j in previous[key] and previous[key][j] != ident for j, ident in current.items()),
                        fragments=sum(j in ever[key] and j not in previous[key] for j in current),
                        near=int(near.sum()), near_in_view=sum(bool(near[j]) for j in visible)))
                    previous[key] = current
                    ever[key].update(current)
                for key, model in models.items():
                    model.observe_tracks(tracks['engineered' if key == 'm5_engineered' else 'slots'], t, pose[t])
                if t >= 8 and t % 8 == 0 and t+64 < len(pose) and visible:
                    tape = requested[t:t+64][None]
                    future_xy, future_yaw = pose[t+1:t+65, :2][None], pose[t+1:t+65, 2][None]
                    predicted = {key: model.walkers(None, future_xy, future_yaw, None, geometry, requested=tape)[0]
                                 for key, model in models.items()}
                    slots = tracks['slots']
                    predicted['cv_slots'] = carried(np.nan_to_num(slots['xy'][t]), slots['velocity'][t], 1, 64, geometry)[0]
                    predicted['persistence_slots'] = np.broadcast_to(slots['xy'][t], (64, *slots['xy'][t].shape))
                    for j in visible:
                        for h in HORIZONS:
                            errors = {}
                            for arm in ARMS:
                                k = mappings['engineered' if arm == 'm5_engineered' else 'slots'].get(j)
                                value = None if k is None else float(np.linalg.norm(predicted[arm][h-1, k]-truth[t+h, j]))
                                errors[arm] = value if value is not None and np.isfinite(value) else None
                            rows.append(dict(research_id=research_id, episode=meta['index'], frame=t, walker=j,
                                horizon=h, waiting=bool(waiting[t, j]), future_visible=bool(sight[t+h, j]),
                                future_near=bool(np.linalg.norm(truth[t+h, j]-pose[t+h, :2]) <= 3), error=errors))
                for model in models.values():
                    model.record(requested[t])
            print(meta['index'], 'offline unit scored', flush=True)
        for name, values in (('cases', rows), ('detection', detections)):
            with (out/f'{name}.jsonl').open('w') as f:
                for row in values:
                    f.write(json.dumps(row)+'\n')
        report = summarize(rows, detections) | dict(research_id=research_id, status='awaiting_literature_review',
                    cases_sha256=digest(out/'cases.jsonl'), detection_sha256=digest(out/'detection.jsonl'))
        atomic_json(out/'result.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'm5', 's1', 'mazes', 'out', 'research', 'tracks-review', 's1-review'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    evaluate(a.corpus, a.m5, a.s1, a.mazes, a.out, a.research, a.tracks_review, a.s1_review)


if __name__ == '__main__':
    main()
