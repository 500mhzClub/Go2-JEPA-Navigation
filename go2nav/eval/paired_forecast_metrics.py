"""F004 physical metrics and paired layout bootstrap; no inference or simulator.

Predictions and targets are world coordinates (scene, future, step, slot, xy).
Futures0..5 are alternative actions; future6 is factual. Inference artifacts must
bind this order separately. Missing/degenerate statistics are null, never a pass.
"""
import itertools

import numpy as np

HORIZONS = {1.6: 3, 3.2: 7, 6.4: 15}
PAIRS = np.array(list(itertools.combinations(range(6), 2)))
MODES = {'factual': slice(6, 7), 'branches': slice(0, 6), 'pooled': slice(None)}
STRATA = ('all', 'rude', 'polite', 'reactive', 'none', 'waiting', 'not_waiting')
KINDS = {'none': 0, 'rude': 1, 'polite': 2, 'reactive': 3}


def need(value, message):
    if not value:
        raise ValueError(message)


class LayoutBootstrap:
    """Resample complete layouts; sufficient statistics preserve every within-layout row."""
    def __init__(self, layouts, draws=2000, seed=2026111604):
        self.layouts = np.unique(layouts)
        need(len(self.layouts) > 0 and draws >= 1, 'Empty bootstrap population')
        choices = np.random.default_rng(seed).integers(len(self.layouts), size=(draws, len(self.layouts)))
        self.frequency = np.stack([np.bincount(x, minlength=len(self.layouts)) for x in choices])
        self.draws = draws

    def sums(self, values, groups):
        values, groups = np.asarray(values, float), np.asarray(groups)
        need(len(values) == len(groups) and np.isfinite(values).all(), 'Invalid grouped values')
        index = np.searchsorted(self.layouts, groups)
        need(np.all(index < len(self.layouts)), 'Unknown layout')
        need(np.array_equal(self.layouts[index], groups), 'Unknown layout')
        return np.bincount(index, weights=values, minlength=len(self.layouts))

    def mean_draws(self, values, groups):
        count = self.frequency @ self.sums(np.ones(len(values)), groups)
        total = self.frequency @ self.sums(values, groups)
        return np.divide(total, count, out=np.full(self.draws, np.nan), where=count > 0)

    def interval(self, draws, clusters):
        valid = np.isfinite(draws)
        # A single contributing layout cannot establish between-layout uncertainty.
        return dict(ci95=np.quantile(draws[valid], [.025, .975]).tolist()
                    if valid.sum() >= min(1900, self.draws) and clusters >= 2 else None,
                    valid_draws=int(valid.sum()), missing_draws=int((~valid).sum()))

    def describe(self, values, groups):
        values = np.asarray(values, float)
        clusters = len(np.unique(groups))
        need(values.ndim == 1 and np.isfinite(values).all(), 'Nonfinite statistic')
        interval = self.interval(self.mean_draws(values, groups), clusters)
        return dict(count=len(values), layouts=clusters,
                    mean=float(values.mean()) if len(values) else None,
                    median=float(np.median(values)) if len(values) else None,
                    p95=float(np.quantile(values, .95)) if len(values) else None, **interval)

    def regression(self, true, predicted, groups):
        """Shared slope and per-axis intercept; use one column for magnitude regression."""
        true, predicted = np.asarray(true, float), np.asarray(predicted, float)
        need(true.ndim == 2 and predicted.shape == true.shape and len(groups) == len(true), 'Bad regression shape')
        need(np.isfinite(true).all() and np.isfinite(predicted).all(), 'Nonfinite regression')
        n = self.sums(np.ones(len(true)), groups)
        tx = np.stack([self.sums(true[:, k], groups) for k in range(true.shape[1])], 1)
        py = np.stack([self.sums(predicted[:, k], groups) for k in range(true.shape[1])], 1)
        xx = self.sums(np.square(true).sum(1), groups)
        xy = self.sums((true*predicted).sum(1), groups)
        weights = np.concatenate((np.ones((1, len(n))), self.frequency), 0)
        count, sx, sy = weights@n, weights@tx, weights@py
        safe_count = np.maximum(count, 1)
        variance = weights@xx - np.square(sx).sum(1)/safe_count
        covariance = weights@xy - (sx*sy).sum(1)/safe_count
        usable = (count > 0) & (variance > 1e-12)
        slope = np.divide(covariance, variance, out=np.full(len(count), np.nan), where=usable)
        intercept = (sy-slope[:, None]*sx)/safe_count[:, None]
        clusters = int((n > 0).sum())
        return dict(count=len(true), layouts=clusters, slope=float(slope[0]) if usable[0] else None,
                    intercept=intercept[0].tolist() if usable[0] else None,
                    offset_m=float(np.linalg.norm(intercept[0])) if usable[0] else None,
                    **self.interval(slope[1:], clusters))


def validate_data(data):
    n = len(data['maze'])
    need(n > 0 and data['target'].shape == (n, 7, 16, 6, 2), 'Wrong target population/shape')
    need(data['valid'].shape == (n, 7, 16, 6) and data['valid'].dtype == bool, 'Wrong validity mask')
    need(data['walker'].shape == data['waiting'].shape == (n, 6), 'Wrong initial identity/stratum shape')
    need(data['future_commands'].shape == (n, 7, 64, 3), 'Wrong command axes')
    need(np.isfinite(data['target']).all(), 'Nonfinite physical targets')
    need(not data['target'][~data['valid']].any(), 'Unmasked invalid targets')
    need(not (data['valid'] & (data['walker'][:, None, None] < 0)).any(), 'Unmatched target counted')
    need(np.all(np.isin(data['kind'], list(KINDS.values()))), 'Unknown behavior label')
    keys = np.stack((data['episode'], data['t0']), 1)
    need(len(np.unique(keys, axis=0)) == n, 'Duplicate scene keys')


def validate_prediction(prediction, data):
    need(prediction.shape == data['target'].shape and np.issubdtype(prediction.dtype, np.floating), 'Wrong prediction shape/type')
    need(np.isfinite(prediction).all(), 'Nonfinite prediction, including unmatched slots')


def baselines(data):
    current = np.nan_to_num(data['history_xy'][:, -1]).astype(np.float64)
    velocity = data['history_velocity'][:, -1].astype(np.float64)*data['history_seen'][:, -1, :, None]
    times = .4*np.arange(1, 17)
    cv = current[:, None] + times[None, :, None, None]*velocity[:, None]
    shape = data['target'].shape
    return dict(persistence=np.broadcast_to(current[:, None, None], shape),
                cv=np.broadcast_to(cv[:, None], shape))


def strata(row, name):
    if name == 'all':
        return np.ones(len(row['layout']), bool)
    if name in ('waiting', 'not_waiting'):
        return row['waiting'] if name == 'waiting' else ~row['waiting']
    return row['kind'] == KINDS[name]


def trajectory_rows(data, prediction, endpoint, mode):
    selection = MODES[mode]
    target = data['target'][:, selection, :endpoint+1].astype(np.float64)
    error = np.linalg.norm(prediction[:, selection, :endpoint+1].astype(np.float64)-target, axis=-1)
    valid = data['valid'][:, selection, :endpoint+1]
    count = valid.sum(2)
    ade = np.divide((error*valid).sum(2), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    shape = count.shape
    meta = dict(layout=np.broadcast_to(data['maze'][:, None, None], shape),
                kind=np.broadcast_to(data['kind'][:, None, None], shape),
                waiting=np.broadcast_to(data['waiting'][:, None], shape))
    return {key: dict(value=value[mask], **{k:v[mask] for k,v in meta.items()})
            for key,value,mask in [('ade',ade,count > 0), ('fde',error[:, :, -1],valid[:, :, -1])]}


def response_rows(data, prediction, endpoint):
    a, b = PAIRS.T
    valid = data['valid'][:, a, endpoint] & data['valid'][:, b, endpoint]
    truth = data['target'][:, b, endpoint].astype(float)-data['target'][:, a, endpoint].astype(float)
    pred = prediction[:, b, endpoint].astype(float)-prediction[:, a, endpoint].astype(float)
    same = np.all(data['future_commands'][:, a] == data['future_commands'][:, b], axis=(2, 3))
    shape = valid.shape
    meta = dict(layout=np.broadcast_to(data['maze'][:, None, None], shape),
                kind=np.broadcast_to(data['kind'][:, None, None], shape),
                waiting=np.broadcast_to(data['waiting'][:, None], shape),
                same_command=np.broadcast_to(same[:, :, None], shape))
    true, predicted = truth[valid], pred[valid]
    return dict(true=true, predicted=predicted, error=np.linalg.norm(predicted-true, axis=1),
                true_norm=np.linalg.norm(true, axis=1), pred_norm=np.linalg.norm(predicted, axis=1),
                **{k:v[valid] for k,v in meta.items()})


def response_summary(row, selected, bootstrap):
    true, pred = row['true'][selected], row['predicted'][selected]
    tn, pn, group = row['true_norm'][selected], row['pred_norm'][selected], row['layout'][selected]
    err, same = row['error'][selected], row['same_command'][selected]
    responsive, nonresponsive, zero = tn > .05, tn <= .05, tn <= 1e-6
    direction = responsive & (pn > 1e-8)
    cosine = np.clip((true[direction]*pred[direction]).sum(1)/(tn[direction]*pn[direction]), -1, 1)
    vector = bootstrap.regression(true, pred, group)
    return dict(error=bootstrap.describe(err, group),
                responsive_error=bootstrap.describe(err[responsive], group[responsive]),
                nonresponsive_error=bootstrap.describe(err[nonresponsive], group[nonresponsive]),
                exact_zero_error=bootstrap.describe(err[zero], group[zero]),
                same_command_error=bootstrap.describe(err[same], group[same]),
                false_response_rate=bootstrap.describe((pn[nonresponsive] > .05).astype(float), group[nonresponsive]),
                exact_zero_false_response_rate=bootstrap.describe((pn[zero] > .05).astype(float), group[zero]),
                zero_response_error=bootstrap.describe(tn, group),
                error_minus_zero_response=bootstrap.describe(err-tn, group),
                magnitude_mae=bootstrap.describe(np.abs(pn-tn), group),
                true_magnitude=bootstrap.describe(tn, group), predicted_magnitude=bootstrap.describe(pn, group),
                direction=dict(count=int(direction.sum()), undefined_prediction_count=int((responsive & ~direction).sum()),
                               mean_cosine=float(cosine.mean()) if len(cosine) else None,
                               mean_angle_deg=float(np.degrees(np.arccos(cosine)).mean()) if len(cosine) else None),
                magnitude_slope=bootstrap.regression(tn[:, None], pn[:, None], group), vector_slope=vector,
                vector_slope_lower_ci_above_0_5_on_F004_population=bool(vector['ci95'] is not None and vector['ci95'][0] > .5))


def compare_rows(candidate, reference, bootstrap, response=False):
    for k in ('layout', 'kind', 'waiting'):
        need(np.array_equal(candidate[k], reference[k]), 'Mismatched paired populations')
    output = {}
    for name in STRATA:
        selected = strata(candidate, name)
        group = candidate['layout']
        if not response:
            output[name] = bootstrap.describe((candidate['value']-reference['value'])[selected], group[selected])
            continue
        need(np.array_equal(candidate['true'], reference['true']), 'Mismatched paired physical truth')
        err = candidate['error']-reference['error']
        true = candidate['true_norm']
        fpr = (candidate['pred_norm'] > .05).astype(float)-(reference['pred_norm'] > .05).astype(float)
        output[name] = {key: bootstrap.describe(value[mask & selected], group[mask & selected])
                        for key,value,mask in [('error',err,np.ones(len(err),bool)),
                                               ('responsive_error',err,true > .05),
                                               ('false_response_rate',fpr,true <= .05)]}
    return output


def coverage(data):
    matched = data['walker'] >= 0
    out = dict(scenes=len(matched), episodes=len(np.unique(data['episode'])), layouts=len(np.unique(data['maze'])),
               initial_matched=int(matched.sum()), initial_visible=int(data['visible_count'].sum()),
               initial_seen_tracks=int(data['seen_count'].sum()), horizons={})
    for h,endpoint in HORIZONS.items():
        safe = (data['contact'] < 0) | (data['contact'] > (endpoint+1)*4)
        base = np.broadcast_to(matched[:, None], data['valid'][:, :, endpoint].shape)
        a,b = PAIRS.T
        pairs = base[:, a] & base[:, b]
        valid_pairs = data['valid'][:, a, endpoint] & data['valid'][:, b, endpoint]
        out['horizons'][str(h)] = dict(initial_mapped_futures=int(base.sum()),
            valid_endpoint_targets=int(data['valid'][:, :, endpoint].sum()),
            contact_censored_targets=int((base & ~safe[..., None]).sum()),
            missing_truth_without_contact=int((base & safe[..., None] & ~data['valid'][:, :, endpoint]).sum()),
            initial_mapped_pairs=int(pairs.sum()), valid_response_pairs=int(valid_pairs.sum()))
    return out


def engineering_bars(models, comparisons):
    checks = {}
    for h in ('3.2','6.4'):
        for metric in ('error','responsive_error'):
            candidate = models['paired'][h]['response']['all'][metric]['mean']
            baseline = models['branch'][h]['response']['all'][metric]['mean']
            ci = comparisons['paired_minus_branch'][h]['response']['all'][metric]['ci95']
            checks[f'{h}/{metric}_20pct_and_ci'] = bool(candidate is not None and baseline is not None
                and candidate <= .8*baseline and ci is not None and ci[1] < 0)
        diff = comparisons['paired_minus_branch'][h]['response']['all']['false_response_rate']['mean']
        checks[f'{h}/false_response_nonincrease'] = bool(diff is not None and diff <= 0)
    for h in ('1.6','3.2','6.4'):
        for metric in ('ade','fde'):
            candidate = models['paired'][h]['trajectory']['factual'][metric]['all']['mean']
            baseline = models['factual'][h]['trajectory']['factual'][metric]['all']['mean']
            checks[f'{h}/factual_{metric}_within5pct'] = bool(candidate is not None and baseline is not None and candidate <= 1.05*baseline)
    return dict(passed=all(checks.values()), checks=checks,
                meaning='Preregistered engineering checks only; exact-result literature review determines interpretation.')


def evaluate(data, learned):
    """Score all final models and deterministic baselines on exactly the same labels."""
    validate_data(data)
    need(set(learned) == {'factual','branch','paired','m5'}, 'All four learned model outputs are required')
    predictions = dict(learned, **baselines(data))
    for value in predictions.values():
        validate_prediction(value, data)
    bootstrap = LayoutBootstrap(data['maze'])
    models = {k:{} for k in predictions}
    comparisons = {f'{a}_minus_{b}':{} for a,b in [('branch','factual'),('paired','branch'),('paired','factual')]}
    for horizon,endpoint in HORIZONS.items():
        h = str(horizon)
        responses = {m:response_rows(data,p,endpoint) for m,p in predictions.items()}
        trajectories = {m:{mode:trajectory_rows(data,p,endpoint,mode) for mode in MODES} for m,p in predictions.items()}
        for model in predictions:
            row = responses[model]
            models[model][h] = dict(response={s:response_summary(row,strata(row,s),bootstrap) for s in STRATA},
                trajectory={mode:{metric:{s:bootstrap.describe(r['value'][strata(r,s)],r['layout'][strata(r,s)])
                                          for s in STRATA} for metric,r in metrics.items()}
                            for mode,metrics in trajectories[model].items()})
        for pair in comparisons:
            a,b = pair.split('_minus_')
            comparisons[pair][h] = dict(response=compare_rows(responses[a],responses[b],bootstrap,True),
                trajectory={mode:{metric:compare_rows(trajectories[a][mode][metric],trajectories[b][mode][metric],bootstrap)
                                  for metric in ('ade','fde')} for mode in MODES})
    beats = {}
    for model in learned:
        beats[model] = {}
        for h in map(str,HORIZONS):
            beats[model][h] = {}
            for mode in MODES:
                beats[model][h][mode] = {}
                for metric in ('ade','fde'):
                    values = {m:models[m][h]['trajectory'][mode][metric]['all']['mean'] for m in (model,'cv','persistence')}
                    beats[model][h][mode][metric] = bool(all(v is not None for v in values.values())
                        and values[model] < min(values['cv'],values['persistence']))
    return dict(coverage=coverage(data),models=models,comparisons=comparisons,
                engineering_bars=engineering_bars(models,comparisons),beats_both_motion_baselines=beats,
                uncertainty=dict(unit='layout',draws=2000,seed=2026111604,interval='percentile95',
                                 minimum_valid_draws=1900,minimum_contributing_layouts=2,
                                 limits='Reused validation; layout IDs are not proof of independent topology. Correlated sibling rows remain together. No multiplicity-adjusted discovery claim.'))
