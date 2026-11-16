"""F004 physical-evaluator fixtures and independent bootstrap arithmetic."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from f004_software import fixture
from go2nav.data.paired_forecasts import extract
from go2nav.eval.paired_forecast_metrics import (LayoutBootstrap, baselines, compare_rows,
    engineering_bars, evaluate, response_rows, response_summary, trajectory_rows, validate_data, validate_prediction)
from go2nav.research import atomic_json,digest,require_record


def data_fixture():
    one,_ = extract(*fixture())
    data = {k:np.repeat(v,4,axis=0) for k,v in one.items()}
    data['episode'][:] = np.arange(4)
    data['maze'][:] = [1200,1200,1201,1201]
    data['kind'][:] = [1,2,3,3]
    data['waiting'][1::2,0] = True
    return data


def bar_fixture():
    models = {m:{} for m in ('factual','branch','paired')}
    differences = {'paired_minus_branch':{}}
    for h in ('1.6','3.2','6.4'):
        for m in models:
            models[m][h] = dict(response={'all':{k:dict(mean=.8 if m == 'paired' else 1.)
                                                for k in ('error','responsive_error')}},
                trajectory={'factual':{k:{'all':dict(mean=1.05 if m == 'paired' else 1.)} for k in ('ade','fde')}})
        differences['paired_minus_branch'][h] = dict(response={'all':dict(error=dict(ci95=[-.3,-.1]),
            responsive_error=dict(ci95=[-.3,-.1]),false_response_rate=dict(mean=0.))})
    return models,differences


class Checks(unittest.TestCase):
    def test_unbalanced_layout_means_match_explicit_resampled_rows(self):
        groups = np.array([10,10,10,10,20,20,30])
        values = np.array([1.,2.,3.,7.,10.,14.,-1.])
        bootstrap = LayoutBootstrap(groups)
        actual = bootstrap.mean_draws(values,groups)
        expected = []
        for frequency in bootstrap.frequency:
            rows = np.concatenate([np.tile(np.flatnonzero(groups==u),n) for u,n in zip(bootstrap.layouts,frequency)])
            expected.append(values[rows].mean())
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        # Every sibling in a layout shares one draw weight, including unequal cluster sizes.
        self.assertTrue(np.all(bootstrap.frequency.sum(1)==3))
        np.testing.assert_array_equal(bootstrap.frequency,LayoutBootstrap(groups).frequency)
        self.assertEqual(bootstrap.describe(values-values,groups)['ci95'],[0.,0.])
        self.assertIsNone(bootstrap.describe(values[:4],groups[:4])['ci95'])
        self.assertIsNone(bootstrap.describe(np.array([]),np.array([],int))['ci95'])

    def test_vector_and_magnitude_slopes_and_bootstrap(self):
        groups = np.array([10,10,10,20,20,30,30])
        truth = np.array([[1.,0.],[2.,1.],[3.,-1.],[1.,1.],[4.,-2.],[2.,2.],[-1.,3.]])
        predicted = -.7*truth+np.array([.2,-.4])
        bootstrap = LayoutBootstrap(groups)
        result = bootstrap.regression(truth,predicted,groups)
        self.assertAlmostEqual(result['slope'],-.7)
        np.testing.assert_allclose(result['intercept'],[.2,-.4],atol=1e-12)
        slopes = []
        for frequency in bootstrap.frequency:
            rows = np.concatenate([np.tile(np.flatnonzero(groups==u),n) for u,n in zip(bootstrap.layouts,frequency)])
            x,y = truth[rows],predicted[rows]
            xc,yc = x-x.mean(0),y-y.mean(0)
            slopes.append(float((xc*yc).sum()/(xc*xc).sum()))
        np.testing.assert_allclose(result['ci95'],np.quantile(slopes,[.025,.975]),atol=1e-12)
        tn = np.linalg.norm(truth,axis=1)[:,None]
        self.assertAlmostEqual(bootstrap.regression(tn,tn,groups)['slope'],1.)
        self.assertAlmostEqual(bootstrap.regression(truth,-truth,groups)['slope'],-1.)
        degenerate = bootstrap.regression(np.zeros_like(truth),predicted,groups)
        self.assertIsNone(degenerate['slope']);self.assertIsNone(degenerate['ci95'])

    def test_contact_prefix_ade_endpoint_masks_and_motion_baselines(self):
        data = data_fixture()
        prediction = data['target'].copy()
        prediction[...,0] += np.arange(1,17)[None,None,:,None]
        rows = trajectory_rows(data,prediction,7,'branches')
        self.assertEqual(len(rows['ade']['value']),24)
        self.assertEqual(len(rows['fde']['value']),20)
        self.assertAlmostEqual(rows['ade']['value'][2],2.)  # Contact16 leaves only steps4/8/12.
        self.assertTrue((rows['fde']['value']==8).all())
        data['history_velocity'][:,-1,0] = [2.,-1.]
        predictions = baselines(data)
        expected = data['history_xy'][0,-1,0]+6.4*np.array([2.,-1.])
        np.testing.assert_allclose(predictions['cv'][0,6,-1,0],expected)
        for name,p in predictions.items():
            np.testing.assert_array_equal(p[:,0],p[:,6])
        np.testing.assert_array_equal(predictions['persistence'][:,6,-1,0],data['history_xy'][:,-1,0])

    def test_perfect_opposite_zero_and_invented_responses(self):
        data = data_fixture()
        bootstrap = LayoutBootstrap(data['maze'])
        row = response_rows(data,data['target'],15)
        result = response_summary(row,np.ones(len(row['layout']),bool),bootstrap)
        self.assertEqual(result['error']['mean'],0)
        self.assertAlmostEqual(result['direction']['mean_angle_deg'],0)
        opposite = response_rows(data,-data['target'],15)
        result = response_summary(opposite,np.ones(len(opposite['layout']),bool),bootstrap)
        self.assertAlmostEqual(result['direction']['mean_angle_deg'],180)
        self.assertAlmostEqual(result['magnitude_slope']['slope'],1)
        self.assertAlmostEqual(result['vector_slope']['slope'],-1)
        zero = response_rows(data,baselines(data)['persistence'],15)
        result = response_summary(zero,np.ones(len(zero['layout']),bool),bootstrap)
        self.assertEqual(result['direction']['count'],0)
        self.assertGreater(result['direction']['undefined_prediction_count'],0)
        # Equal physical futures, unequal commands, spurious predicted changes.
        equal = copy.deepcopy(data)
        common = np.broadcast_to(equal['target'][:,0:1],equal['target'].shape)
        equal['target'] = np.where(equal['valid'][...,None],common,0).copy()
        invented = equal['target'].copy()
        invented[...,0] += np.arange(7)[None,:,None,None]*.2
        row = response_rows(equal,invented,3)
        result = response_summary(row,np.ones(len(row['layout']),bool),bootstrap)
        self.assertEqual(result['false_response_rate']['mean'],1.)
        self.assertIsNone(result['responsive_error']['mean'])
        self.assertIsNone(result['vector_slope']['slope'])

    def test_engineering_bars_reject_hidden_regressions_and_empty_evidence(self):
        models,comparisons = bar_fixture()
        self.assertTrue(engineering_bars(models,comparisons)['passed'])
        bad = copy.deepcopy(models)
        bad['paired']['3.2']['response']['all']['responsive_error']['mean'] = .81
        self.assertFalse(engineering_bars(bad,comparisons)['passed'])
        bad = copy.deepcopy(comparisons)
        bad['paired_minus_branch']['6.4']['response']['all']['false_response_rate']['mean'] = .001
        self.assertFalse(engineering_bars(models,bad)['passed'])
        bad = copy.deepcopy(comparisons)
        bad['paired_minus_branch']['3.2']['response']['all']['error']['ci95'] = None
        self.assertFalse(engineering_bars(models,bad)['passed'])
        bad = copy.deepcopy(models)
        bad['paired']['1.6']['trajectory']['factual']['ade']['all']['mean'] = 1.050001
        self.assertFalse(engineering_bars(bad,comparisons)['passed'])
        for m in models:
            models[m]['3.2']['response']['all']['error']['mean'] = 0
        comparisons['paired_minus_branch']['3.2']['response']['all']['error']['ci95'] = [0.,0.]
        self.assertFalse(engineering_bars(models,comparisons)['passed'])

    def test_population_and_pairing_faults(self):
        data = data_fixture()
        validate_data(data)
        bad = copy.deepcopy(data);bad['episode'][1] = bad['episode'][0]
        with self.assertRaises(ValueError):validate_data(bad)
        bad = copy.deepcopy(data);bad['valid'][0,0,0,5] = True
        with self.assertRaises(ValueError):validate_data(bad)
        p = data['target'].copy();p[0,0,0,5,0] = np.nan
        with self.assertRaises(ValueError):validate_prediction(p,data)
        rows = trajectory_rows(data,data['target'],3,'factual')['fde']
        other = copy.deepcopy(rows);other['layout'] += 1
        with self.assertRaises(ValueError):compare_rows(rows,other,LayoutBootstrap(data['maze']))

    def test_full_report_exact_perfection_and_no_automatic_benefit_claim(self):
        data = data_fixture()
        prediction = {m:data['target'].copy() for m in ('factual','branch','paired','m5')}
        report = evaluate(data,prediction)
        self.assertEqual(report['coverage']['scenes'],4)
        self.assertEqual(report['coverage']['layouts'],2)
        self.assertEqual(report['models']['paired']['6.4']['trajectory']['factual']['fde']['all']['mean'],0.)
        self.assertEqual(report['comparisons']['paired_minus_branch']['6.4']['response']['all']['error']['ci95'],[0.,0.])
        self.assertFalse(report['engineering_bars']['passed'])
        self.assertIsNone(report['models']['m5']['1.6']['response']['none']['error']['mean'])
        json.dumps(report,allow_nan=False)
        with self.assertRaises(ValueError):evaluate(data,{'factual':data['target']})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    out = Path(args.out);out.mkdir(parents=True,exist_ok=False)
    _,rh = require_record('F004')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    files = ['go2nav/eval/paired_forecast_metrics.py','tools/research/f004_metrics_checks.py',
             'go2nav/data/paired_forecasts.py','tools/research/f004_software.py']
    atomic_json(out/'result.json',dict(research_id='F004',phase='metrics_software',record_sha256=rh,
        status='passed' if result.wasSuccessful() else 'technical_blocked',
        created_at=datetime.now(timezone.utc).isoformat(),checks=result.testsRun,
        source_sha256={p:digest(ROOT/p) for p in files},
        preimplementation_sha256=digest(ROOT/'outputs/gated/F004/evaluation_preimplementation.json'),
        failures=[dict(test=str(t),traceback=e) for t,e in result.errors+result.failures],
        limits='Synthetic metrics and independent scalar bootstrap checks only. Actual inference, artifact gates and trained numerical checks are separate. No scientific forecast result.'))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':main()
