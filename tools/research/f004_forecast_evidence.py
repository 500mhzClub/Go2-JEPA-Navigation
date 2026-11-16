"""Exact evidence boundaries for F004 numerical, prediction and scoring stages.

No executable launch path: GPU work additionally requires the qualified serial
supervisor and a fresh preconsultation bound to its actual admission inventory.
"""
import json
from pathlib import Path

import torch

from f004_runtime import ARMS, BASE, ROOT, need, sources
from go2nav.data.paired_forecasts import Cache
from go2nav.eval.paired_forecast_inference import InputView, MODELS, NUMERIC_POLICY, read_artifact
from go2nav.models.object_jepa import ObjectConfig
from go2nav.research import digest, require_record, require_review, verify_files

INFERENCE_SOURCES = ('go2nav/eval/paired_forecast_inference.py', 'tools/research/f004_forecast_evidence.py',
                     'tools/research/f004_inference_checks.py')


def implementation_sources():
    return dict(sources(), **{p: digest(ROOT/p) for p in INFERENCE_SOURCES})


def read(path):
    return json.loads(Path(path).read_text())


def bind(bound, path):
    path = Path(path)
    relative = str(path.relative_to(ROOT))
    value = digest(path)
    need(relative not in bound or bound[relative] == value, 'Evidence changed during inventory')
    bound[relative] = value


def reviewed(bound, review_path, expected_result):
    review_path, expected_result = Path(review_path), Path(expected_result)
    review = require_review(review_path)
    need(review['status'] == 'passed', f'Unqualified result review: {review_path}')
    need((review_path.parent/review['result']).resolve() == expected_result.resolve(), 'Review names a different result')
    for path in (review_path, expected_result):
        bind(bound, path)
    return review, read(expected_result)


def common_evidence():
    record, rh = require_record('F004')
    bound = {}
    for path in (ROOT/'docs/research/records/F004.json', BASE/'evaluation_preimplementation.json',
                 BASE/'inference_preimplementation.json', BASE/'additional_inputs.json'):
        bind(bound, path)
    for phase in ('software', 'data', 'metrics_software', 'inspection_software', 'inference_software'):
        review, result = reviewed(bound, BASE/phase/'review.json', BASE/phase/'result.json')
        need(result['record_sha256'] == rh, f'Stale {phase} protocol')
        verify_files(ROOT, result['source_sha256'])
        if phase == 'data':
            verify_files(BASE/'data', result['files'])
            bind(bound, BASE/'data/verification.json')
            need(digest(BASE/'data/verification.json') == review['measurements']['verification_sha256'], 'Changed independent data verification')
        if phase == 'inference_software':
            need(result['preimplementation_sha256'] == digest(BASE/'inference_preimplementation.json'), 'Changed inference consultation')
    need(digest(BASE/'additional_inputs.json') == record['protocol']['inputs']['additional_manifest_sha256'], 'Changed additional inputs')
    additional = read(BASE/'additional_inputs.json')
    verify_files(ROOT/'outputs/data/ring_corpus_v1', additional['rasters'])
    return record, rh, bound, additional


def selected_checkpoint(name, bound, record, rh, additional):
    need(name in MODELS, 'Unknown model')
    if name == 'm5':
        for field in ('m5_model', 'm5_config'):
            row = additional[field]
            need(digest(ROOT/row['path']) == row['sha256'], f'Changed frozen {field}')
            bind(bound, ROOT/row['path'])
        row = additional['m5_model']
        checkpoint = torch.load(ROOT/row['path'], map_location='cpu', weights_only=True)
        config = ObjectConfig(**read(ROOT/additional['m5_config']['path'])['model']).to_dict()
        need(checkpoint['config'] == config, 'M5 architecture differs from frozen configuration')
        return dict(name=name, path=row['path'], sha256=row['sha256'], config=config, step=checkpoint['step'],
                    selection='frozen_M5_reference_not_a_new_F004_training_arm')
    folder = BASE/'runs'/f'{name}_s0'
    identity = dict(arm=name, seed=0)
    review, training = reviewed(bound, folder/'review.json', folder/'science/result.json')
    opreview, operations = reviewed(bound, folder/'operations/review.json', folder/'operations/result.json')
    inspection_review, inspection = reviewed(bound, folder/'inspection_review.json', folder/'inspection.json')
    for r in (review, opreview, inspection_review, training, operations, inspection):
        need(r['run_identity'] == identity, 'Wrong trained-arm identity')
    for r in (training, operations, inspection):
        need(r['record_sha256'] == rh, 'Wrong trained-arm protocol')
    need(training['final_step'] == 20000 and training['selection'] == 'final_step_only', 'Nonfinal checkpoint selected')
    need(training['binding']['model'] == record['protocol']['model']
         and training['binding']['settings'] == record['protocol']['training'], 'Training recipe changed')
    need(training['binding']['arm'] == name and training['binding']['seed'] == 0
         and training['binding']['record_sha256'] == rh, 'Wrong training binding')
    need(inspection['science_result_sha256'] == digest(folder/'science/result.json')
         and inspection['operations_result_sha256'] == digest(folder/'operations/result.json'), 'Stale training inspection')
    verify_files(ROOT, inspection['source_sha256'])
    path = folder/'science/model.pt'
    need(digest(path) == training['checkpoint_sha256'] == inspection['artifact_sha256']['model.pt'], 'Changed selected checkpoint')
    bind(bound, path)
    return dict(name=name, path=str(path.relative_to(ROOT)), sha256=digest(path),
                config=record['protocol']['model'], step=20000, selection='final_step_only',
                training_binding=training['binding'])


def base_binding(name):
    record, rh, bound, additional = common_evidence()
    model = selected_checkpoint(name, bound, record, rh, additional)
    binding = dict(record_sha256=rh, model=model, source_sha256=implementation_sources(),
                   data_result_sha256=digest(BASE/'data/result.json'),
                   additional_inputs_sha256=digest(BASE/'additional_inputs.json'), numerical_policy=NUMERIC_POLICY)
    return binding, bound


def require_operations(bound, folder, name, phase):
    review, result = reviewed(bound, folder/'operations/review.json', folder/'operations/result.json')
    identity = dict(model=name, phase=phase)
    need(review['run_identity'] == result['run_identity'] == identity, 'Wrong evaluation operation identity')
    need(result['returncode'] == 0 and result['child_terminal'] and result['inputs_unchanged']
         and not result['stalled'] and not result['actions'] and not result['errors'], 'Unsuccessful evaluation operations')
    need(result['child_artifact_sha256']['result.json'] == digest(folder/'science/result.json'), 'Stale operations artifact inventory')


def prediction_binding(name):
    binding, bound = base_binding(name)
    folder = BASE/'evaluation/numerical'/name
    review, result = reviewed(bound, folder/'review.json', folder/'science/result.json')
    need(result['binding'] == binding, 'Numerical review belongs to different source/data/checkpoint')
    keys = InputView(Cache(BASE/'data', 'val').arrays).keys(NUMERIC_POLICY['scenes'])
    verified, _ = read_artifact(folder/'science', binding, keys, 'numerical')
    need(verified['numerical']['passed'] and verified['scenes'] == NUMERIC_POLICY['scenes'], 'Actual selected-checkpoint numerical check failed')
    require_operations(bound, folder, name, 'numerical')
    for filename in result['files']:
        bind(bound, folder/'science'/filename)
    return dict(binding, numerical_result_sha256=digest(folder/'science/result.json'),
                numerical_review_sha256=digest(folder/'review.json'), prediction_policy='GPU_float32_math_batch32'), bound


def scoring_inputs():
    """All four reviewed learned outputs on the exact full validation order."""
    cache = Cache(BASE/'data', 'val')
    keys = InputView(cache.arrays).keys()
    predictions, bound = {}, {}
    for name in MODELS:
        binding, inputs = prediction_binding(name)
        bound.update(inputs)
        folder = BASE/'evaluation/prediction'/name
        _, result = reviewed(bound, folder/'review.json', folder/'science/result.json')
        _, arrays = read_artifact(folder/'science', binding, keys, 'prediction')
        require_operations(bound, folder, name, 'prediction')
        for filename in result['files']:
            bind(bound, folder/'science'/filename)
        predictions[name] = arrays['world']
    return cache.arrays, predictions, bound
