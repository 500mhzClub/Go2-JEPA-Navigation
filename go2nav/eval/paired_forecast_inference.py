"""F004 observable-only inference, fixed numerical comparisons and ordered artifacts.

These primitives launch no jobs and approve no evidence. Scientific callers must
use the separately qualified supervisor and exact-checkpoint literature reviews.
"""
from contextlib import contextmanager
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from go2nav.data.paired_forecasts import INPUTS, SCHEMA, need
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, verify_files
from go2nav.training.object_jepa import rotate
from go2nav.training.paired_forecasts import features, forecast

MODELS = ('factual', 'branch', 'paired', 'm5')
FLAVORS = ('cpu1', 'cpu32', 'gpu1', 'gpu32')
FRAMES = ('body', 'world')
NUMERIC_POLICY = dict(scenes=64, futures=7, steps=16, slots=6, dtype='float32', backend='math',
                      batches=[1, 32], atol=1e-4, rtol=1e-4,
                      population='first64_retained_val_scenes_all7_futures',
                      comparison='all6_flavor_pairs_in_body_and_world_every_coordinate')


class InputView:
    """Keep only observed/candidate inputs and case keys; no labels enter inference."""
    def __init__(self, arrays):
        self.n = len(arrays['episode'])
        need(self.n > 0, 'Empty input population')
        self.arrays = {k: arrays[k] for k in (*INPUTS, 'episode', 't0')}
        for key, array in self.arrays.items():
            shape, dtype = SCHEMA[key]
            need(array.shape == (self.n, *shape) and array.dtype == np.dtype(dtype), f'Input shape/dtype: {key}')
        scenes = np.stack((self.arrays['episode'], self.arrays['t0']), 1)
        need(len(np.unique(scenes, axis=0)) == self.n, 'Duplicate scene identity')

    def take(self, points, futures):
        need(points.ndim == futures.ndim == 1 and len(points) == len(futures), 'Unaligned input selection')
        need(np.issubdtype(points.dtype, np.integer) and np.issubdtype(futures.dtype, np.integer), 'Noninteger selection')
        need(np.all((points >= 0) & (points < self.n)) and np.all((futures >= 0) & (futures < 7)), 'Input selection range')
        return {k: self.arrays[k][points, futures] if k in ('future_commands', 'future_pose')
                else self.arrays[k][points] for k in INPUTS}

    def keys(self, scenes=None):
        n = self.n if scenes is None else scenes
        need(isinstance(n, int) and 0 < n <= self.n, 'Invalid scene count')
        return np.stack((np.repeat(self.arrays['episode'][:n], 7),
                         np.repeat(self.arrays['t0'][:n], 7), np.tile(np.arange(7), n)), 1).astype(np.int64)


def validate_keys(keys, expected=None):
    need(keys.ndim == 2 and keys.shape[1] == 3 and keys.dtype == np.int64
         and len(keys) > 0 and len(keys) % 7 == 0, 'Wrong case key shape/type/count')
    grouped = keys.reshape(-1, 7, 3)
    need(np.all(grouped[:, :, :2] == grouped[:, :1, :2])
         and np.all(grouped[:, :, 2] == np.arange(7)), 'Reordered or missing future keys')
    need(len(np.unique(keys, axis=0)) == len(keys), 'Duplicate case keys')
    if expected is not None:
        need(np.array_equal(keys, expected), 'Case keys differ from fixed population/order')


def validate_array(value, keys):
    need(value.shape == (len(keys)//7, 7, 16, 6, 2) and value.dtype == np.float32, 'Wrong forecast shape/dtype')
    need(np.isfinite(value).all(), 'Nonfinite forecast, including unmatched slots')


def load_model(path, expected, device):
    """Load only a bound local checkpoint, with strict state shape/dtype checks."""
    need(expected['name'] in MODELS and digest(path) == expected['sha256'], 'Wrong model/checkpoint digest')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    need(checkpoint['config'] == expected['config'] and checkpoint['step'] == expected['step'], 'Wrong checkpoint config/step')
    if expected['name'] != 'm5':
        need(expected['step'] == 20000 and checkpoint['binding'] == expected['training_binding'], 'Not the bound final training checkpoint')
    model = ObjectJEPA(ObjectConfig(**expected['config']))
    initial = model.state_dict()
    need(checkpoint['model'].keys() == initial.keys(), 'Wrong model state keys')
    for name, value in checkpoint['model'].items():
        need(value.shape == initial[name].shape and value.dtype == initial[name].dtype
             and bool(torch.isfinite(value).all()), f'Model tensor shape/dtype/finiteness: {name}')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model.to(device)


@contextmanager
def inference_policy():
    """Same explicit float32 math policy as training; restore caller backend flags."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def predict(model, view, rasters, batch, *, scenes=None, progress=None):
    """All seven futures per selected scene; body and world coordinates, unshuffled slots."""
    need(batch in (1, 32), 'Unqualified inference batch')
    need(all(p.dtype == torch.float32 for p in model.parameters()), 'Float32 model required')
    keys = view.keys(scenes)
    validate_keys(keys)
    count = len(keys)
    n = count//7
    points, futures = np.repeat(np.arange(n), 7), np.tile(np.arange(7), n)
    output = {frame: np.empty((count, 16, 6, 2), dtype=np.float32) for frame in FRAMES}
    device = next(model.parameters()).device
    model.eval()
    with inference_policy():
        for start in range(0, count, batch):
            end = min(count, start+batch)
            raw = view.take(points[start:end], futures[start:end])
            inputs = features(raw, rasters, device)
            body = forecast(model, inputs, rasters, 16)
            world = rotate(body, inputs['heading'][:, None, None])+inputs['origin'][:, None, None]
            need(body.dtype == world.dtype == torch.float32 and body.shape == world.shape == (end-start, 16, 6, 2),
                 'Inference changed shape/dtype')
            for frame, value in (('body', body), ('world', world)):
                array = value.cpu().numpy()
                need(np.isfinite(array).all(), f'Nonfinite {frame} forecast')
                output[frame][start:end] = array
            if progress:
                progress(end, count)
    return {k: v.reshape(n, 7, 16, 6, 2) for k, v in output.items()}, keys


def numerical_comparison(arrays, keys):
    """All six comparisons in both frames; no configurable tolerance or endpoint mask."""
    validate_keys(keys)
    names = {f'{flavor}_{frame}' for flavor in FLAVORS for frame in FRAMES}
    need(set(arrays) == names, 'Incomplete numerical flavors/frames')
    for value in arrays.values():
        validate_array(value, keys)
    comparisons = {}
    for first, second in itertools.combinations(FLAVORS, 2):
        for frame in FRAMES:
            reference, candidate = arrays[f'{first}_{frame}'].astype(np.float64), arrays[f'{second}_{frame}'].astype(np.float64)
            difference = np.abs(candidate-reference)
            limit = NUMERIC_POLICY['atol']+NUMERIC_POLICY['rtol']*np.abs(reference)
            bad = difference > limit
            by_step = bad.sum(axis=(0, 1, 3, 4)).tolist()
            first_bad = np.argwhere(bad)[0].tolist() if bad.any() else None
            comparisons[f'{second}_vs_{first}/{frame}'] = dict(
                passed=not bool(bad.any()), coordinates=bad.size, failed_coordinates=int(bad.sum()),
                failed_by_step=by_step, first_failed_index=first_bad,
                max_absolute_error=float(difference.max()), max_tolerance_ratio=float((difference/limit).max()))
    return dict(passed=all(r['passed'] for r in comparisons.values()), comparisons=comparisons,
                policy=NUMERIC_POLICY, scenes=len(keys)//7,
                limits='Fixed sample only; software or numerical agreement is not forecasting accuracy.')


def artifact_names(phase):
    need(phase in ('numerical', 'prediction'), 'Unknown forecast artifact phase')
    return {f'{flavor}_{frame}' for flavor in FLAVORS for frame in FRAMES} if phase == 'numerical' else {'world'}


def write_artifact(folder, arrays, keys, binding, phase):
    """Fresh immutable array artifact; no passing literature review is synthesized."""
    need(set(arrays) == artifact_names(phase), 'Wrong artifact arrays')
    validate_keys(keys)
    for value in arrays.values():
        validate_array(value, keys)
    measurement = numerical_comparison(arrays, keys) if phase == 'numerical' else None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in dict(arrays, keys=keys).items():
        path = folder/(name+'.npy')
        with path.open('xb') as file:
            np.save(file, value, allow_pickle=False)
        files[path.name] = digest(path)
    result = dict(research_id='F004', schema='physical_forecasts/1', phase=phase, binding=binding,
                  status='awaiting_literature_review' if measurement is None or measurement['passed'] else 'technical_blocked',
                  scenes=len(keys)//7, keys_order='scene_major_future_minor_0_to_6', files=files,
                  numerical=measurement, next_step='Inspect exact arrays and consult literature before interpreting or admitting dependent work.')
    atomic_json(folder/'result.json', result)
    return result


def read_artifact(folder, expected_binding, expected_keys, phase):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text())
    need(result['research_id'] == 'F004' and result['schema'] == 'physical_forecasts/1'
         and result['phase'] == phase and result['binding'] == expected_binding, 'Stale/wrong forecast provenance')
    names = artifact_names(phase)
    need(set(result['files']) == {k+'.npy' for k in names | {'keys'}}, 'Wrong artifact file inventory')
    verify_files(folder, result['files'])
    keys = np.load(folder/'keys.npy', allow_pickle=False)
    validate_keys(keys, expected_keys)
    need(result['scenes'] == len(keys)//7 and result['keys_order'] == 'scene_major_future_minor_0_to_6', 'Artifact population/order mismatch')
    arrays = {name: np.load(folder/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in names}
    for value in arrays.values():
        validate_array(value, keys)
    measurement = numerical_comparison(arrays, keys) if phase == 'numerical' else None
    need(result['numerical'] == measurement, 'Numerical summary differs from saved arrays')
    status = 'awaiting_literature_review' if measurement is None or measurement['passed'] else 'technical_blocked'
    need(result['status'] == status, 'Artifact status differs from measurements')
    return result, arrays
