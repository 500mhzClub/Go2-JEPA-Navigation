"""Explicit C014 full-float64 inference and checkpoint-specific admission.

Training and cached features retain their original precision. Numerical validators
use ``load_full64``; scientific evaluators must use the reviewed loaders.
"""
import json
from pathlib import Path

import torch

from go2nav.research import REPO, digest, require_record, require_review, verify_files
from go2nav.training.balls_slots import load_encoder, load_world

POLICY = 'encoder64_predictor64'
IMPLEMENTATION_SOURCES = (
    'go2nav/eval/slots_inference.py',
    'go2nav/eval/balls_binding.py',
    'go2nav/eval/balls_probes.py',
    'go2nav/eval/balls_planning.py',
    'go2nav/models/balls_slots.py',
    'go2nav/training/balls_slots.py',
    'go2nav/training/balls.py',
)


def _identity(state):
    kind, n, seed = state['kind'], state['n'], state['seed']
    allowed = (n == 3 and seed in (0, 1, 2) and kind in ('slots', 'o0', 'o1')) \
        or (n in (1, 6) and seed == 0 and kind in ('slots', 'o1'))
    if not allowed:
        raise ValueError('Checkpoint is outside the approved object-model cases')
    return dict(kind=kind, n=n, seed=seed, name=f'{kind}_n{n}_s{seed}')


def load_full64(checkpoint, device='cpu'):
    """Fresh strict load for numerical validation; supplies no scientific admission."""
    before = digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = _identity(state)
    if identity['kind'] == 'slots':
        model, _ = load_encoder(checkpoint, 'cpu')
    else:
        model = load_world(checkpoint, 'cpu')
    model.double().eval()
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point() and (tensor.dtype != torch.float64 or not torch.isfinite(tensor).all()):
            raise ValueError('Inference requires finite float64 parameters and buffers')
    if digest(checkpoint) != before:
        raise ValueError('Checkpoint changed during inference loading')
    return model.to(device), state


def require_numerical_review(checkpoint, review_path):
    """Bind a complete C014 review to exact weights, identity, policy and code."""
    _, record_hash = require_record('C014')
    review_path = Path(review_path)
    review = require_review(review_path)
    result = json.loads((review_path.parent/review['result']).read_text())
    if review['research_id'] != 'C014' or review['status'] != 'passed' \
            or result.get('research_id') != 'C014' or result.get('research_sha256') != record_hash:
        raise ValueError('A passing C014 numerical review is required')
    if result.get('selected_policy') != POLICY:
        raise ValueError('Numerical review does not qualify the deployed full64 policy')
    if result.get('status') != 'awaiting_literature_review' \
            or any(result.get(k) is not True for k in ('complete', 'checks_passed', 'weights_unchanged', 'inputs_unchanged')):
        raise ValueError('Numerical evidence is incomplete, failed or mutated')
    if result.get('checkpoint_sha256') != digest(checkpoint):
        raise ValueError('Numerical review belongs to a different checkpoint')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = _identity(state)
    if any(result.get(k) != v for k, v in identity.items()):
        raise ValueError('Numerical review checkpoint identity differs')
    expected = {'slots': 1104, 'o0': 2772, 'o1': 4164}[identity['kind']]
    if result.get('comparison_count') != expected:
        raise ValueError('Numerical review comparison coverage differs')
    sources = result.get('implementation_sha256', {})
    if set(sources) != set(IMPLEMENTATION_SOURCES):
        raise ValueError('Numerical review lacks the complete inference implementation')
    verify_files(REPO, sources)
    inputs = result.get('inputs_sha256', {})
    resolved = {str((REPO/Path(p)).resolve()): value for p, value in inputs.items()}
    if resolved.get(str(Path(checkpoint).resolve())) != result['checkpoint_sha256']:
        raise ValueError('Numerical input inventory omits the selected checkpoint')
    verify_files(REPO, inputs)
    return result


def load_reviewed_world(checkpoint, review_path, device='cuda:0'):
    evidence = require_numerical_review(checkpoint, review_path)
    if evidence['kind'] not in ('o0', 'o1'):
        raise ValueError('World inference needs a reviewed predictor')
    model, state = load_full64(checkpoint, device)
    if digest(checkpoint) != evidence['checkpoint_sha256'] or _identity(state)['name'] != evidence['name']:
        raise ValueError('Checkpoint changed after numerical admission')
    return model


def load_reviewed_encoder(checkpoint, review_path, device='cuda:0'):
    evidence = require_numerical_review(checkpoint, review_path)
    if evidence['kind'] != 'slots':
        raise ValueError('Binding needs a reviewed encoder')
    model, state = load_full64(checkpoint, device)
    if digest(checkpoint) != evidence['checkpoint_sha256'] or _identity(state)['name'] != evidence['name']:
        raise ValueError('Checkpoint changed after numerical admission')
    return model, state
