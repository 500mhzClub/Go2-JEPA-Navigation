"""Read-only CPU inspection of a terminal F004 arm; never approves or launches work."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import psutil
import torch

from f004_runtime import ARMS, BASE, ROOT, need
from go2nav.models.object_jepa import ObjectConfig, ObjectJEPA
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

SOURCES = ('tools/research/f004_inspect_training.py', 'tools/research/f004_inspection_checks.py')


def read(path):
    return json.loads(Path(path).read_text())


def expected_binding(admission):
    return {k: admission[k] for k in ('record_sha256', 'arm', 'seed', 'inputs_sha256',
                                     'source_sha256', 'model', 'settings')}


def learning_rate(settings, step):
    """Independent scalar schedule for the last completed update (step0 is initialization)."""
    if step == 0:
        return settings['lr']
    index = step-1
    warmup, total = settings['warmup_steps'], settings['steps']
    multiplier = ((index+1)/warmup if index < warmup else
                  .5*(1+math.cos(math.pi*(index-warmup)/(total-warmup-1))))
    return settings['lr']*multiplier


def replay_sampler(n, settings, steps):
    """Replay only random draws, independently of Sampler.draw and its state loader."""
    rng = np.random.default_rng(settings['seed'])
    states = {0: copy.deepcopy(rng.bit_generator.state)}
    pairs = settings['pairs_per_batch']
    wanted = set(steps)
    for step in range(1, max(wanted, default=0)+1):
        rng.integers(n, size=pairs)
        rng.integers(6, size=pairs)
        rng.integers(5, size=pairs)
        rng.random((pairs, 6))
        if step in wanted:
            states[step] = copy.deepcopy(rng.bit_generator.state)
    return states


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_tree(v, right[k]) for k, v in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(right) is type(left) and len(left) == len(right)
                and all(equal_tree(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def inspect_checkpoint(checkpoint, model, initial, initial_rng, binding, step, n, sampler_rng,
                       *, require_cuda=True):
    need(checkpoint['step'] == step and checkpoint['binding'] == binding, 'Checkpoint step/binding mismatch')
    need(checkpoint['config'] == model.config.to_dict(), 'Checkpoint model config mismatch')
    need(checkpoint['scheduler'] == dict(next_update=step, formula='F004.rate'), 'Checkpoint scheduler mismatch')
    state = checkpoint['model']
    need(state.keys() == initial.keys(), 'Checkpoint model keys mismatch')
    settings = binding['settings']
    calls = settings['pairs_per_batch']//settings['microbatch_pairs']*settings['rollout_steps']
    need(settings['pairs_per_batch'] % settings['microbatch_pairs'] == 0, 'Incomplete microbatch')
    for name, value in state.items():
        need(isinstance(value, torch.Tensor) and value.device.type == 'cpu', f'Non-CPU tensor: {name}')
        need(value.shape == initial[name].shape and value.dtype == initial[name].dtype, f'Tensor shape/dtype: {name}')
        need(bool(torch.isfinite(value).all()), f'Nonfinite model tensor: {name}')
        if name.endswith('num_batches_tracked'):
            need(int(value) == step*calls, f'BatchNorm count: {name}')
        if name.endswith('running_var'):
            need(bool((value >= 0).all()), f'Negative BatchNorm variance: {name}')
        if step == 0 or name == 'mask_token' or name.startswith('mask_proj.'):
            need(torch.equal(value, initial[name]), f'Initialization/unused parameter changed: {name}')
    model.load_state_dict(state, strict=True)
    # Parameter IDs are matched in the actual named_parameters order; no optimizer load can hide missing state.
    expected = torch.optim.AdamW(model.parameters(), lr=learning_rate(settings, step),
                                weight_decay=settings['weight_decay']).state_dict()
    optimizer = checkpoint['optimizer']
    need(equal_tree(optimizer['param_groups'], expected['param_groups']), 'Optimizer groups/schedule mismatch')
    parameters = list(model.named_parameters())
    active = {i for i, (name, _) in enumerate(parameters)
              if name != 'mask_token' and not name.startswith('mask_proj.')}
    need(set(optimizer['state']) == (active if step else set()), 'Missing/extra optimizer state')
    for index, entry in optimizer['state'].items():
        need(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Optimizer state fields mismatch')
        counter = entry['step']
        need(isinstance(counter, torch.Tensor) and counter.ndim == 0 and bool(torch.isfinite(counter))
             and float(counter) == step, 'Optimizer update count mismatch')
        parameter = parameters[index][1]
        for name in ('exp_avg', 'exp_avg_sq'):
            value = entry[name]
            need(value.shape == parameter.shape and value.dtype == parameter.dtype
                 and bool(torch.isfinite(value).all()), 'Optimizer tensor shape/dtype/finiteness')
        need(bool((entry['exp_avg_sq'] >= 0).all()), 'Negative optimizer squared moment')
    need(checkpoint['sampler'] == dict(n=n, arm=binding['arm'], rng=sampler_rng), 'Sampler stream mismatch')
    need(equal_tree(checkpoint['torch_rng'], initial_rng), 'CPU RNG changed from seeded initialization')
    cuda_rng = checkpoint['cuda_rng']
    need(isinstance(cuda_rng, list) and (bool(cuda_rng) if require_cuda else not cuda_rng), 'Wrong device RNG coverage')
    for rng in cuda_rng:
        need(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8 and rng.ndim == 1
             and rng.numel() > 0 and rng.device.type == 'cpu', 'Invalid saved device RNG')
    return dict(step=step, parameters=len(parameters), optimizer_parameters=len(optimizer['state']),
                batchnorm_forwards=step*calls, lr=learning_rate(settings, step))


def inspect_logs(training, validation, settings, val_points):
    steps = settings['steps']
    expected = sorted({1, *range(settings['log_every'], steps+1, settings['log_every'])})
    need([r['step'] for r in training] == expected, 'Missing/duplicate/reordered training steps')
    previous = -1.
    for row in training:
        fields = ('elapsed_seconds', 'loss', 'trajectory_m', 'response_m', 'grad_norm', 'lr',
                  'trajectory_targets', 'response_targets')
        need(all(math.isfinite(row[k]) and row[k] >= 0 for k in fields), 'Invalid training numeric value')
        need(row['elapsed_seconds'] > previous, 'Training elapsed time not increasing')
        previous = row['elapsed_seconds']
        need(row['trajectory_targets'] > 0 and row['response_targets'] <= row['trajectory_targets'], 'Training target counts')
        need(row['lr'] == learning_rate(settings, row['step']), 'Logged learning rate mismatch')
    expected = list(range(settings['validation_every'], steps+1, settings['validation_every']))
    need([r['step'] for r in validation] == expected, 'Missing/duplicate/reordered validation steps')
    coverage = None
    for row in validation:
        need(row['points'] == min(settings['validation_points'], val_points)
             and row['selection'] == 'monitoring_only_final_step_selected', 'Validation population/selection mismatch')
        need(len(row['counts']) == len(row['endpoint_error_m']) == 3, 'Validation horizons mismatch')
        need(all(isinstance(c, int) and 0 < c <= row['points']*7*6 for c in row['counts']), 'Validation target counts')
        need(all(math.isfinite(x) and x >= 0 for x in row['endpoint_error_m']), 'Invalid validation error')
        need(coverage is None or coverage == row['counts'], 'Validation coverage changed during training')
        coverage = row['counts']
    return dict(training_rows=len(training), validation_rows=len(validation), validation_counts=coverage,
                last_logged_elapsed_seconds=previous)


def original_child_absent(child):
    try:
        process = psutil.Process(child['pid'])
        need(process.create_time() != child['creation_time'], 'Original training child remains present')
        return 'pid_reused_original_absent'
    except psutil.NoSuchProcess:
        return 'pid_absent'


def inspect_operation(operation, admission, child, ledger):
    identity = dict(arm=admission['arm'], seed=admission['seed'])
    need(operation['run_identity'] == identity and operation['record_sha256'] == admission['record_sha256'],
         'Operations identity/protocol mismatch')
    need(operation['status'] == 'awaiting_literature_review' and operation['returncode'] == 0
         and operation['child_terminal'] and operation['inputs_unchanged'] and not operation['stalled']
         and not operation['actions'] and not operation['errors'], 'Unsuccessful/nonterminal operations')
    need(math.isfinite(operation['elapsed_seconds']) and operation['elapsed_seconds'] > 0, 'Invalid operation elapsed time')
    need(operation['interval_id'] == admission['interval']['id'], 'Operation interval mismatch')
    matches = [r for r in ledger if r['id'] == operation['interval_id']]
    need(len(matches) == 1, 'Missing/duplicate ledger interval')
    row = matches[0]
    need(row['end'] is not None and math.isfinite(row['end']) and row['end'] >= row['start'], 'Unclosed/invalid interval')
    need({k: row[k] for k in admission['interval'] if k != 'end'} ==
         {k: v for k, v in admission['interval'].items() if k != 'end'}, 'Ledger identity changed')
    need(row.get('closure') == 'F004 supervisor: owned child terminal confirmed', 'Unverified interval closure')
    need(child['cwd'] == admission['cwd'] and child['creation_time'] >= admission['supervisor_created'], 'Child identity mismatch')
    expected_tail = [str(Path(admission['cwd'])/'tools/research/f004_train.py'), '--admission',
                     str(Path(admission['science_out']).parent/'operations/admission.json')]
    need(child['argv'][1:] == expected_tail, 'Child command mismatch')
    return dict(child_state=original_child_absent(child), interval_seconds=row['end']-row['start'])


def inspect(arm):
    record, rh = require_record('F004')
    folder = BASE/'runs'/f'{arm}_s0'
    opdir, science = folder/'operations', folder/'science'
    # Refuse live/partial runs before loading any scientific checkpoint.
    operation, admission, child = [read(opdir/name) for name in ('result.json', 'admission.json', 'child.json')]
    need(admission['arm'] == arm and admission['seed'] == 0 and admission['record_sha256'] == rh, 'Admission identity mismatch')
    need(admission['model'] == record['protocol']['model'] and admission['settings'] == record['protocol']['training'],
         'Admission settings differ from frozen F004')
    need(digest(opdir/'admission.json') == operation['admission_sha256'], 'Changed admission')
    operational = inspect_operation(operation, admission, child, read(admission['ledger']))
    pinned = Path(admission['cwd'])
    need(Path(admission['science_out']).resolve() == science.resolve(), 'Wrong scientific output')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pinned, text=True).strip()
    need(commit == admission['source_commit'], 'Pinned checkout moved')
    verify_files(pinned, admission['inputs_sha256'])
    verify_files(pinned, admission['source_sha256'])
    verify_files(ROOT, admission['source_sha256'])
    artifacts = {str(p.relative_to(science)): digest(p) for p in science.rglob('*') if p.is_file()}
    need(artifacts == operation['child_artifact_sha256'], 'Scientific artifact inventory changed')
    result, manifest = read(science/'result.json'), read(science/'manifest.json')
    need(all(manifest[k] == v for k, v in admission.items()), 'Science manifest differs from admission')
    need(manifest['device'] and manifest['torch_version'] == torch.__version__ and manifest['hip_version'], 'Missing/different training runtime')
    binding = expected_binding(admission)
    settings = admission['settings']
    need(result['run_identity'] == dict(arm=arm, seed=0) and result['record_sha256'] == rh
         and result['binding'] == binding and result['final_step'] == settings['steps']
         and result['status'] == 'awaiting_literature_review' and result['selection'] == 'final_step_only',
         'Scientific result identity/selection mismatch')
    for key, filename in (('checkpoint_sha256', 'model.pt'), ('initial_sha256', 'initial.pt'),
                          ('training_log_sha256', 'training.jsonl'), ('validation_log_sha256', 'validation.jsonl')):
        need(result[key] == artifacts[filename], f'Result digest mismatch: {filename}')
    train_n = len(np.load(BASE/'data/train/episode.npy', mmap_mode='r'))
    val_n = len(np.load(BASE/'data/val/episode.npy', mmap_mode='r'))
    periodic = list(range(settings['checkpoint_every'], settings['steps']+1, settings['checkpoint_every']))
    expected_names = {'initial.pt', 'model.pt', *(f'checkpoint_{s:05d}.pt' for s in periodic)}
    need({p.name for p in science.glob('*.pt')} == expected_names, 'Missing/extra checkpoints')
    rng_states = replay_sampler(train_n, settings, periodic)
    torch.manual_seed(settings['seed'])
    model = ObjectJEPA(ObjectConfig(**admission['model']))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    initial_rng = torch.get_rng_state().clone()
    measurements = []
    for step, filename in [(0, 'initial.pt')]+[(s, f'checkpoint_{s:05d}.pt') for s in periodic]+[(settings['steps'], 'model.pt')]:
        checkpoint = torch.load(science/filename, map_location='cpu', weights_only=True)
        measured = inspect_checkpoint(checkpoint, model, initial, initial_rng, binding, step, train_n, rng_states[step])
        measurements.append(dict(filename=filename, **measured))
    last_periodic = torch.load(science/f'checkpoint_{settings["steps"]:05d}.pt', map_location='cpu', weights_only=True)
    need(equal_tree(checkpoint, last_periodic), 'Final validation changed checkpoint state')
    training, validation = [[json.loads(line) for line in (science/name).read_text().splitlines()]
                            for name in ('training.jsonl', 'validation.jsonl')]
    logs = inspect_logs(training, validation, settings, val_n)
    progress = read(science/'progress.json')
    need(progress['step'] == settings['steps'] and math.isfinite(progress['elapsed_seconds'])
         and 0 < progress['elapsed_seconds'] <= result['elapsed_seconds'], 'Final progress mismatch')
    need(logs['last_logged_elapsed_seconds'] <= result['elapsed_seconds'] <= operation['elapsed_seconds'], 'Terminal elapsed-time mismatch')
    return dict(research_id='F004', phase='training_inspection', status='awaiting_literature_review',
                run_identity=dict(arm=arm, seed=0), record_sha256=rh,
                created_at=datetime.now(timezone.utc).isoformat(),
                science_result_sha256=digest(science/'result.json'), operations_result_sha256=digest(opdir/'result.json'),
                admission_sha256=digest(opdir/'admission.json'), artifact_sha256=artifacts,
                source_sha256={p: digest(ROOT/p) for p in SOURCES},
                preimplementation_sha256=digest(BASE/'inspection_preimplementation.json'),
                checkpoints=measurements, logs=logs, operations=operational,
                limits='Terminal integrity only. No physical forecast, GPU numerical/restart equivalence or benefit conclusion.',
                next_step='Consult literature and review actual training, inspection and operations separately before successor work.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', required=True, choices=ARMS)
    args = ap.parse_args()
    out = BASE/'runs'/f'{args.arm}_s0'/'inspection.json'
    need(not out.exists(), 'Refusing inspection overwrite')
    review = require_review(BASE/'inspection_software/review.json')
    need(review['status'] == 'passed', 'Unqualified inspection software')
    qualified = read(BASE/'inspection_software'/review['result'])
    verify_files(ROOT, qualified['source_sha256'])
    need(qualified['preimplementation_sha256'] == digest(BASE/'inspection_preimplementation.json'), 'Stale inspection protocol')
    torch.set_num_threads(2)
    result = inspect(args.arm)
    need(not torch.cuda.is_initialized(), 'CPU inspection initialized GPU')
    atomic_json(out, result)
    print(json.dumps(dict(path=str(out), status=result['status'], sha256=digest(out))))


if __name__ == '__main__':
    main()
