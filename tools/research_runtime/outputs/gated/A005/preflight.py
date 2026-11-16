"""CPU-only preparation; never signals or launches a trainer."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

original = Path.cwd().resolve()
validation = original.parent/'Go2-JEPA-Navigation-p25-validation'
sys.path[:0] = [str(original), str(validation)]
assert os.environ.get('HIP_VISIBLE_DEVICES') == ''
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
from go2nav.training import balls
from go2nav.research import atomic_json, digest, provenance, require_record, verify_files

assert Path(balls.__file__).resolve() == original/'go2nav/training/balls.py'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == '7dfc83722da0a9fb805ccc879fe7e5459709a2fd'
require_record('A005')
assert not torch.cuda.is_initialized()
assets = original/'outputs/gated/A004/recovery_assets'
index = json.loads((assets/'index.json').read_text())
out = original/'outputs/gated/A005/preparation'
out.mkdir(exist_ok=False)
recovery = original/'outputs/balls/recovery_A005'
assert not recovery.exists(), 'Refuse to overwrite any previously prepared recovery'

def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b

def finite(x):
    if isinstance(x, torch.Tensor):
        assert x.device.type == 'cpu'
        if x.is_floating_point() or x.is_complex():
            assert torch.isfinite(x).all().item()
    elif isinstance(x, dict):
        for value in x.values():
            finite(value)
    elif isinstance(x, (list, tuple)):
        for value in x:
            finite(value)

rows, inputs = {}, [assets/'index.json', Path(__file__).resolve()]
for name, expected in index.items():
    src = assets/name
    verify_files(src, expected['files'])
    cfg = json.loads((src/'config.json').read_text())
    model, head, state = balls.load(src/'checkpoint.pt', device='cpu')
    assert state['step'] == 105000 and cfg['total_steps'] == 303000
    assert cfg['seed'] == 0 and cfg['run_seed'] == 2026111500
    assert cfg['kind'] == 'l1' and cfg['n'] in (1, 3)
    assert name == f"l1_n{cfg['n']}_s0"
    for k, ck in [('kind', 'kind'), ('n', 'n'), ('train', 'train'), ('config', 'model')]:
        assert json.dumps(state[k], sort_keys=True) == json.dumps(cfg[ck], sort_keys=True)
    assert cfg['train'] == balls.DEFAULTS
    equal(model.state_dict(), state['model_state_dict'])
    equal(head.state_dict(), state['head_state_dict'])
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, balls.schedule(cfg['train']['warmup'], cfg['total_steps']))
    optimizer.load_state_dict(state['optimiser'])
    scheduler.load_state_dict(state['scheduler'])
    equal(optimizer.state_dict(), state['optimiser'])
    equal(scheduler.state_dict(), state['scheduler'])
    assert scheduler.last_epoch == state['step']
    assert scheduler.get_last_lr() == [g['lr'] for g in optimizer.param_groups]
    torch.set_rng_state(state['rng'])
    equal(torch.get_rng_state(), state['rng'])
    assert state['cuda_rng'].dtype == torch.uint8 and state['cuda_rng'].numel() == 16
    finite(state)
    dataset_files = [original/f"outputs/data/balls/n{cfg['n']}/{split}/{file}"
                     for split in ('train', 'val') for file in ('meta.json', 'labels.npz', 'frames.npy')]
    data_hashes = {str(p.relative_to(original)): digest(p) for p in dataset_files}
    inputs.extend(src/f for f in expected['files'])
    rows[name] = dict(step=state['step'], state_restored_exactly=True, cpu_rng_restored_exactly=True,
                     gpu_rng_saved_not_executed=True, all_tensor_state_finite=True,
                     checkpoint_sha256=expected['checkpoint_sha256'], inputs=data_hashes)

# Create isolated mutable continuations only after both preflights passed.
for name, row in rows.items():
    dst = recovery/name
    dst.mkdir(parents=True, exist_ok=False)
    for file in ('checkpoint.pt', 'config.json', 'metrics.jsonl'):
        with (assets/name/file).open('rb') as source, (dst/file).open('xb') as target:
            shutil.copyfileobj(source, target)
        assert digest(dst/file) == index[name]['files'][file]
    row['prepared_output'] = str(dst.relative_to(original))
    row['original_checkpoint_unchanged'] = digest(original/'outputs/balls/runs'/name/'checkpoint.pt') == row['checkpoint_sha256']
    assert row['original_checkpoint_unchanged']
    atomic_json(dst/'recovery_lineage.json', dict(research_id='A005', source_commit=commit,
                original_run=f'outputs/balls/runs/{name}', immutable_input=f'outputs/gated/A004/recovery_assets/{name}',
                checkpoint_sha256=row['checkpoint_sha256'], resume_step=105000, authorization='pending', execution_started=False))

source_hashes = {str(p.relative_to(original)): digest(p) for p in sorted((original/'go2nav').rglob('*.py'))}
manifest = provenance('A005', dict(stage='cpu_preparation', original_commit=commit), inputs)
manifest['original_source_sha256'] = source_hashes
atomic_json(out/'manifest.json', manifest)
assert not torch.cuda.is_initialized()
atomic_json(out/'result.json', dict(research_id='A005', status='passed', stage='cpu_preparation',
            original_commit=commit, runs=rows, cuda_initialized=False, gpu_seconds=0,
            optimizer_updates=0, signals_sent=0, training_launched=False, execution_authorization='pending'))
print(json.dumps(dict(status='preparation_passed', runs=list(rows), execution='not_started', cuda_initialized=False)))
