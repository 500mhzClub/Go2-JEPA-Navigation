"""A015 CPU-only final audit. Run from the original pinned checkout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    original = Path.cwd().resolve()
    validation = original.parent/'Go2-JEPA-Navigation-p25-validation'
    sys.path[:0] = [str(original), str(validation)]
    from go2nav.research import atomic_json, digest, require_record, require_review, verify_files
    record, rh = require_record('A015', records=validation/'docs/research/records')
    base = original/'outputs/gated/A014'
    plan = json.loads((base/'continuation_plan.json').read_text())
    command = next(c for c in plan['recovery_commands'] if c['name'] == args.run)
    assert original == Path(command['cwd']).resolve()
    run = base/'runs'/args.run
    execution = json.loads((run/'execution.json').read_text())
    if execution['status'] != 'exited_awaiting_literature_review' or execution.get('returncode') != 0:
        raise SystemExit('Refusing final audit: run is not a successful terminal execution.')
    try:
        process = psutil.Process(execution['pid'])
        if process.create_time() == execution['creation_time'] and process.status() != psutil.STATUS_ZOMBIE:
            raise SystemExit('Refusing final audit: original child identity remains live.')
    except psutil.NoSuchProcess:
        pass
    result = json.loads((run/'result.json').read_text())
    assert result['execution_sha256'] == digest(run/'execution.json') and result['returncode'] == 0
    for name in ['preparation','software']:
        assert require_review(base/name/'review.json')['status'] == 'passed'
        assert execution[name+'_review_sha256'] == digest(base/name/'review.json')
    assert require_review('outputs/gated/A015/software/review.json')['status'] == 'passed'
    software = json.loads(Path('outputs/gated/A015/software/result.json').read_text())
    verify_files(Path(__file__).parent, software['source_sha256'])
    prepared = json.loads((base/'preparation/result.json').read_text())
    assert prepared['plan_sha256'] == digest(base/'continuation_plan.json')
    assert execution['command'] == command and execution['start_step'] == command['start_step']
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=original, text=True).strip() == plan['original_source_commit']
    verify_files(original, plan['original_source_sha256'])
    verify_files(original, command['data_sha256'])
    root = original/command['argv'][command['argv'].index('--out')+1]
    assert digest(root/'config.json') == command['input_sha256']['config.json']
    lineage = json.loads((root/'recovery_lineage.json').read_text())
    initial_root = Path(lineage['source'])
    assert lineage['source_checkpoint_sha256'] == command['checkpoint_sha256']
    verify_files(initial_root, command['input_sha256'])
    assert os.environ.get('HIP_VISIBLE_DEVICES') == os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from go2nav.training import balls
    from state_contract import audit, read_rows
    assert Path(balls.__file__).resolve() == original/'go2nav/training/balls.py'
    assert not torch.cuda.is_initialized()
    out = original/'outputs/gated/A015/finals'/args.run
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    manifest = dict(research_id='A015', research_sha256=rh, name=args.run, started_at=time.time(),
                    execution_sha256=digest(run/'execution.json'), continuation_result_sha256=digest(run/'result.json'),
                    initial_checkpoint_sha256=command['checkpoint_sha256'], plan_sha256=digest(base/'continuation_plan.json'),
                    original_source_commit=plan['original_source_commit'], software_review_sha256=digest('outputs/gated/A015/software/review.json'))
    atomic_json(out/'manifest.json', manifest)
    try:
        for name in ['checkpoint.pt','model.pt','config.json','metrics.jsonl','recovery_lineage.json']:
            with (root/name).open('rb') as source, (out/name).open('xb') as target:
                shutil.copyfileobj(source, target)
            hashes[name] = digest(out/name)
            assert hashes[name] == digest(root/name)
        manifest['snapshot_sha256'] = hashes
        atomic_json(out/'manifest.json', manifest)
        full = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False)
        final = torch.load(out/'model.pt', map_location='cpu', weights_only=False)
        initial = torch.load(initial_root/'checkpoint.pt', map_location='cpu', weights_only=False)
        cfg = json.loads((out/'config.json').read_text())
        checks = audit(full, final, cfg, initial, read_rows(out/'metrics.jsonl'), read_rows(initial_root/'metrics.jsonl'), balls)
        verify_files(root, hashes)
        verify_files(original, plan['original_source_sha256'])
        atomic_json(out/'result.json', dict(research_id='A015', status='passed', name=args.run,
                    manifest_sha256=digest(out/'manifest.json'), **checks,
                    checkpoint_sha256=hashes['checkpoint.pt'], model_sha256=hashes['model.pt'],
                    cpu_only=True, optimizer_updates_executed=0,
                    next_step='Literature review of actual audit, then bind A014 final review; A003 remains pending.'))
    except BaseException as exc:
        atomic_json(out/'result.json', dict(research_id='A015', status='technical_blocked', name=args.run,
                    manifest_sha256=digest(out/'manifest.json'), error=repr(exc), cpu_only=True))
        raise
    print((out/'result.json').read_text())

if __name__ == '__main__':
    main()
