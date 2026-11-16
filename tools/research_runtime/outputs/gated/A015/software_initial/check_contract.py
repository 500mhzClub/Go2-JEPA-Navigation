"""A015 fault checks using synthetic final metadata over a preserved real schema."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

original = Path.cwd().resolve()
validation = original.parent/'Go2-JEPA-Navigation-p25-validation'
sys.path[:0] = [str(original), str(validation)]
assert os.environ.get('HIP_VISIBLE_DEVICES') == os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
from go2nav.training import balls
from go2nav.research import atomic_json, digest, require_record
from state_contract import audit, read_rows

require_record('A015', records=validation/'docs/research/records')
base = original/'outputs/gated/A015'
out = base/'software'
out.mkdir(exist_ok=False)
checks = {}
started = time.time()
try:
    for name in ['state_contract.py','verify_final.py','check_contract.py']:
        ast.parse((base/name).read_text())
    source = original/'outputs/balls/recovery_A005/l1_n1_s0'
    cfg = json.loads((source/'config.json').read_text())
    initial = torch.load(source/'checkpoint.pt', map_location='cpu', weights_only=False)
    initial_rows = read_rows(source/'metrics.jsonl')
    full = copy.deepcopy(initial)
    total, delta = cfg['total_steps'], cfg['total_steps']-initial['step']
    full['step'] = total
    for state in full['optimiser']['state'].values():
        state['step'] += delta
    full['optimiser']['param_groups'][0]['lr'] = 0.
    full['scheduler']['last_epoch'] = total
    full['scheduler']['_step_count'] += delta
    full['scheduler']['_last_lr'] = [0.]
    final = {k:copy.deepcopy(full[k]) for k in ['config','model_state_dict','kind','n','train','head_state_dict','step']}
    rows = copy.deepcopy([r for r in initial_rows if r['step'] <= initial['step']])
    train_template = next(r for r in reversed(rows) if 'validation' not in r)
    val_template = next(r for r in reversed(rows) if 'validation' in r)
    for step in range(initial['step']+500, total+1, 500):
        row = copy.deepcopy(train_template)
        row.update(step=step, epoch=(step-1)//(total//cfg['train']['epochs']),
                   lr=cfg['train']['lr']*balls.schedule(cfg['train']['warmup'], total)(step))
        rows.append(row)
        if step % cfg['train']['validate_every'] == 0 or step == total:
            row = copy.deepcopy(val_template); row['step'] = step; rows.append(row)
    positive = audit(full, final, cfg, initial, rows, initial_rows, balls)
    checks['synthetic_complete_schema_accepted'] = True
    def reject(name, mutate):
        a,b,c,r = copy.deepcopy(full),copy.deepcopy(final),copy.deepcopy(cfg),copy.deepcopy(rows)
        mutate(a,b,c,r)
        try:
            audit(a,b,c,initial,r,initial_rows,balls)
        except (ValueError,KeyError,RuntimeError,AssertionError) as exc:
            checks[name] = dict(rejected=True,error=str(exc))
        else:
            raise AssertionError('Fault accepted: '+name)
    reject('unfinished_step',lambda a,b,c,r:a.update(step=302500))
    reject('inference_only',lambda a,b,c,r:b.update(inference_only=True))
    reject('config_mismatch',lambda a,b,c,r:a.update(n=3))
    key = next(iter(full['model_state_dict']))
    reject('model_full_mismatch',lambda a,b,c,r:b['model_state_dict'][key].add_(1))
    reject('nonfinite_model_state',lambda a,b,c,r:a['model_state_dict'][key].fill_(float('nan')))
    opt_key = next(iter(full['optimiser']['state']))
    reject('missing_optimizer_state',lambda a,b,c,r:a['optimiser']['state'].pop(opt_key))
    reject('wrong_moment_shape',lambda a,b,c,r:a['optimiser']['state'][opt_key].update(exp_avg=torch.zeros(1)))
    reject('wrong_optimizer_counter',lambda a,b,c,r:a['optimiser']['state'][opt_key]['step'].sub_(1))
    reject('wrong_scheduler_epoch',lambda a,b,c,r:a['scheduler'].update(last_epoch=302999))
    reject('invalid_gpu_rng_shape',lambda a,b,c,r:a.update(cuda_rng=torch.empty(0,dtype=torch.uint8)))
    reject('missing_metric_row',lambda a,b,c,r:r.pop(-2))
    reject('duplicate_metric_row',lambda a,b,c,r:r.insert(-1,copy.deepcopy(r[-2])))
    reject('nonfinite_metric',lambda a,b,c,r:r[-2].update(loss=float('inf')))
    reject('wrong_metric_lr',lambda a,b,c,r:r[-2].update(lr=1.))
    # The live-run guard must fail before output creation or GPU admission.
    ledger = original/'outputs/gated/gpu_intervals.json';before = digest(ledger)
    refusal = subprocess.run([sys.executable,str(base/'verify_final.py'),'--run','l1_n1_s0'],
                             cwd=original,capture_output=True,text=True,timeout=20)
    (out/'live_run_refusal.txt').write_text(refusal.stdout+refusal.stderr)
    assert refusal.returncode != 0 and 'not a successful terminal execution' in refusal.stderr
    assert not (base/'finals/l1_n1_s0').exists() and digest(ledger)==before
    checks['live_run_refused_without_audit_output_or_gpu_interval'] = True
    assert not torch.cuda.is_initialized()
    atomic_json(out/'result.json',dict(research_id='A015',status='passed',elapsed_seconds=time.time()-started,
                checks=checks,positive_fixture_checks=positive,
                fixture='Synthetic final counters/metric rows over immutable105000 real-schema checkpoint; not trained to303000',
                input_checkpoint_sha256=digest(source/'checkpoint.pt'),
                source_sha256={n:digest(base/n) for n in ['state_contract.py','verify_final.py','check_contract.py']},
                cuda_initialized=False,optimizer_updates_executed=0,actual_final_runs_audited=0))
    print(json.dumps(dict(checks=len(checks),status='passed',elapsed_seconds=time.time()-started,actual_final_runs=0)))
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='A015',status='technical_blocked',checks=checks,error=repr(exc),
                source_sha256={n:digest(base/n) for n in ['state_contract.py','verify_final.py','check_contract.py']}))
    raise
