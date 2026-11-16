"""CPU-only original balls completion contract; no training or inference."""
import json
import math
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, name='state'):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b), name+' tensor mismatch')
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), name+' keys mismatch')
        for key in a:
            equal(a[key], b[key], name+'.'+str(key))
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), name+' length mismatch')
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, name+'.'+str(i))
    else:
        require(a == b, name+' value mismatch')


def finite(value, name='state'):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu', name+' must be on CPU')
        if value.is_floating_point() or value.is_complex():
            require(torch.isfinite(value).all().item(), name+' nonfinite tensor')
    elif isinstance(value, dict):
        for k, v in value.items():
            finite(v, name+'.'+str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite(v, name+'.'+str(i))
    elif isinstance(value, (int, float)):
        require(math.isfinite(value), name+' nonfinite number')


def counter(value, name):
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, name+' nonscalar counter')
        value = value.item()
    require(isinstance(value, (int, float)) and math.isfinite(value)
            and value >= 0 and value == int(value), name+' invalid counter')
    return int(value)


def json_equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def read_rows(path):
    text = path.read_text()
    require(text.endswith('\n'), 'metrics incomplete trailing row')
    return [json.loads(line) for line in text.splitlines() if line]


def audit(full, final, cfg, initial, rows, initial_rows, balls):
    total = cfg['total_steps']
    require(total == 303000, 'recipe total must be303000')
    require(cfg['train'] == balls.DEFAULTS and cfg['seed'] == 0
            and cfg['run_seed'] == 2026111500, 'original recipe mismatch')
    require(initial['step'] in (105000, 155000), 'unexpected initial step')
    for label, state in [('full', full), ('final', final)]:
        require(state.get('step') == total and not state.get('inference_only'), label+' unfinished/inference-only')
        for field, config_field in [('kind','kind'), ('n','n'), ('train','train'), ('config','model')]:
            require(json_equal(state.get(field), cfg[config_field]), label+' config mismatch: '+field)
        finite(state, label)
    for field in ['model_state_dict', 'head_state_dict']:
        equal(full[field], final[field], 'final/full '+field)
    model = balls.LeWorldModel(balls.model_config())
    model.load_state_dict(full['model_state_dict'], strict=True)
    equal(model.state_dict(), full['model_state_dict'], 'model restore')
    head = balls.FourierHead() if cfg['kind'] == 'l1' else None
    if head is None:
        require(full['head_state_dict'] is None, 'unexpected head')
    else:
        head.load_state_dict(full['head_state_dict'], strict=True)
        equal(head.state_dict(), full['head_state_dict'], 'head restore')
    params = list(model.parameters()) + ([] if head is None else list(head.parameters()))
    opt, old = full['optimiser'], initial['optimiser']
    require(len(opt['param_groups']) == len(old['param_groups']) == 1, 'optimizer groups')
    group, old_group = opt['param_groups'][0], old['param_groups'][0]
    require(group['params'] == old_group['params'] and len(group['params']) == len(params)
            and len(set(group['params'])) == len(params), 'optimizer parameter mapping')
    expected_lr = cfg['train']['lr'] * balls.schedule(cfg['train']['warmup'], total)(total)
    equal(group, old_group | {'lr': expected_lr}, 'optimizer group metadata')
    require(opt['state'].keys() == old['state'].keys() and bool(opt['state']), 'optimizer state coverage')
    require(set(opt['state']) <= set(group['params']), 'optimizer unknown parameter state')
    delta = total-initial['step']
    old_counters, counters = [], []
    for key, param in zip(group['params'], params):
        if key not in opt['state']:
            continue
        state, old_state = opt['state'][key], old['state'][key]
        require(state.keys() == old_state.keys(), 'optimizer per-parameter fields')
        for field in ['exp_avg', 'exp_avg_sq']:
            require(state[field].shape == param.shape and state[field].dtype == param.dtype,
                    'optimizer moment shape/dtype: '+str(key)+'.'+field)
        a, b = counter(old_state['step'], 'input'), counter(state['step'], 'final')
        require(a <= initial['step'] and b == a+delta, 'optimizer update counter mismatch')
        old_counters.append(a); counters.append(b)
    scheduler_state = full['scheduler']
    old_scheduler = initial['scheduler']
    expected_scheduler = old_scheduler | {'last_epoch':total,
                                         '_step_count':old_scheduler['_step_count']+delta,
                                         '_last_lr':[expected_lr]}
    equal(scheduler_state, expected_scheduler, 'scheduler metadata')
    require(scheduler_state['base_lrs'] == [cfg['train']['lr']], 'scheduler base lr')
    optimizer = torch.optim.AdamW(params, lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, balls.schedule(cfg['train']['warmup'], total))
    optimizer.load_state_dict(opt)
    scheduler.load_state_dict(scheduler_state)
    equal(optimizer.state_dict(), opt, 'optimizer restore')
    equal(scheduler.state_dict(), scheduler_state, 'scheduler restore')
    require(scheduler.last_epoch == total and scheduler.get_last_lr() == [expected_lr], 'scheduler final cursor/lr')
    for key in ['rng', 'cuda_rng']:
        value = full[key]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                and value.shape == initial[key].shape, key+' structure mismatch')
    torch.set_rng_state(full['rng'])
    equal(torch.get_rng_state(), full['rng'], 'CPU RNG restore')
    finite(rows, 'metrics')
    train_steps = [r['step'] for r in rows if 'validation' not in r]
    val_steps = [r['step'] for r in rows if 'validation' in r]
    expected_train = list(range(cfg['train']['log_every'], total+1, cfg['train']['log_every']))
    expected_val = list(range(cfg['train']['validate_every'], total+1, cfg['train']['validate_every']))
    if total not in expected_val:
        expected_val.append(total)
    require(train_steps == expected_train, 'training metric coverage/order/duplicates')
    require(val_steps == expected_val, 'validation metric coverage/order/duplicates')
    order = [(r['step'], int('validation' in r)) for r in rows]
    require(order == sorted(order), 'combined metric order')
    history = [r for r in rows if r['step'] <= initial['step']]
    expected_history = [r for r in initial_rows if r['step'] <= initial['step']]
    equal(history, expected_history, 'preserved metric history')
    losses = ['pred','sigreg','loss'] + (['aux_encoded','aux_predicted'] if head is not None else [])
    for row in rows:
        values = row.get('validation', row)
        required = losses if 'validation' in row else losses+['epoch','lr','grad','steps_per_s']
        require(all(k in values and isinstance(values[k], (int, float)) for k in required), 'metric required fields')
        if 'validation' not in row and row['step'] > initial['step']:
            lr = cfg['train']['lr'] * balls.schedule(cfg['train']['warmup'], total)(row['step'])
            require(math.isclose(row['lr'], lr, rel_tol=1e-12, abs_tol=1e-15), 'metric learning rate mismatch')
    require(not torch.cuda.is_initialized(), 'unexpected GPU initialization')
    return dict(step=total, full_state_restored_exactly=True, final_matches_full=True,
                all_state_and_logged_values_finite=True, training_rows=len(train_steps), validation_rows=len(val_steps),
                optimizer_populated_states=len(counters), optimizer_step_min=min(counters), optimizer_step_max=max(counters),
                input_optimizer_step_min=min(old_counters), input_optimizer_step_max=max(old_counters),
                prior_optimizer_counter_deficit=initial['step']-min(old_counters),
                resumed_optimizer_updates_verified=delta, cpu_rng_restored=True,
                gpu_rng_structure_checked_only=True, cuda_initialized=False)
