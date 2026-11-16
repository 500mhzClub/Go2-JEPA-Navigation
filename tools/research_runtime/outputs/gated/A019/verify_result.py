"""CPU artifact verification after A019; never reruns a model or creates a review."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files
from stages import inputs, route


def need(value, message):
    if not value:
        raise ValueError(message)


def artifact_checks(source, derived, result, kind, seed, source_hash):
    import torch
    from go2nav.training.balls import LeWorldModel, model_config, FourierHead
    from dataclasses import asdict
    identity = source['identity']
    need((source['kind'], source['n'], source['step'], source['research_id']) == (kind, 3, 303000, 'C001'), 'Wrong source identity')
    need((identity['kind'], identity['n'], identity['seed'], identity['run_seed']) == (kind, 3, seed, 2026111500+seed), 'Wrong source seed')
    need((result['research_id'], result['kind'], result['n'], result['step']) == ('A019', kind, 3, 303000), 'Wrong numerical identity')
    need(result['status'] == 'awaiting_literature_review' and not result['invalid_state'] and result['weights_unchanged'], 'Invalid numerical outcome')
    need(result['source_sha256'] == derived['source_sha256'] == source_hash, 'Wrong source digest')
    expected = {'config','kind','n','train','head_state_dict','model_state_dict','step','research_id',
                'calibration','source_sha256','inference_only','inference_dtype','attention_backend'}
    need(set(derived) == expected and derived['inference_only'] is True, 'Not exact inference-only export')
    need((derived['research_id'], derived['inference_dtype'], derived['attention_backend']) == ('A019','float64','math'), 'Wrong inference policy')
    for key in ['config','kind','n','train','step']:
        need(derived[key] == source[key], 'Changed source configuration: '+key)
    need(json.dumps(source['config'],sort_keys=True) == json.dumps(asdict(model_config()),sort_keys=True), 'Wrong model architecture')
    model = LeWorldModel(model_config())
    model.load_state_dict(source['model_state_dict'], strict=True)
    names = set(dict(model.named_parameters())); need(len(names) == 127, 'Wrong base parameter coverage')
    model.double().load_state_dict(derived['model_state_dict'], strict=True)
    original, final = source['model_state_dict'], derived['model_state_dict']
    allowed = {p+'.'+k for p in ['projector.net.1','pred_proj.net.1']
               for k in ['running_mean','running_var','num_batches_tracked']}
    need(original.keys() == final.keys(), 'Changed state keys')
    changed = []
    for key, old in original.items():
        new = final[key]
        need(old.device.type == new.device.type == 'cpu' and torch.isfinite(old).all() and torch.isfinite(new).all(), 'Nonfinite/device state')
        target = old.double() if old.is_floating_point() else old
        need(new.dtype == target.dtype and new.shape == target.shape, 'Wrong export dtype/shape')
        if not torch.equal(target, new):
            need(key in allowed, 'Learned weight or unrelated buffer changed: '+key); changed.append(key)
    for p in ['projector.net.1','pred_proj.net.1']:
        need(bool((final[p+'.running_var'] >= 0).all()) and final[p+'.num_batches_tracked'].item() == 0, 'Invalid calibrated statistics')
    oldhead, newhead = source['head_state_dict'], derived['head_state_dict']
    if kind == 'l0':
        need(oldhead is None and newhead is None, 'Unexpected head')
    else:
        head = FourierHead(); head.load_state_dict(oldhead, strict=True)
        need(oldhead.keys() == newhead.keys(), 'Changed head keys')
        for k,v in oldhead.items():
            need(v.dtype == newhead[k].dtype and v.shape == newhead[k].shape and torch.isfinite(v).all()
                 and torch.equal(v,newhead[k]), 'Changed/nonfinite head')
    need(result['calibration'] == derived['calibration'], 'Calibration metadata differs')
    for p, count in [('projector.net.1',16384), ('pred_proj.net.1',12288)]:
        entry = result['calibration'][p]; need(entry['count'] == count, 'Wrong calibration sample count')
        for key in ['mean_change','variance_change']:
            need(all(math.isfinite(entry[key][k]) and entry[key][k]>=0 for k in ['max_abs','rms']), 'Nonfinite calibration diagnostic')
            field = p+('.running_mean' if key=='mean_change' else '.running_var')
            old, new = original[field].double(), final[field].float().double()
            delta = old-new
            need(entry[key]['passed'] == bool(torch.allclose(old,new,rtol=1e-4,atol=1e-5)), 'Wrong calibration comparison flag')
            for k, value in [('max_abs',float(delta.abs().max())),('rms',float(delta.square().mean().sqrt()))]:
                need(math.isclose(entry[key][k],value,rel_tol=1e-12,abs_tol=1e-12), 'Wrong calibration change diagnostic')
    count = 0; maximum = 0.
    def comparison(item):
        nonlocal count, maximum
        need(item['passed'] is True and all(math.isfinite(item[k]) and item[k]>=0 for k in ['max_abs','rms']), 'Invalid numerical comparison')
        count += 1; maximum = max(maximum, item['max_abs'])
    for phase in ['before','after']:
        x = result[phase]
        for split in ['train','val']:
            need(set(x[split]) == {'saved','encoder_batch','predictor_batch','both_batch'}, 'Incomplete batch modes')
            for item in x[split].values():
                need(item['finite'] is True and all(math.isfinite(item[k]) and item[k]>=0 for k in ['pred_mse','latent_std']), 'Invalid mode diagnostic')
        need(set(x['invariance']) == {'repeat','single','permuted'}, 'Incomplete grouping checks')
        for group in x['invariance'].values():
            need(set(group) == {'encoded','predicted'}, 'Wrong grouping output coverage')
            for item in group.values(): comparison(item)
        parity = x['parity']
        need(parity['passed'] is True and not parity['nonfinite_gradients'], 'Invalid parity result')
        need(set(parity['gradients']) == names and set(parity['forward']) == {'encoded','predicted'}, 'Incomplete gradient/forward coverage')
        for item in [*parity['forward'].values(), *parity['gradients'].values()]: comparison(item)
    need(count == 270 and not torch.cuda.is_initialized(), 'Wrong comparison count or GPU initialized')
    return dict(reported_comparisons=270, reported_max_abs=maximum, gradient_tensors_per_phase=127,
                learned_weights_and_head_exact=True, changed_buffers=changed,
                finite_float64_math_inference_only=True, cuda_initialized=False)


def run(name, consultation):
    need(all(os.environ.get(k)=='' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']), 'CPU-only environment required')
    base = Path('outputs/gated/A019'); _, rh = require_record('A019')
    need(require_review(base/'software/review.json')['status']=='passed', 'Unreviewed verifier')
    sw = json.loads((base/'software/result.json').read_text()); need(sw['research_sha256']==rh, 'Changed protocol')
    verify_files(base,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    p = route(Path.cwd(), name); files = inputs(p); science, ops = p['science'],p['operations']
    o = json.loads((ops/'result.json').read_text()); om = json.loads((ops/'manifest.json').read_text())
    need(o['name']==name and o['child_terminal'] and o['returncode']==0 and not o['timed_out'] and not o['actions'] and not o['errors'] and o['inputs_unchanged'], 'Incomplete operation')
    need(o['manifest_sha256']==digest(ops/'manifest.json'), 'Changed operation manifest')
    try:need(psutil.Process(om['pid']).create_time()!=om['creation_time'],'Exact numerical child still exists')
    except psutil.NoSuchProcess:pass
    verify_files(science,o['child_artifact_sha256']);verify_files(Path.cwd(),om['preconsultation']['inputs_sha256'])
    sm = json.loads((science/'manifest.json').read_text())
    need(sm['research_sha256']==rh and sm['research_id']=='A019', 'Wrong numerical manifest')
    rows = json.loads(Path('outputs/gated/gpu_intervals.json').read_text())
    outer = [r for r in rows if r['id']==om['outer_interval_id']]
    inner = [r for r in rows if r['id']==sm['run_id']]
    need(len(outer)==len(inner)==1 and outer[0]['id']!=inner[0]['id'], 'Missing/ambiguous intervals')
    for row,folder,supervisor in [(outer[0],ops,'A019'),(inner[0],science,None)]:
        need(row['research_id']=='A019' and row.get('supervisor_research_id')==supervisor
             and (Path(om['cwd'])/row['out']).resolve()==folder.resolve()
             and row['end'] is not None and row['end']>=row['start'], 'Invalid closed interval')
    need(outer[0]['start']<=inner[0]['start']<=inner[0]['end']<=outer[0]['end'], 'Unnested actual intervals')
    pre = json.loads(Path(consultation).read_text())
    expected = dict(name=name,research_sha256=rh,scientific_result_sha256=digest(science/'result.json'),
                    operational_result_sha256=digest(ops/'result.json'),software_review_sha256=digest(base/'software/review.json'))
    need(all(pre.get(k)==v for k,v in expected.items()) and all(pre.get(k) for k in ['consulted_at','sources','consequences']), 'Missing/stale artifact consultation')
    started=time.monotonic();out=base/'verification'/name;out.mkdir(parents=True,exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        source=torch.load(science/'source.pt',map_location='cpu',weights_only=False)
        derived=torch.load(science/'calibrated.pt',map_location='cpu',weights_only=False)
        result=json.loads((science/'result.json').read_text())
        need(digest(science/'source.pt')==digest(p['checkpoint']) and result['calibrated_sha256']==digest(science/'calibrated.pt'),'Snapshot/export changed')
        checks=artifact_checks(source,derived,result,p['kind'],p['seed'],digest(p['checkpoint']))
        verify_files(science,o['child_artifact_sha256']);verify_files(Path.cwd(),om['preconsultation']['inputs_sha256'])
        evidence=dict(research_id='A019',name=name,kind=p['kind'],n=3,seed=p['seed'],status='awaiting_literature_review',checks=checks,
                      **{k:v for k,v in expected.items() if k!='name'},calibrated_sha256=digest(science/'calibrated.pt'),
                      preconsultation=pre,preconsultation_sha256=digest(consultation),elapsed_seconds=time.monotonic()-started,
                      limitation='CPU artifact and reported-comparison verification only; no GPU forward/backward rerun. Original runtime is source-bound. No scientific review is generated.')
        atomic_json(out/'result.json',evidence);return evidence
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='A019',name=name,status='technical_blocked',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--preconsultation',required=True)
    args=parser.parse_args();print(json.dumps(run(args.run,args.preconsultation)),flush=True)
