"""Reviewed additional-seed inference and immutable original probe-data admission."""
import importlib.util
import json
from pathlib import Path
from go2nav.research import digest, require_record, require_review, reviewed_artifact, verify_files

ORDER=['l0_n3_s1','l1_n3_s1','l0_n3_s2','l1_n3_s2']


def need(value,message):
    if not value:raise ValueError(message)


def numerical_parent(root,name):
    need(name in ORDER,'Unplanned model/count/seed');root=Path(root)
    base=Path('outputs/gated/A019')
    need(require_review(base/'software/review.json')['status']=='passed','Numerical software not reviewed')
    sw=json.loads((base/'software/result.json').read_text())
    need(sw['research_sha256']==require_record('A019')[1],'Numerical protocol changed')
    verify_files(base,sw['source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    spec=importlib.util.spec_from_file_location('c011_a019_parents',base/'stages.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=set(module.inputs(module.route(root,name)))
    parent=root/'outputs/gated/A019/inference'/name;checkpoint=parent/'calibrated.pt'
    review=require_review(parent/'review.json');raw=reviewed_artifact(parent/'review.json',checkpoint,'calibrated_sha256')
    config_path=root/'outputs/gated/C001'/name/'config.json';config=json.loads(config_path.read_text())
    kind,n,seed=name[:2],3,int(name[-1])
    need((config['kind'],config['n'],config['seed'])==(kind,n,seed),'Wrong training identity')
    need(review['status']=='passed' and review['research_id']=='A019' and raw['research_id']=='A019','Wrong numerical review identity/status')
    need((raw['kind'],raw['n'],raw['step'])==(kind,n,303000) and raw['weights_unchanged'] and not raw['invalid_state'],'Incomplete numerical result')
    need(review.get('run_identity')==dict(kind=kind,n=n,seed=seed),'Missing/wrong verified numerical seed identity')
    need(raw['source_sha256']==digest(root/'outputs/gated/C001'/name/'model.pt')==digest(parent/'source.pt'),'Wrong numerical source')
    verification=root/'outputs/gated/A019/verification'/name
    need((root/review['verification_review']).resolve()==(verification/'review.json').resolve()
         and review['verification_review_sha256']==digest(verification/'review.json'),'Wrong/stale artifact verification binding')
    vr=require_review(verification/'review.json');v=json.loads((verification/vr['result']).read_text())
    need(vr['status']=='passed' and vr['research_id']=='A019' and
         (v['research_id'],v['name'],v['kind'],v['n'],v['seed'])==('A019',name,kind,n,seed),'Wrong artifact verification identity')
    need(v['calibrated_sha256']==digest(checkpoint) and v['scientific_result_sha256']==digest(parent/'result.json'),'Changed verified numerical artifact')
    c=v['checks'];need(c['reported_comparisons']==270 and c['gradient_tensors_per_phase']==127 and
         c['learned_weights_and_head_exact'] and c['finite_float64_math_inference_only'] and not c['cuda_initialized'],'Incomplete artifact checks')
    op=root/'outputs/gated/A019/runs'/name;opr=require_review(op/'review.json');o=json.loads((op/opr['result']).read_text())
    need(opr['status']=='passed' and opr['research_id']==o['research_id']=='A019' and o['name']==name,'Wrong numerical operation review')
    need(o['returncode']==0 and o['child_terminal'] and not o['timed_out'] and not o['actions'] and not o['errors'] and o['inputs_unchanged'],'Incomplete numerical operation')
    need(o['manifest_sha256']==digest(op/'manifest.json') and v['operational_result_sha256']==digest(op/'result.json'),'Changed operation evidence')
    verify_files(parent,o['child_artifact_sha256'])
    need(all(f in o['child_artifact_sha256'] for f in ['result.json','manifest.json','source.pt','calibrated.pt']),'Missing operation artifact binding')
    om=json.loads((op/'manifest.json').read_text());verify_files(root,om['preconsultation']['inputs_sha256'])
    files |= {root/f for f in om['preconsultation']['inputs_sha256']}
    files |= {parent/f for f in o['child_artifact_sha256']}
    files |= {parent/'review.json',verification/'review.json',verification/vr['result'],op/'review.json',op/opr['result'],op/'manifest.json',config_path}
    return dict(config=config,checkpoint=checkpoint,parent=parent,evidence=raw,files=files)


def original_probe_data(root):
    root=Path(root);baseline=root/'outputs/gated/probes/l0_n3_s0'
    br=require_review(baseline/'review.json');raw=json.loads((baseline/br['result']).read_text())
    need(br['status'] in ['passed','measured_negative'] and
         (raw['research_id'],raw['n'],raw['kind'],raw['seed'])==('C006',3,'l0',0),'Wrong original probe evidence')
    need(raw['manifest_sha256']==digest(baseline/'manifest.json') and raw['cases_sha256']==digest(baseline/'cases.npz'),'Changed original probe evidence')
    pre=root/'outputs/gated/C006/pre_l0_n3_s0.json';manifest=json.loads((baseline/'manifest.json').read_text())
    need(any((root/f).resolve()==pre.resolve() and h==digest(pre) for f,h in manifest['inputs'].items()),'Original data consultation changed')
    original=json.loads(pre.read_text())['data_sha256']
    names=[f'outputs/data/balls/n3/{split}/{file}' for split in ['train','test'] for file in ['meta.json','labels.npz','frames.npy']]
    hashes={f:original[f] for f in names};verify_files(root,hashes)
    controls=root/'outputs/gated/B001/n3';cr=require_review(controls/'review.json');c=json.loads((controls/cr['result']).read_text())
    need(cr['status']=='passed' and c['n']==3 and c['controls_passed'] and c['cases_sha256']==digest(controls/'cases.jsonl'),'Original controls invalid')
    cm=json.loads((controls/'manifest.json').read_text());verify_files(root,cm['inputs'])
    files={baseline/'review.json',baseline/br['result'],baseline/'manifest.json',baseline/'cases.npz',pre}
    files |= {controls/f for f in ['review.json','result.json','manifest.json','cases.jsonl']}
    files |= {root/f for f in names} | {root/f for f in cm['inputs']}
    return files,hashes


def previous_reviews(root,name):
    files=set();elapsed=0.
    for previous in ORDER[:ORDER.index(name)]:
        science=root/'outputs/gated/probes'/previous;sr=require_review(science/'review.json');s=json.loads((science/sr['result']).read_text())
        need(sr['status'] in ['passed','measured_negative'] and s['research_id']=='C011' and
             (s['kind'],s['n'],s['seed'])==(previous[:2],3,int(previous[-1])),'Wrong previous probe identity')
        need((s['train_windows'],s['test_windows'],s['test_episodes'])==(4096,5000,1000)
             and s['cases_sha256']==digest(science/'cases.npz'),'Incomplete previous probes')
        op=root/'outputs/gated/C011/supervision'/previous;opr=require_review(op/'review.json');o=json.loads((op/opr['result']).read_text())
        need(opr['status']=='passed' and o['research_id']=='C011' and o['name']==previous and
             o['returncode']==0 and o['child_terminal'] and not o['signals'] and not o['errors'] and not o['stop_reason']
             and o['inputs_unchanged'],'Incomplete previous probe operation')
        verify_files(science,o['child_artifact_sha256']);elapsed+=o['elapsed_seconds']
        files |= {science/'review.json',science/sr['result'],op/'review.json',op/opr['result']}
    need(elapsed+900<=3600,'C011 total CPU allowance exhausted')
    return files


def consultation(path,root,name,files,data,rh):
    pre=json.loads(Path(path).read_text())
    expected=dict(name=name,research_sha256=rh,inputs_sha256={str(f):digest(f) for f in sorted(files)},
                  data_sha256=data,software_review_sha256=digest(root/'outputs/gated/C011/software/review.json'))
    need(all(pre.get(k)==v for k,v in expected.items()),'Wrong/stale complete probe consultation')
    need(all(pre.get(k) for k in ['sources','consequences','consulted_at']),'Missing substantive consultation')
    need(all(all(s.get(k) for k in ['url','version','sections','consequence']) for s in pre['sources']),'Incomplete consultation source')
    return pre


def context(root,name):
    root=Path(root);record,rh=require_record('C011');need(name in ORDER==record['protocol']['order'],'Unplanned C011 run')
    base=root/'outputs/gated/C011';need(require_review(base/'software/review.json')['status']=='passed','Unreviewed C011 software')
    software=json.loads((base/'software/result.json').read_text());need(software['research_sha256']==rh,'C011 protocol changed')
    verify_files(base,software['source_sha256']);verify_files(root,software['repository_source_sha256']);verify_files(root,software['record_sha256'])
    ctx=numerical_parent(root,name);datafiles,data=original_probe_data(root)
    ctx['files'] |= datafiles | previous_reviews(root,name)
    pre=base/('pre_'+name+'.json');ctx['consult']=consultation(pre,root,name,ctx['files'],data,rh)
    ctx.update(software=software,research_sha256=rh,pre=pre);return ctx
