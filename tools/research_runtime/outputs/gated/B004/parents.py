"""Additional-seed planning admission; no model execution or automatic review."""
import importlib.util
import json
from pathlib import Path
from go2nav.research import digest,require_record,require_review,verify_files

ORDER=['l0_n3_s1','l1_n3_s1','l0_n3_s2','l1_n3_s2']


def need(v,message):
    if not v:raise ValueError(message)


def inherited():
    base=Path('outputs/gated/C011');need(require_review(base/'software/review.json')['status']=='passed','Unreviewed probe software')
    sw=json.loads((base/'software/result.json').read_text());need(sw['research_sha256']==require_record('C011')[1],'Changed C011 protocol')
    verify_files(base,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    spec=importlib.util.spec_from_file_location('b004_numerical_parents',base/'parents.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def probe_parent(root,name,ctx):
    sf=root/'outputs/gated/probes'/name;sr=require_review(sf/'review.json');s=json.loads((sf/sr['result']).read_text())
    need(sr['status'] in ['passed','measured_negative'] and sr['research_id']==s['research_id']=='C011','Wrong probe review')
    need((s['kind'],s['n'],s['seed'])==(name[:2],3,int(name[-1])),'Wrong probe identity')
    need((s['train_windows'],s['test_windows'],s['test_episodes'])==(4096,5000,1000),'Incomplete probes')
    need(s['checkpoint_sha256']==digest(ctx['checkpoint']) and s['manifest_sha256']==digest(sf/'manifest.json')
         and s['cases_sha256']==digest(sf/'cases.npz'),'Changed probe artifacts')
    op=root/'outputs/gated/C011/supervision'/name;orr=require_review(op/'review.json');o=json.loads((op/orr['result']).read_text())
    need(orr['status']=='passed' and orr['research_id']==o['research_id']=='C011' and o['name']==name,'Wrong probe operation review')
    need(o['returncode']==0 and o['child_terminal'] and not o['signals'] and not o['errors'] and not o['stop_reason']
         and o['inputs_unchanged'] and o['no_C011_gpu_intervals'],'Incomplete probe operation')
    need(all(k in o['child_artifact_sha256'] for k in ['result.json','manifest.json','cases.npz']),'Missing probe artifact binding')
    verify_files(sf,o['child_artifact_sha256'])
    manifest=json.loads((sf/'manifest.json').read_text());verify_files(root,manifest['inputs'])
    need(manifest['research_sha256']==require_record('C011')[1],'Probe protocol mismatch')
    return {sf/f for f in o['child_artifact_sha256']} | {sf/'review.json',op/'review.json',op/orr['result']} | {root/f for f in manifest['inputs']}


def previous_reviews(root,name):
    files=set();elapsed=0.
    for prev in ORDER[:ORDER.index(name)]:
        sf=root/'outputs/gated/planning'/prev;sr=require_review(sf/'review.json');s=json.loads((sf/sr['result']).read_text())
        need(sr['status'] in ['passed','measured_negative'] and sr['research_id']==s['research_id']=='B004' and
             (s['kind'],s['n'],s['seed'])==(prev[:2],3,int(prev[-1])) and s['complete_cases']==3000,'Incomplete previous planning')
        ins=root/'outputs/gated/B004/inspection'/prev;ir=require_review(ins/'review.json');i=json.loads((ins/ir['result']).read_text())
        need((root/sr['inspection_review']).resolve()==(ins/'review.json').resolve() and sr['inspection_review_sha256']==digest(ins/'review.json'),'Stale inspection binding')
        need(ir['status']=='passed' and ir['research_id']==i['research_id']=='B004' and i['name']==prev and
             i['all_task_scoring_summary_checks_passed'] and i['saved_singleton_numerics_passed'] and
             i['checks']['cases']==3000 and i['scientific_result_sha256']==digest(sf/'result.json'),'Incomplete previous inspection')
        op=root/'outputs/gated/B004/operations'/prev;orr=require_review(op/'review.json');o=json.loads((op/orr['result']).read_text())
        need(orr['status']=='passed' and orr['research_id']==o['research_id']=='B004' and o['name']==prev and
             o['returncode']==0 and o['child_terminal'] and not o['signals'] and not o['errors'] and not o['stop_reason']
             and o['inputs_unchanged'] and o['no_B004_gpu_intervals'],'Incomplete previous operation')
        need(i['operational_result_sha256']==digest(op/'result.json'),'Stale inspection operation')
        verify_files(sf,o['child_artifact_sha256']);elapsed+=o['elapsed_seconds']
        files |= {sf/'review.json',sf/sr['result'],op/'review.json',op/orr['result'],ins/'review.json',ins/ir['result']}
    need(elapsed+7200<=28800,'B004 total CPU allowance exhausted');return files


def inputs(root,name):
    root=Path(root);need(name in ORDER,'Unplanned additional planning seed')
    inherited_parents=inherited();ctx=inherited_parents.numerical_parent(root,name)
    datafiles,data=inherited_parents.original_probe_data(root)
    ctx['files'] |= datafiles | probe_parent(root,name,ctx) | previous_reviews(root,name)
    ctx['data']=data;return ctx


def consultation(path,root,name,ctx,rh):
    pre=json.loads(Path(path).read_text())
    expected=dict(name=name,research_sha256=rh,b001_record_sha256=digest('docs/research/records/B001.json'),
        checkpoint_sha256=digest(ctx['checkpoint']),inputs_sha256={str(f):digest(f) for f in sorted(ctx['files'])},
        data_sha256=ctx['data'],software_review_sha256=digest(root/'outputs/gated/B004/software/review.json'))
    need(all(pre.get(k)==v for k,v in expected.items()),'Stale/incomplete planning consultation')
    need(all(pre.get(k) for k in ['consulted_at','sources','consequences']) and
         all(all(s.get(k) for k in ['url','version','sections','consequence']) for s in pre['sources']),'Missing substantive consultation')
    return pre


def context(root,name):
    root=Path(root);record,rh=require_record('B004');need(record['protocol']['order']==ORDER,'Changed scope/order')
    base=root/'outputs/gated/B004';need(require_review(base/'software/review.json')['status']=='passed','Unreviewed B004 software')
    sw=json.loads((base/'software/result.json').read_text());need(sw['research_sha256']==rh,'Changed protocol')
    verify_files(base,sw['source_sha256']);verify_files(root,sw['repository_source_sha256']);verify_files(root,sw['record_sha256'])
    verify_files(root,sw['inherited_source_sha256'])
    ctx=inputs(root,name);pre=base/f'pre_{name}.json'
    ctx.update(consult=consultation(pre,root,name,ctx,rh),pre=pre,research_sha256=rh,software=sw);return ctx
