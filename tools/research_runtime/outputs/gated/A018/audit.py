"""Audit newly completed baseline seeds on CPU; never approve a review."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

BASE=Path('outputs/gated/A018')
NAMES=['l0_n3_s1','l1_n3_s1','l0_n3_s2','l1_n3_s2']
REQUIRED=['checkpoint.pt','model.pt','config.json','metrics.jsonl','lineage.json','result.json','manifest.json']


def need(condition,message):
    if not condition:raise ValueError(message)


def terminal_checks(op,om,sm,rows,science,operations):
    need(op['child_terminal'] and op['returncode']==0 and not op['timed_out'] and not op['actions']
         and not op['errors'] and op['inputs_unchanged'],'Incomplete/unsuccessful operation')
    need(op['raw_status']=='awaiting_literature_review','Incomplete science')
    need(op['research_id']==om['research_id']=='C010' and op['scientific_research_id']=='C001','Wrong operation identity')
    need(op['stage']==om['stage']=='train' and op['name']==om['name']==science.name,'Wrong stage/run')
    try:need(psutil.Process(om['pid']).create_time()!=om['creation_time'],'Exact trainer still live')
    except psutil.NoSuchProcess:pass
    ids=[om['outer_interval_id'],sm['run_id']];closed=set(op['closed_interval_ids'])
    need(len(set(ids))==2 and ids[0] in closed and closed.issubset(ids),'Wrong closure evidence')
    selected=[r for r in rows if r['id'] in ids]
    need(len(selected)==2 and all(r['end'] is not None and r['end']>=r['start'] for r in selected),'Open/ambiguous/invalid owned intervals')
    for ident,path,marker in [(ids[0],operations,'C010'),(ids[1],science,None)]:
        r=next(r for r in selected if r['id']==ident)
        need(r['research_id']=='C001' and r.get('supervisor_research_id')==marker and
             (Path(om['cwd'])/r['out']).resolve()==path.resolve(),'Wrong ledger owner/path')


def source_checks(audit_root,runtime_root,bound):
    verify_files(runtime_root,bound)
    computational={f:h for f,h in bound.items() if f.startswith(('go2nav/','configs/'))}
    need(any(f.startswith('go2nav/') for f in computational),'Missing computational source inventory')
    verify_files(audit_root,computational)


def admission(name):
    need(name in NAMES,'Unplanned fresh baseline')
    root=Path.cwd();science=root/'outputs/gated/C001'/name;operations=root/'outputs/gated/C010/runs'/name
    op=json.loads((operations/'result.json').read_text());om=json.loads((operations/'manifest.json').read_text())
    sm=json.loads((science/'manifest.json').read_text())
    terminal_checks(op,om,sm,json.loads((root/'outputs/gated/gpu_intervals.json').read_text()),science,operations)
    need(op['manifest_sha256']==digest(operations/'manifest.json'),'Changed operation manifest')
    need(set(REQUIRED).issubset(op['child_artifact_sha256']),'Missing complete training artifacts')
    verify_files(science,op['child_artifact_sha256'])
    runtime=Path(om['cwd'])
    need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==om['source_commit']==sm['code_commit'],
         'Pinned runtime commit changed before audit')
    source_checks(root,runtime,sm['source_sha256'])
    verify_files(root,sm['inputs']);verify_files(root,om['preconsultation']['inputs_sha256'])
    need(sm['research_id']=='C001' and om['scientific_research_id']=='C001','Wrong science identity')
    cb=root/'outputs/gated/C010';need(require_review(cb/'software/review.json')['status']=='passed','C010 software not passed')
    cs=json.loads((cb/'software/result.json').read_text())
    need(digest(cb/'software/review.json')==om['software_review_sha256'],'C010 review changed')
    need(cs['research_sha256']==om['research_sha256']==digest(root/'docs/research/records/C010.json'),'C010 protocol changed')
    need(sm['research_sha256']==om['scientific_research_sha256']==digest(root/'docs/research/records/C001.json'),'C001 protocol changed')
    verify_files(cb,cs['source_sha256']);verify_files(root,cs['repository_source_sha256'])
    spec=importlib.util.spec_from_file_location('a018_c010_stages',cb/'stages.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=set(module.original_data(root))
    files|={science/f for f in REQUIRED}|{operations/'result.json',operations/'manifest.json',cb/'software/review.json',cb/'software/result.json'}
    return science,operations,op,om,sm,files


def consultation(path,name,rh,science,operations):
    value=json.loads(Path(path).read_text())
    expected=dict(name=name,research_sha256=rh,
        operation_result_sha256=digest(operations/'result.json'),scientific_result_sha256=digest(science/'result.json'),
        checkpoint_sha256=digest(science/'checkpoint.pt'),model_sha256=digest(science/'model.pt'),
        software_review_sha256=digest(BASE/'software/review.json'))
    need(all(value.get(k)==v for k,v in expected.items()),'Wrong/stale actual audit consultation')
    need(value.get('consulted_at') and value.get('sources') and value.get('consequences'),'Missing substantive consultation')
    need(all(all(s.get(k) for k in ['url','version','sections','consequence']) for s in value['sources']),'Incomplete source consultation')
    return value


def run(name,pre):
    need(all(os.environ.get(k)=='' for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']),'CPU-only environment required')
    _,rh=require_record('A018');need(require_review(BASE/'software/review.json')['status']=='passed','Audit software not passed')
    sw=json.loads((BASE/'software/result.json').read_text());need(sw['research_sha256']==rh,'Audit protocol changed')
    verify_files(BASE,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    science,operations,op,om,sm,files=admission(name)
    prevalue=consultation(pre,name,rh,science,operations);files.add(Path(pre))
    out=BASE/'runs'/name;out.mkdir(parents=True,exist_ok=False);snapshot=out/'snapshot';snapshot.mkdir()
    started=time.monotonic();manifest=dict(research_id='A018',name=name,research_sha256=rh,
        software_review_sha256=digest(BASE/'software/review.json'),preconsultation=prevalue,
        input_sha256={str(f):digest(f) for f in files},source_commit=om['source_commit'],snapshot_sha256={})
    atomic_json(out/'manifest.json',manifest)
    try:
        for file in REQUIRED:
            with (science/file).open('rb') as source,(snapshot/file).open('xb') as target:shutil.copyfileobj(source,target)
            manifest['snapshot_sha256'][file]=digest(snapshot/file)
        verify_files(science,manifest['snapshot_sha256']);atomic_json(out/'manifest.json',manifest)
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from state_contract import same_json,state_checks
        full=torch.load(snapshot/'checkpoint.pt',map_location='cpu',weights_only=False)
        final=torch.load(snapshot/'model.pt',map_location='cpu',weights_only=False)
        cfg=json.loads((snapshot/'config.json').read_text());raw=json.loads((snapshot/'result.json').read_text())
        text=(snapshot/'metrics.jsonl').read_text();need(text.endswith('\n'),'Incomplete trailing log row')
        rows=[json.loads(line) for line in text.splitlines()]
        need(name==f'{cfg["kind"]}_n{cfg["n"]}_s{cfg["seed"]}','Wrong saved seed/run')
        need(raw['checkpoint_sha256']==manifest['snapshot_sha256']['model.pt'],'Wrong raw model digest')
        need(same_json(sm['config'],{k:cfg[k] for k in ['kind','n','seed','train','model']}),'Training manifest config differs')
        lineage=json.loads((snapshot/'lineage.json').read_text())
        need(lineage['component']=='balls_'+cfg['kind'] and lineage['seed']==cfg['seed'] and same_json(lineage['config'],cfg),'Wrong training lineage')
        need(not lineage['parents'] and lineage['data']==sm['inputs'],'Wrong lineage parents/data')
        need(lineage['checkpoint']==dict(file='model.pt',sha256=raw['checkpoint_sha256']),'Wrong lineage checkpoint')
        need(lineage['code']==dict(commit=sm['code_commit'],uncommitted_changes=False),'Wrong/dirty lineage source')
        shape=np.load('outputs/data/balls/n3/train/frames.npy',mmap_mode='r').shape
        checks=state_checks(full,final,cfg,rows,raw,shape)
        verify_files(science,manifest['snapshot_sha256']);verify_files(Path.cwd(),manifest['input_sha256'])
        verify_files(Path.cwd(),om['preconsultation']['inputs_sha256']);source_checks(Path.cwd(),Path(om['cwd']),sm['source_sha256'])
        result=dict(research_id='A018',name=name,status='awaiting_literature_review',**checks,
            checkpoint_sha256=digest(snapshot/'checkpoint.pt'),model_sha256=digest(snapshot/'model.pt'),
            manifest_sha256=digest(out/'manifest.json'),elapsed_seconds=time.monotonic()-started,cpu_only=True,
            limitation='Artifact consistency only; no forward/gradient/update or calibrated inference. GPU RNG bytes are not device-restored.',
            next_step='Substantively review actual A018, C001 training and C010 operations separately before any inference or successor.')
        atomic_json(out/'result.json',result);return result
    except BaseException as exc:
        atomic_json(out/'result.json',dict(research_id='A018',name=name,status='technical_blocked',error=repr(exc),manifest_sha256=digest(out/'manifest.json')))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--preconsultation',required=True);a=p.parse_args()
    print(json.dumps(run(a.run,a.preconsultation)),flush=True)
