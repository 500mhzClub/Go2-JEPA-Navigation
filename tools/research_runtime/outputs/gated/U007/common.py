"""A019 operational integration; numerical and artifact computations unchanged."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

BASE=Path('outputs/gated/U007')
ORIGINAL=Path('outputs/gated/A019')
HELPERS=('common.py','science.py','run_check.py','verify_result.py','check.py')

def need(value,message):
    if not value:raise ValueError(message)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def parents():
    need(require_review(ORIGINAL/'software/review.json')['status']=='passed','A019 software unreviewed')
    sw=json.loads((ORIGINAL/'software/result.json').read_text())
    need(sw['research_sha256']==require_record('A019')[1],'A019 protocol changed')
    verify_files(ORIGINAL,sw['source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    audit=module('u007_audit_metadata',Path('outputs/gated/U006/metadata.py'));audit.qualified()
    audit.reconcile_sources(Path.cwd(),sw['repository_source_sha256'])
    training=module('u007_training',Path('outputs/gated/U005/common.py'))
    _,_,_,unlimited,_=training.parents()
    return module('u007_original_stages',ORIGINAL/'stages.py'),module('u007_original_supervision',ORIGINAL/'supervision.py'),unlimited,audit

def qualified():
    parents();_,rh=require_record('U007')
    need(require_review(BASE/'software/review.json')['status']=='passed','U007 software unreviewed')
    sw=json.loads((BASE/'software/result.json').read_text())
    need(sw['research_sha256']==rh,'U007 protocol changed')
    verify_files(BASE,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    return dict(research_id='U007',research_sha256=rh,software_review_sha256=digest(BASE/'software/review.json'),authorization_sha256=require_record('U001')[1])

def reconcile_sources(root,bound):
    _,_,_,audit=parents();return audit.reconcile_sources(root,bound)

def route(root,name):
    stages,_,_,_=parents();return stages.route(root,name)

def parent_integration(raw,am,om,expected):
    need(raw['integration_research_id']==am['integration_research_id']=='U006','Wrong full-state audit integration')
    need(am['integration']==expected,'Wrong full-state audit qualification')
    need(om['integration_research_id']=='U005' and om['budget_policy']=='U001','Wrong training integration')

def inputs(p):
    stages,_,_,audit=parents();files=set(stages.inputs(p));root=p['root'];name=p['name']
    folder=root/'outputs/gated/A018/runs'/name
    raw=json.loads((folder/'result.json').read_text());am=json.loads((folder/'manifest.json').read_text())
    om=json.loads((root/'outputs/gated/C010/runs'/name/'manifest.json').read_text())
    parent_integration(raw,am,om,audit.qualified())
    for label in ['A019','U001','U006','U007']:
        base=root/'outputs/gated'/label
        files|={base/'software/result.json',base/'software/review.json',root/f'docs/research/records/{label}.json'}
        sw=json.loads((base/'software/result.json').read_text())
        files|={base/name for name in sw['source_sha256']}
    need(all(path.is_file() for path in files),'Required input absent')
    return sorted({path.resolve() for path in files})

def arguments(python,p):
    stages,_,_,_=parents();original=stages.arguments(python,p)
    return [str(python),str(BASE/'science.py'),original[2],*original[3:]]

def extra_consultation(pre):
    expected=dict(u007_record_sha256=require_record('U007')[1],u007_software_review_sha256=digest(BASE/'software/review.json'),u001_authorization_sha256=require_record('U001')[1])
    need(all(pre.get(k)==v for k,v in expected.items()),'Wrong integration consultation')

def consultation(path,p,files,rh,sh):
    stages,_,_,_=parents();pre=stages.consultation(path,p,files,rh,sh);extra_consultation(pre);return pre

def report_remaining(manifest):
    # This evaluator's sole accessor call writes metadata, not a stopping rule.
    need(manifest['budget_policy']=='U001' and manifest['authorization_sha256']==require_record('U001')[1],'Wrong unlimited authorization')
    return None

def admit(ledger,out,science):
    need(science=='A019','Wrong numerical scope')
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text())
        need(all(row['end'] is not None for row in rows),'Open GPU interval')
        row=dict(id=uuid.uuid4().hex,research_id='A019',supervisor_research_id='A019',integration_research_id='U007',budget_policy='U001',out=str(out),start=time.time(),end=None)
        rows.append(row);atomic_json(ledger,rows)
    return row

def progress_paths(p,out):
    return [out/'evaluation.log']+[p['science']/name for name in ['source.pt','calibrated.pt.partial','calibrated.pt','result.json']]

def check_policy(o,om,sm,rows,result,expected):
    need(o['research_id']==om['research_id']==sm['research_id']=='A019','Wrong numerical research identity')
    need(o['integration_research_id']==om['integration_research_id']=='U007','Wrong numerical integration')
    need(o['budget_policy']==om['budget_policy']==sm['budget_policy']=='U001','Wrong numerical time policy')
    need(om['authorization_sha256']==sm['authorization_sha256']==expected['authorization_sha256'],'Wrong numerical authorization')
    need(om['integration_research_sha256']==expected['research_sha256'] and om['software_review_sha256']==expected['software_review_sha256'],'Wrong numerical qualification')
    need(om['source_sha256']==expected['source_sha256'],'Wrong numerical helper inventory')
    need(om['work_seconds'] is None and sm['budget_hours'] is None and result['budget_remaining_hours'] is None,'Wrong numerical budget metadata')
    need(om['stall_seconds']==1800 and o['timeout_type'] is None,'Numerical operation stalled')
    pre=om['preconsultation']
    need(pre['u007_record_sha256']==expected['research_sha256'] and pre['u007_software_review_sha256']==expected['software_review_sha256'] and pre['u001_authorization_sha256']==expected['authorization_sha256'],'Wrong numerical consultation')
    selected=[row for row in rows if row['id'] in {om['outer_interval_id'],sm['run_id']}]
    need(len(selected)==2 and all(row['budget_policy']=='U001' for row in selected),'Wrong numerical interval policy')
    outer=next(row for row in selected if row['id']==om['outer_interval_id'])
    need(outer['integration_research_id']=='U007','Wrong outer integration')

def operation_metadata(o,om,sm,rows,science):
    expected=qualified();expected['source_sha256']=json.loads((BASE/'software/result.json').read_text())['source_sha256']
    check_policy(o,om,sm,rows,json.loads((science/'result.json').read_text()),expected)
    runtime=Path(om['cwd'])
    need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==om['source_commit']==sm['code_commit'],'Pinned numerical runtime changed')
    source=module('u007_original_source_check',Path('outputs/gated/A018/audit.py'))
    source.source_checks(Path.cwd(),runtime,sm['source_sha256'])
