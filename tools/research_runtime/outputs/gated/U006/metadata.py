"""Admit U005 operational metadata while preserving the original A018 contract."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from go2nav.research import digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/U006')
ORIGINAL = Path('outputs/gated/A018')
TRAINING = Path('outputs/gated/U005')
HELPERS = ('metadata.py','audit.py','check.py')

def need(condition, message):
    if not condition: raise ValueError(message)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def training_software():
    need(require_review(TRAINING/'software/review.json')['status'] == 'passed', 'U005 software unreviewed')
    sw = json.loads((TRAINING/'software/result.json').read_text())
    need(sw['research_sha256'] == require_record('U005')[1], 'U005 protocol changed')
    verify_files(TRAINING,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    return sw

def reconcile_sources(root, bound):
    parent = module('u006_training',TRAINING/'common.py')
    _, _, core, _, _ = parent.parents()
    numerical = json.loads(Path('outputs/gated/C014/o0_n3_s0/result.json').read_text())
    return core.reconcile(root,bound,numerical['implementation_sha256'])

def qualified(*, check_self=True):
    record, rh = require_record('U006');p=record['protocol']
    need(require_review(ORIGINAL/'software/review.json')['status'] == 'passed', 'A018 software unreviewed')
    for file,key in [('audit.py','inherited_audit_sha256'),('state_contract.py','inherited_state_contract_sha256'),('software/review.json','inherited_review_sha256')]:
        need(digest(ORIGINAL/file)==p[key], 'Original A018 changed: '+file)
    sw=json.loads((ORIGINAL/'software/result.json').read_text())
    need(sw['research_sha256']==require_record('A018')[1], 'A018 protocol changed')
    verify_files(ORIGINAL,sw['source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
    reconcile_sources(Path.cwd(),sw['repository_source_sha256']);training_software()
    need(digest(TRAINING/'software/review.json')==p['u005_review_sha256'], 'U005 review changed')
    if check_self:
        need(require_review(BASE/'software/review.json')['status']=='passed', 'U006 software unreviewed')
        current=json.loads((BASE/'software/result.json').read_text())
        need(current['research_sha256']==rh, 'U006 protocol changed')
        verify_files(BASE,current['source_sha256']);verify_files(Path.cwd(),current['repository_source_sha256'])
    return dict(research_id='U006',research_sha256=rh,
        software_review_sha256=digest(BASE/'software/review.json') if check_self else None,
        authorization_sha256=require_record('U001')[1])

def expected_metadata(sw):
    return dict(authorization_sha256=require_record('U001')[1],u005_record_sha256=require_record('U005')[1],
        software_review_sha256=digest(TRAINING/'software/review.json'),source_sha256=sw['source_sha256'],
        research_sha256=require_record('C010')[1],scientific_research_sha256=require_record('C001')[1])

def check_policy(op,om,sm,rows,expected):
    need(op['integration_research_id']==om['integration_research_id']=='U005', 'Wrong training integration')
    need(op['budget_policy']==om['budget_policy']==sm['budget_policy']=='U001', 'Wrong time policy')
    need(om['authorization_sha256']==sm['authorization_sha256']==expected['authorization_sha256'], 'Wrong time authorization')
    need(om['integration_research_sha256']==expected['u005_record_sha256'], 'Wrong U005 protocol')
    need(om['software_review_sha256']==expected['software_review_sha256'], 'Wrong U005 software review')
    need(om['source_sha256']==expected['source_sha256'], 'Wrong U005 helper inventory')
    need(om['research_sha256']==expected['research_sha256'] and
         om['scientific_research_sha256']==sm['research_sha256']==expected['scientific_research_sha256'], 'Wrong original protocol')
    need(om['work_seconds'] is None and sm['budget_hours'] is None and om['stall_seconds']==1800, 'Wrong execution time policy')
    need(op['timeout_type'] is None, 'Unexpected stall termination')
    pre=om['preconsultation']
    need(pre['u005_record_sha256']==expected['u005_record_sha256'] and
         pre['u001_authorization_sha256']==expected['authorization_sha256'] and
         pre['software_review_sha256']==expected['software_review_sha256'], 'Wrong training consultation identity')
    owned={om['outer_interval_id'],sm['run_id']}
    selected=[row for row in rows if row['id'] in owned]
    need(len(selected)==2 and all(row['budget_policy']=='U001' for row in selected), 'Wrong owned interval policy')
    outer=next(row for row in selected if row['id']==om['outer_interval_id'])
    need(outer['integration_research_id']=='U005', 'Wrong outer integration')

def training_metadata(root,op,om,sm,rows):
    sw=training_software();check_policy(op,om,sm,rows,expected_metadata(sw))
    files=set()
    for label in ['A018','C010','U001','U003','U005']:
        base=root/'outputs/gated'/label
        files|={base/'software/review.json',base/'software/result.json',root/f'docs/research/records/{label}.json'}
        source=json.loads((base/'software/result.json').read_text())['source_sha256']
        verify_files(base,source);files|={base/name for name in source}
    files|={root/BASE/'software/review.json',root/BASE/'software/result.json',root/'docs/research/records/U006.json'}
    files|={root/BASE/name for name in HELPERS}
    return files

def check_consultation(value):
    need(value.get('u006_record_sha256')==require_record('U006')[1], 'Wrong audit integration consultation')
    need(value.get('u006_software_review_sha256')==digest(BASE/'software/review.json'), 'Wrong audit integration software')
    need(value.get('u001_authorization_sha256')==require_record('U001')[1], 'Wrong audit authorization')
