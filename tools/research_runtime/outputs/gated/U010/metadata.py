"""C008 provenance integration for U009; full-state computation stays original."""
import importlib.util,json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import digest,require_record,require_review,verify_files
BASE=Path('outputs/gated/U010');ORIGINAL=Path('outputs/gated/C008');TRAINING=Path('outputs/gated/U009')
HELPERS=('metadata.py','audit.py','check.py')
CASES={f'{kind}_n{n}_s{seed}' for n,seed in [(3,1),(3,2),(1,0),(6,0)] for kind in (['slots','o0','o1'] if n==3 else ['slots','o1'])}
def need(value,message):
 if not value:raise ValueError(message)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def training_software():
 need(require_review(TRAINING/'software/review.json')['status']=='passed','U009 software unreviewed')
 sw=json.loads((TRAINING/'software/result.json').read_text());need(sw['research_sha256']==require_record('U009')[1],'U009 protocol changed')
 verify_files(TRAINING,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256']);return sw

def reconcile_sources(root,bound):
 base=Path('outputs/gated/U006');need(require_review(base/'software/review.json')['status']=='passed','Source reconciliation unreviewed')
 sw=json.loads((base/'software/result.json').read_text());verify_files(base,sw['source_sha256'])
 helper=module('u010_reconciliation',base/'metadata.py');helper.qualified();return helper.reconcile_sources(root,bound)

def qualified(check_self=True):
 r,rh=require_record('U010');p=r['protocol']
 need(digest(ORIGINAL/'audit.py')==p['inherited_audit_sha256'] and digest(ORIGINAL/'software/review.json')==p['inherited_review_sha256'],'Original C008 changed')
 need(require_review(ORIGINAL/'software/review.json')['status']=='passed','C008 software unreviewed')
 sw=json.loads((ORIGINAL/'software/result.json').read_text());need(sw['research_sha256']==require_record('C008')[1] and sw['c001_record_sha256']==require_record('C001')[1],'C008 protocol changed')
 verify_files(ORIGINAL,sw['source_sha256']);reconcile_sources(Path.cwd(),sw['repository_source_sha256']);training_software()
 need(digest(TRAINING/'software/review.json')==p['u009_review_sha256'] and digest('outputs/gated/U006/software/review.json')==p['u006_review_sha256'],'Integration parent review changed')
 if check_self:
  need(require_review(BASE/'software/review.json')['status']=='passed','U010 software unreviewed')
  sw=json.loads((BASE/'software/result.json').read_text());need(sw['research_sha256']==rh,'U010 protocol changed')
  verify_files(BASE,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
 return dict(research_id='U010',research_sha256=rh,software_review_sha256=digest(BASE/'software/review.json') if check_self else None,authorization_sha256=require_record('U001')[1])

def case_name(name):need(name in CASES,'Unapproved/completed training case')

def expected_policy(sw):
 return dict(authorization_sha256=require_record('U001')[1],u009_record_sha256=require_record('U009')[1],software_review_sha256=digest(TRAINING/'software/review.json'),source_sha256=sw['source_sha256'],c005_record_sha256=require_record('C005')[1],c001_record_sha256=require_record('C001')[1])

def check_policy(op,om,sm,rows,expected):
 case_name(om['name'])
 need(op['name']==om['name'] and op['research_id']==om['research_id']=='C005' and sm['research_id']=='C001','Wrong original identity')
 need(op['integration_research_id']==om['integration_research_id']=='U009','Wrong training integration')
 need(op['budget_policy']==om['budget_policy']==sm['budget_policy']=='U001','Wrong time policy')
 need(om['authorization_sha256']==sm['authorization_sha256']==expected['authorization_sha256'],'Wrong authorization')
 need(om['integration_research_sha256']==expected['u009_record_sha256'],'Wrong U009 protocol')
 need(om['research_sha256']==expected['c005_record_sha256'] and sm['research_sha256']==expected['c001_record_sha256'],'Wrong original protocol')
 need(om['software_review_sha256']==expected['software_review_sha256'] and om['source_sha256']==expected['source_sha256'],'Wrong helper qualification')
 need(om['work_seconds'] is None and om['remaining_at_admission']==dict(global_seconds=None,bc_seconds=None) and sm['budget_hours'] is None,'Superseded time cap present')
 need(om['cleanup_reserve_seconds']==45 and op['timeout_type'] is None and op['status']=='awaiting_literature_review','Wrong terminal time policy')
 pre=om['preconsultation']
 for field in ['u009_record_sha256','u001_authorization_sha256','software_review_sha256','c001_record_sha256','c005_record_sha256']:
  key='authorization_sha256' if field=='u001_authorization_sha256' else field
  need(pre[field]==expected[key],'Wrong training consultation: '+field)
 selected=[r for r in rows if r['id'] in {om['outer_interval_id'],sm['run_id']}]
 need(len(selected)==2 and all(r['budget_policy']=='U001' for r in selected),'Wrong owned interval policy')
 outer=next(r for r in selected if r['id']==om['outer_interval_id']);need(outer['integration_research_id']=='U009','Wrong outer integration')

def check_prerequisites(pre,prerequisites):
 need(pre['inputs_sha256']=={str(f):digest(f) for f in prerequisites},'Changed/incomplete training prerequisite inventory')

def training_metadata(runtime,op,om,sm,rows):
 sw=training_software();check_policy(op,om,sm,rows,expected_policy(sw))
 kind,n,seed=om['name'].split('_');n=int(n[1:]);seed=int(seed[1:]);runtime=Path(runtime)
 cache=runtime/f'outputs/gated/C001/extract_n{n}_s{seed}'
 args=SimpleNamespace(stage=kind,n=n,seed=seed,data=str(runtime/'outputs/data/balls'),checkpoint=None,cache=str(cache) if kind!='slots' else None,parent_review=str(cache/'review.json') if kind!='slots' else None)
 helper=module('u010_training',TRAINING/'common.py');_,prerequisites=helper.input_files(args,runtime)
 check_prerequisites(om['preconsultation'],prerequisites)
 files=set(prerequisites)|{Path(f) for f in sm['inputs']}
 if kind!='slots':
  index=json.loads((cache/'index.json').read_text());files|={cache/'index.json',cache/'train.npy',cache/'val.npy',Path(index['encoder'])}
 for label in ['C008','U009','U010']:
  base=Path('outputs/gated')/label;files|={Path(f'docs/research/records/{label}.json'),base/'software/result.json',base/'software/review.json'}
  files|={base/f for f in json.loads((base/'software/result.json').read_text())['source_sha256']}
 files|={Path('docs/research/records/U001.json')};return files

def check_consultation(value,expected):
 need(all(value.get(k)==v for k,v in expected.items()),'Stale or incomplete actual audit consultation')
 need(value.get('consulted_at') and value.get('sources') and value.get('consequences'),'Missing substantive consultation')
 need(all(all(s.get(k) for k in ['url','version','sections','consequence']) for s in value['sources']),'Incomplete literature source')

def consultation(path,name,files,science,operation):
 value=json.loads(Path(path).read_text())
 expected=dict(name=name,research_sha256=require_record('C008')[1],c001_record_sha256=require_record('C001')[1],u010_record_sha256=require_record('U010')[1],u010_software_review_sha256=digest(BASE/'software/review.json'),u009_software_review_sha256=digest(TRAINING/'software/review.json'),u001_authorization_sha256=require_record('U001')[1],scientific_result_sha256=digest(science/'result.json'),operational_result_sha256=digest(operation/'result.json'),inputs_sha256={str(f.resolve()):digest(f) for f in files})
 check_consultation(value,expected);return value
