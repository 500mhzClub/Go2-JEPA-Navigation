"""Approved later slot stages under U001, preserving original computation."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys,time,uuid
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
BASE=Path('outputs/gated/U009');ORIGINAL=Path('outputs/gated/C005/v2')
HELPERS=('common.py','science.py','run_stage.py','check.py')
ROUTES={(stage,n,seed) for n,seed in [(3,1),(3,2),(1,0),(6,0)] for stage in (['slots','extract','o0','o1'] if n==3 else ['slots','extract','o1'])}
def need(value,message):
 if not value:raise ValueError(message)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def parents():
 need(require_review(ORIGINAL/'software/review.json')['status']=='passed','Original supervision unreviewed')
 sw=json.loads((ORIGINAL/'software/result.json').read_text());need(sw['research_sha256']==require_record('C005')[1],'Original supervision protocol changed');verify_files(ORIGINAL,sw['source_sha256'])
 base=Path('outputs/gated/U005');need(require_review(base/'software/review.json')['status']=='passed','Training integration unreviewed')
 sw=json.loads((base/'software/result.json').read_text());verify_files(base,sw['source_sha256']);verify_files(Path.cwd(),sw['repository_source_sha256']);verify_files(Path.cwd(),sw['record_sha256'])
 training=module('u009_training',base/'common.py');stages,_,core,unlimited,_=training.parents()
 return module('u009_original_supervision',ORIGINAL/'supervision.py'),training,stages,core,unlimited

def canonical(args,root=None):
 root=Path.cwd() if root is None else Path(root)
 need((args.stage,args.n,args.seed) in ROUTES,'Unapproved slot stage/count/seed')
 name=f'{args.stage}_n{args.n}_s{args.seed}';folder=root/'outputs/gated/C001';suffix=f'_n{args.n}_s{args.seed}'
 expected=dict(checkpoint=None,cache=None,parent_review=None)
 if args.stage=='extract':expected.update(checkpoint=folder/('slots'+suffix)/'model.pt',parent_review=folder/('slots'+suffix)/'review.json')
 if args.stage in ['o0','o1']:expected.update(cache=folder/('extract'+suffix),parent_review=folder/('extract'+suffix)/'review.json')
 need((root/args.data).resolve()==(root/'outputs/data/balls').resolve(),'Original data route required')
 for key,value in expected.items():
  actual=getattr(args,key)
  need(actual is None if value is None else actual is not None and (root/actual).resolve()==value.resolve(),'Wrong stage parent option: '+key)
 return dict(root=root,name=name,science=folder/name,operations=root/ORIGINAL/'runs'/name,data=root/'outputs/data/balls')

def prior_stage(root,name,stages):
 sf=root/'outputs/gated/C001'/name;review,raw,files=stages.reviewed(sf,'C001')
 op=root/ORIGINAL/'runs'/name;_,operation,of=stages.reviewed(op,'C005');om=json.loads((op/'manifest.json').read_text())
 need(operation['name']==om['name']==name and operation['returncode']==0 and operation['child_terminal'] and not any(operation[k] for k in ['timed_out','actions','errors']),'Previous stage incomplete')
 need(operation['manifest_sha256']==digest(op/'manifest.json') and om['integration_research_id']=='U009' and om['budget_policy']=='U001','Wrong prior operation')
 verify_files(sf,operation['child_artifact_sha256']);need(operation['child_artifact_sha256'].get('result.json')==digest(sf/'result.json'),'Prior scientific result not bound')
 files|=of|{op/'manifest.json'}|{sf/f for f in operation['child_artifact_sha256']}
 kind,n,seed=name.split('_');n=int(n[1:]);seed=int(seed[1:])
 if kind=='extract':
  index=sf/'index.json';need(raw['cache_sha256']==digest(index),'Wrong cache digest');ci=json.loads(index.read_text())
  need((ci['n'],ci['seed'])==(n,seed),'Wrong cache identity');verify_files(sf,{f'{s}.npy':h for s,h in ci['splits'].items()})
  need(set(ci['splits'])=={'train','val'},'Wrong cache splits');inspection=sf/review['inspection'];need(digest(inspection)==review['inspection_sha256'],'Cache inspection changed')
  files|={index,inspection,sf/'train.npy',sf/'val.npy'}
 else:
  need((raw['kind'],raw['n'],raw['seed'])==(kind,n,seed) and raw['checkpoint_sha256']==digest(sf/'model.pt'),'Wrong selected training identity')
  audit=root/review['final_audit_review'];need(digest(audit)==review['final_audit_review_sha256'],'Full-state audit review changed');ar=require_review(audit);av=json.loads((audit.parent/ar['result']).read_text())
  need(ar['status']=='passed' and av['name']==name and av['scientific_result_sha256']==digest(sf/'result.json') and av['operational_result_sha256']==digest(op/'result.json'),'Wrong full-state audit')
  verify_files(root,av['input_sha256']);need(av['checks']['final_epoch']==(8 if kind=='slots' else 30),'Incomplete training epochs')
  files|={audit,audit.parent/ar['result']}|{root/f for f in av['input_sha256']}
 return files

def count_complete(root,n,stages):
 files=set()
 for phase,rid in [('probes','C001'),('planning','B001')]:
  p=root/f'outputs/gated/{phase}/o1_n{n}_s0';r=require_review(p/'review.json');raw=json.loads((p/r['result']).read_text());identity=r.get('run_identity',{})
  need(r['status'] in ['passed','measured_negative'] and raw.get('status') not in ['budget_incomplete','technical_blocked'],'Incomplete previous count')
  need(tuple(raw.get(k,identity.get(k)) for k in ['n','kind','seed'])==(n,'o1',0),'Wrong previous count identity')
  cases=p/('cases.jsonl' if phase=='planning' else 'cases.npz');need(raw['cases_sha256']==digest(cases),'Changed previous cases')
  if phase=='planning':need(all(raw['arms']['o1'][str(h)]['count']==1000 for h in [4,6,8]),'Partial previous planning')
  files|={p/'review.json',p/r['result'],cases}
 p=root/f'outputs/gated/binding/slots_n{n}_s0';_,raw,bound=stages.reviewed(p,'C002',('passed','measured_negative'))
 need((raw['n'],raw['kind'],raw['seed'])==(n,'slots',0) and digest(p/'cases.jsonl')==raw['cases_sha256'],'Wrong previous count binding')
 return files|bound|{p/'cases.jsonl'}

def input_files(args,root=None):
 p=canonical(args,root);root=p['root'];_,training,stages,core,_=parents()
 files=stages.p26_reviews(root)|training.core_files(root,0,stages,core)
 for seed in (range(1,args.seed) if args.n==3 else [1,2]):files|=stages.core_reviews(root,seed)
 if args.n==3:
  for kind in ['l0','l1']:files|=stages.prior_training(root,f'{kind}_n3_s{args.seed}')
 if args.n==6:files|=count_complete(root,1,stages)
 previous=[] if args.stage=='slots' else ['slots']
 if args.stage in ['o0','o1']:previous+=['extract']
 if args.stage=='o1' and args.n==3:previous+=['o0']
 for stage in previous:files|=prior_stage(root,f'{stage}_n{args.n}_s{args.seed}',stages)
 _,raw,bound=stages.reviewed(root/'outputs/gated/A014/preparation','A014')
 names=[f'outputs/data/balls/n{args.n}/{s}/{f}' for s in ['train','val'] for f in ['meta.json','labels.npz','frames.npy']]
 verify_files(root,{f:raw['data_sha256'][f] for f in names});files|=bound|{root/f for f in names}
 return p,sorted({f.resolve() for f in files})

def consultation(path,args,files):
 original,*_=parents();name=f'{args.stage}_n{args.n}_s{args.seed}'
 pre=original.consultation(path,name,require_record('C001')[1])
 expected=dict(c005_record_sha256=require_record('C005')[1],u009_record_sha256=require_record('U009')[1],u001_authorization_sha256=require_record('U001')[1],software_review_sha256=digest(BASE/'software/review.json'),inputs_sha256={str(f):digest(f) for f in files})
 expected['data_sha256']={str(f):digest(f) for f in [Path(f'outputs/data/balls/n{args.n}/{s}/{name}') for s in ['train','val'] for name in ['meta.json','labels.npz','frames.npy']]}
 need(all(pre.get(k)==v for k,v in expected.items()),'Stale or incomplete slot-stage consultation')
 return pre

def admit(ledger,out,reserve=45):
 with ledger.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(ledger.read_text());need(all(r['end'] is not None for r in rows),'Open GPU interval')
  row=dict(id=uuid.uuid4().hex,research_id='C005',integration_research_id='U009',budget_policy='U001',out=str(out),start=time.time(),end=None)
  rows.append(row);atomic_json(ledger,rows)
 return row,dict(global_seconds=None,bc_seconds=None),None

def progress_paths(science,out):
 return [out/'training.log']+[science/f for f in ['metrics.jsonl','checkpoint.pt','checkpoint.partial','model.pt','model.partial','train.partial.npy','val.partial.npy','train.npy','val.npy','index.json','result.json']]
