"""Scoped CPU checks for unchanged later-stage commands and accounting."""
import ast,copy,json,shutil,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import pytest
import common
from common import BASE,ORIGINAL,HELPERS,ROUTES,canonical,parents,input_files,prior_stage,admit,progress_paths,consultation
from go2nav.research import atomic_json,digest,require_record

def refuses(fn):
 with pytest.raises((ValueError,KeyError,FileNotFoundError,AssertionError,RuntimeError)):fn()
def args_for(stage,n,seed,root):
 sf=root/'outputs/gated/C001';suffix=f'_n{n}_s{seed}'
 return SimpleNamespace(stage=stage,n=n,seed=seed,data=str(root/'outputs/data/balls'),checkpoint=str(sf/('slots'+suffix)/'model.pt') if stage=='extract' else None,cache=str(sf/('extract'+suffix)) if stage in ['o0','o1'] else None,parent_review=str(sf/('slots'+suffix if stage=='extract' else 'extract'+suffix)/'review.json') if stage!='slots' else None,python='/fixture/python')
def function_scope(path,names,scope):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope

def fixture_stage(root,name,review):
 kind,n,seed=name.split('_');n=int(n[1:]);seed=int(seed[1:]);sf=root/'outputs/gated/C001'/name;sf.mkdir(parents=True,exist_ok=True)
 for f in ['model.pt','checkpoint.pt','config.json','metrics.jsonl']:(sf/f).write_bytes(b'synthetic '+name.encode())
 if kind=='extract':
  for split in ['train','val']:(sf/f'{split}.npy').write_bytes(b'fake cache')
  atomic_json(sf/'index.json',dict(n=n,seed=seed,splits={s:digest(sf/f'{s}.npy') for s in ['train','val']}))
  atomic_json(sf/'inspection.json',dict(synthetic=True));review(sf,dict(research_id='C001',cache_sha256=digest(sf/'index.json')))
 else:review(sf,dict(research_id='C001',kind=kind,n=n,seed=seed,checkpoint_sha256=digest(sf/'model.pt')))
 op=root/ORIGINAL/'runs'/name;atomic_json(op/'manifest.json',dict(name=name,integration_research_id='U009',budget_policy='U001'))
 artifacts={f.name:digest(f) for f in sf.iterdir() if f.name!='review.json'}
 review(op,dict(research_id='C005',name=name,returncode=0,child_terminal=True,timed_out=False,actions=[],errors=[],manifest_sha256=digest(op/'manifest.json'),child_artifact_sha256=artifacts))
 r=json.loads((sf/'review.json').read_text())
 if kind=='extract':r.update(inspection='inspection.json',inspection_sha256=digest(sf/'inspection.json'))
 else:
  audit=root/'outputs/gated/C008/runs'/name
  review(audit,dict(research_id='C008',name=name,scientific_result_sha256=digest(sf/'result.json'),operational_result_sha256=digest(op/'result.json'),input_sha256={str(sf/'checkpoint.pt'):digest(sf/'checkpoint.pt')},checks={'final_epoch':8 if kind=='slots' else 30}))
  r.update(final_audit_review=str(audit/'review.json'),final_audit_review_sha256=digest(audit/'review.json'))
 atomic_json(sf/'review.json',r)

began=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);(out/'source').mkdir();checks=[]
for f in HELPERS:shutil.copy2(BASE/f,out/'source'/f)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
frozen={str(p):digest(p) for folder in ['C005/v2','U001','U005','C008'] for p in Path('outputs/gated',folder).glob('*.py')}
try:
 _,rh=require_record('U009');original,training,stages,core,unlimited=parents()
 record=json.loads(Path('docs/research/records/U009.json').read_text())
 for p,h in record['protocol']['sources'].items():assert digest(p)==h
 changes=json.loads((BASE/'controlled_changes.json').read_text());s=(ORIGINAL/'run_stage.py').read_text();assert changes['original_sha256']==digest(ORIGINAL/'run_stage.py')
 for c in changes['changes']:assert s.count(c['before'])==1;s=s.replace(c['before'],c['after'])
 assert s==(BASE/'run_stage.py').read_text()
 for f in HELPERS:ast.parse((BASE/f).read_text())
 assert Path(original.close_intervals.__code__.co_filename).resolve()==(ORIGINAL/'supervision.py').resolve()
 assert Path(unlimited.wait_progress.__code__.co_filename).resolve()==Path('outputs/gated/U001/common.py').resolve()
 calls=[n for n in ast.walk(ast.parse(Path('go2nav/training/balls_slots.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='budget_remaining'];assert len(calls)==2
 checks.append('Controlled wrapper transformation; original trainer/model hashes, closure and U001 process/accounting functions retained; two internal budget comparisons')
 for stage,n,seed in ROUTES:
  a=args_for(stage,n,seed,Path.cwd());p=canonical(a);assert p['science']==Path.cwd()/f'outputs/gated/C001/{stage}_n{n}_s{seed}' and p['operations']==Path.cwd()/ORIGINAL/'runs'/p['name']
  def command(path):
   tree=ast.parse(path.read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
   assignment=next(x for x in main.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='argv' for t in x.targets))
   loop=next(x for x in main.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='option')
   scope=dict(args=a,child_out=p['science'],verification=Path.cwd()/'outputs/gated/C003/recovery/review.json',BASE=BASE)
   exec(compile(ast.Module(body=[assignment,loop],type_ignores=[]),str(path),'exec'),scope);return scope['argv']
  old,new=command(ORIGINAL/'run_stage.py'),command(BASE/'run_stage.py');assert old[2:]==new[2:] and new[:2]==[a.python,str(BASE/'science.py')]
  watched=progress_paths(p['science'],p['operations']);assert p['science']/'checkpoint.partial' in watched and p['science']/'train.partial.npy' in watched and len(watched)==12
 for stage,n,seed in [('slots',3,0),('o0',1,0),('slots',1,1),('o1',6,2),('bad',3,1)]:refuses(lambda:canonical(args_for(stage,n,seed,Path.cwd())))
 a=args_for('extract',3,1,Path.cwd());a.checkpoint='/wrong/model.pt';refuses(lambda:canonical(a))
 checks.append('All14 exact original command tails and canonical paths; checkpoint/mmap progress; five unapproved routes and wrong-parent option refuse')
 fixtures=function_scope('outputs/gated/C010/check.py',['fake_review','core_fixture','p26_fixture','training_fixture'],dict(atomic_json=atomic_json,digest=digest,Path=Path));review=fixtures['fake_review']
 extra=function_scope('outputs/gated/U005/check.py',['add_core_operations'],dict(atomic_json=atomic_json,digest=digest,Path=Path,json=json))
 with tempfile.TemporaryDirectory(prefix='u009_') as temp:
  root=Path(temp);lp=root/'ledger.json';now=time.time();past=dict(id='past',research_id='C001',out='past',start=now-216010,end=now-10);atomic_json(lp,[past]);row,remaining,work=admit(lp,root/'operations');assert work is None and all(v is None for v in remaining.values())
  oldhash=digest(lp);refuses(lambda:admit(lp,root/'again'));assert digest(lp)==oldhash
  sf=root/'science';atomic_json(sf/'manifest.json',dict(run_id='inner'));rows=json.loads(lp.read_text())+[dict(id='inner',research_id='C001',out=str(sf),start=time.time(),end=None),dict(id='foreign',research_id='B001',out='foreign',start=time.time(),end=None)];atomic_json(lp,rows)
  assert set(original.close_intervals(lp,row['id'],None,sf,root))=={row['id'],'inner'};closed=json.loads(lp.read_text());assert closed[0]==past and closed[-1]==rows[-1]
  checks.append('Closed60hour history admits with null limits; open interval refuses unchanged; original nested closure preserves foreign interval')
  data={}
  for n in [1,3,6]:
   for split in ['train','val']:
    for f in ['meta.json','labels.npz','frames.npy']:
     p=root/f'outputs/data/balls/n{n}/{split}/{f}';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original fixture bytes');data[str(p.relative_to(root))]=digest(p)
  review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256=data))
  for seed in [0,1,2]:fixtures['core_fixture'](root,seed)
  extra['add_core_operations'](root,fixtures)
  a=args_for('slots',3,1,root);refuses(lambda:input_files(a,root));fixtures['p26_fixture'](root);refuses(lambda:input_files(a,root))
  for name in stages.ORDER:fixtures['training_fixture'](root,name)
  p,files=input_files(a,root);assert files
  for n,seed in [(3,1),(3,2),(1,0),(6,0)]:
   if n==6:
    for phase,rid,kind in [('probes','C001','o1'),('planning','B001','o1'),('binding','C002','slots')]:
     folder=root/f'outputs/gated/{phase}/{kind}_n1_s0';folder.mkdir(parents=True,exist_ok=True);cf=folder/('cases.npz' if phase=='probes' else 'cases.jsonl');cf.write_bytes(b'synthetic');raw=dict(research_id=rid,n=1,kind=kind,seed=0,cases_sha256=digest(cf))
     if phase=='planning':raw['arms']={'o1':{str(h):{'count':1000} for h in [4,6,8]}}
     review(folder,raw,status='measured_negative')
   for stage in (['slots','extract','o0','o1'] if n==3 else ['slots','extract','o1']):
    a=args_for(stage,n,seed,root)
    if stage!='slots':
     parent='slots' if stage=='extract' else 'extract';f=root/f'outputs/gated/C001/{parent}_n{n}_s{seed}/review.json';saved=f.read_bytes();f.unlink();refuses(lambda:input_files(a,root));f.write_bytes(saved)
    p,files=input_files(a,root);assert files;fixture_stage(root,p['name'],review)
  checks.append('All14 complete synthetic predecessor paths; missing P26, same-seed baseline and each immediate stage refuse; full previous seed/count identity required')
  sf=root/'outputs/gated/C001/slots_n3_s1';cp=sf/'checkpoint.pt';saved=cp.read_bytes();cp.write_bytes(b'changed');refuses(lambda:prior_stage(root,'slots_n3_s1',stages));cp.write_bytes(saved)
  f=root/'outputs/gated/C001/extract_n3_s1/train.npy';saved=f.read_bytes();f.write_bytes(b'changed');refuses(lambda:prior_stage(root,'extract_n3_s1',stages));f.write_bytes(saved)
  f=root/'outputs/gated/planning/o0_n3_s1/result.json';saved=f.read_bytes();f.write_bytes(b'changed');refuses(lambda:input_files(args_for('slots',3,2,root),root));f.write_bytes(saved)
  f=root/'outputs/gated/planning/o1_n1_s0/cases.jsonl';saved=f.read_bytes();f.write_bytes(b'changed');refuses(lambda:input_files(args_for('slots',6,0,root),root));f.write_bytes(saved)
  checks.append('Corrupted full checkpoint, cache split, prior seed and prior count cases refuse')
  a=args_for('slots',3,1,root);_,files=input_files(a,root);pre=root/'pre.json'
  review(root/BASE/'software',dict(research_id='U009'))
  value=dict(name='slots_n3_s1',c001_record_sha256=require_record('C001')[1],c005_record_sha256=require_record('C005')[1],u009_record_sha256=rh,u001_authorization_sha256=require_record('U001')[1],software_review_sha256=digest(root/BASE/'software/review.json'),inputs_sha256={str(f):digest(f) for f in files},data_sha256={str(f):digest(root/f) for f in [Path(f'outputs/data/balls/n3/{s}/{name}') for s in ['train','val'] for name in ['meta.json','labels.npz','frames.npy']]},consulted_at='synthetic',sources=[dict(url='synthetic',version='synthetic',sections='synthetic',consequence='not actual consultation')],consequences='synthetic fixture only')
  with pytest.MonkeyPatch.context() as patch:
   patch.setattr(common,'BASE',root/BASE);patch.chdir(root);atomic_json(pre,value)
   # Parent qualification uses the real repository, while fake consultation binds only fixture files.
   patch.setattr(common,'parents',lambda:(original,training,stages,core,unlimited));consultation(pre,a,files)
   for k in ['c005_record_sha256','u009_record_sha256','u001_authorization_sha256','software_review_sha256','inputs_sha256','data_sha256']:
    bad=dict(value);bad[k]=None;atomic_json(pre,bad);refuses(lambda:consultation(pre,a,files))
  checks.append('Exact original plus six integration/data/input consultation fields; each missing value refuses')
 try:input_files(args_for('slots',3,1,Path.cwd()))
 except FileNotFoundError as exc:assert Path(exc.filename).resolve()==Path('outputs/gated/D003/runs/export/review.json').resolve()
 else:raise AssertionError('Actual absent P26 should refuse')
 assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(p)==h for p,h in frozen.items())
 checks.append('Actual absent P26 refuses before Torch/GPU/output; shared ledger and active/historical helpers unchanged')
 files=json.loads(Path('outputs/gated/U005/software/result.json').read_text())['repository_source_sha256']
 records={p:digest(p) for p in [f'docs/research/records/{r}.json' for r in ['U001','U005','U009','C001','C005','C003']]}
 atomic_json(out/'result.json',dict(research_id='U009',research_sha256=rh,status='awaiting_literature_review',check_groups=len(checks),checks=checks,elapsed_seconds=time.monotonic()-began,source_sha256={f:digest(BASE/f) for f in HELPERS},repository_source_sha256=files,record_sha256=records,controlled_changes_sha256=digest(BASE/'controlled_changes.json'),actual_gpu_cases=0,limitation='CPU operational integration only; no actual later stage or C008 compatibility audit. Original scientific sources and owned-process tests inherited unchanged.'))
 print(json.dumps(dict(check_groups=len(checks),checks=checks,elapsed_seconds=time.monotonic()-began),indent=2))
except BaseException as exc:
 atomic_json(out/'result.json',dict(research_id='U009',status='technical_blocked',checks=checks,error=repr(exc),elapsed_seconds=time.monotonic()-began));raise
