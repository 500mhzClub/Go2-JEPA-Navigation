"""CPU operational integration; no new numerical measurements or automatic review."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import pytest
import common
from common import BASE,ORIGINAL,HELPERS,admit,arguments,check_policy,inputs,module,parent_integration,parents,progress_paths,report_remaining
from go2nav.research import atomic_json,digest,require_record

def refuses(call):
    with pytest.raises((ValueError,FileNotFoundError,KeyError,RuntimeError,AssertionError)):call()

def fixture_functions():
    tree=ast.parse((ORIGINAL/'check.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ['fake_review','parent_fixture']]
    assert len(nodes)==2
    scope=dict(Path=Path,atomic_json=atomic_json,digest=digest,json=json)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL/'check.py'),'exec'),scope)
    return scope

began=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);checks=[]
(out/'source').mkdir()
for name in HELPERS:shutil.copy2(BASE/name,out/'source'/name)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
frozen={str(p):digest(p) for label in ['A019','U001','U005','U006'] for p in Path('outputs/gated',label).glob('*.py')}
try:
    record,rh=require_record('U007')
    for name in HELPERS:ast.parse((BASE/name).read_text())
    for path,expected in record['protocol']['source_sha256'].items():assert digest(path)==expected
    transforms=json.loads((BASE/'controlled_changes.json').read_text())
    for name,changes in transforms.items():
        expected=(ORIGINAL/name).read_text()
        for old,new in changes:assert expected.count(old)==1;expected=expected.replace(old,new)
        assert (BASE/name).read_text()==expected
    def function(path,name):return next(ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert function(BASE/'verify_result.py','artifact_checks')==function(ORIGINAL/'verify_result.py','artifact_checks')
    stages,supervision,unlimited,audit=parents()
    for name in ['route','arguments','inputs','consultation']:
        assert Path(getattr(stages,name).__code__.co_filename).resolve()==(ORIGINAL/'stages.py').resolve()
    assert Path(supervision.close_intervals.__code__.co_filename).resolve()==(ORIGINAL/'supervision.py').resolve()
    for name in ['accounting_run','wait_progress']:
        fn=getattr(unlimited,name);fn=getattr(fn,'__wrapped__',fn)
        assert Path(fn.__code__.co_filename).resolve()==Path('outputs/gated/U001/common.py').resolve()
    checks.append('Exact numerical source and artifact-check AST; controlled operational copies; original stages/closure and reviewed U001 functions')
    tree=ast.parse(Path('go2nav/eval/balls_inference.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='budget_remaining']
    assert len(calls)==1
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and n.value is calls[0])
    assert ast.unparse(assignment.targets[0])=="result['budget_remaining_hours']"
    authorized=dict(budget_policy='U001',authorization_sha256=require_record('U001')[1])
    assert json.dumps(dict(budget_remaining_hours=report_remaining(authorized)),allow_nan=False)=='{"budget_remaining_hours": null}'
    for key in authorized:
        bad=dict(authorized);bad[key]='wrong';refuses(lambda:report_remaining(bad))
    checks.append('Only numerical allowance call is result metadata; strict JSONnull retained; wrong policy/authorization refuse')
    for name in stages.ORDER:
        p=stages.route(Path.cwd(),name);old=stages.arguments('/fixture/python',p);new=arguments('/fixture/python',p)
        assert new[2:]==old[2:] and new[:2]==['/fixture/python',str(BASE/'science.py')]
        assert p['science']==Path.cwd()/'outputs/gated/A019/inference'/name
        assert p['operations']==Path.cwd()/'outputs/gated/A019/runs'/name
        assert progress_paths(p,p['operations'])==[p['operations']/'evaluation.log']+[p['science']/f for f in ['source.pt','calibrated.pt.partial','calibrated.pt','result.json']]
    for name in ['l0_n3_s0','o0_n3_s1','l0_n1_s1']:refuses(lambda:stages.route(Path.cwd(),name))
    checks.append('Four exact numerical command recipes and original paths; progress files and scope refusals')

    fixtures=fixture_functions();review=fixtures['fake_review']
    with tempfile.TemporaryDirectory(prefix='u007_') as temporary:
        root=Path(temporary);lp=root/'ledger.json';now=time.time()
        past=dict(id='past',research_id='A019',out='past',start=now-216010,end=now-10)
        atomic_json(lp,[past]);outer=admit(lp,root/'operations','A019')
        old=digest(lp);refuses(lambda:admit(lp,root/'again','A019'));assert digest(lp)==old
        science=root/'science';science.mkdir();atomic_json(science/'manifest.json',dict(run_id='inner'))
        rows=json.loads(lp.read_text())+[dict(id='inner',research_id='A019',out=str(science),start=time.time(),end=None)]
        atomic_json(lp,rows);assert set(supervision.close_intervals(lp,outer['id'],None,science,root,'A019'))=={outer['id'],'inner'}
        assert json.loads(lp.read_text())[0]==past
        checks.append('Closed60hour ledger admits; open interval refuses; original owned nested closure retained')

        data={}
        for split in ['train','val']:
            for name in ['meta.json','labels.npz','frames.npy']:
                path=root/f'outputs/data/balls/n3/{split}/{name}';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic data')
                data[str(path.relative_to(root))]=digest(path)
        review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256=data))
        for label in ['C010','A018','A019','U001','U006','U007']:
            review(root/'outputs/gated'/label/'software',dict(research_id=label,source_sha256={}))
            atomic_json(root/f'docs/research/records/{label}.json',dict(synthetic=True))
        integration=audit.qualified()
        for name in stages.ORDER:
            fixtures['parent_fixture'](root,name)
            af=root/'outputs/gated/A018/runs'/name;sf=root/'outputs/gated/C001'/name;op=root/'outputs/gated/C010/runs'/name
            am=json.loads((af/'manifest.json').read_text());am.update(integration_research_id='U006',integration=integration);atomic_json(af/'manifest.json',am)
            raw=json.loads((af/'result.json').read_text());raw.update(integration_research_id='U006',manifest_sha256=digest(af/'manifest.json'));review(af,raw)
            tr=json.loads((sf/'review.json').read_text());tr['final_audit_review_sha256']=digest(af/'review.json');atomic_json(sf/'review.json',tr)
            om=json.loads((op/'manifest.json').read_text());om.update(integration_research_id='U005',budget_policy='U001');atomic_json(op/'manifest.json',om)
            operation=json.loads((op/'result.json').read_text());operation['manifest_sha256']=digest(op/'manifest.json');review(op,operation)
            assert inputs(stages.route(root,name))
        name=stages.ORDER[0];p=stages.route(root,name);af=root/'outputs/gated/A018/runs'/name;sf=root/'outputs/gated/C001'/name
        for path in [sf/'model.pt',sf/'checkpoint.pt',af/'manifest.json',af/'review.json',root/'outputs/data/balls/n3/train/frames.npy']:
            saved=path.read_bytes();path.write_bytes(b'changed');refuses(lambda:inputs(p));path.write_bytes(saved)
        raw=json.loads((af/'result.json').read_text());am=json.loads((af/'manifest.json').read_text())
        om=json.loads((root/'outputs/gated/C010/runs'/name/'manifest.json').read_text())
        parent_integration(raw,am,om,integration)
        for index,key,value in [(0,'integration_research_id','wrong'),(1,'integration_research_id','wrong'),(1,'integration',{}),(2,'integration_research_id','wrong'),(2,'budget_policy','wrong')]:
            bad=copy.deepcopy([raw,am,om]);bad[index][key]=value;refuses(lambda:parent_integration(*bad,integration))
        checks.append('Four complete original training/data/audit fixtures plus U005/U006 lineage; five raw tamper and five integration corruptions refuse')

        expected=dict(research_sha256=rh,software_review_sha256='synthetic_review',authorization_sha256=require_record('U001')[1],source_sha256={'synthetic.py':'hash'})
        o=dict(research_id='A019',integration_research_id='U007',budget_policy='U001',timeout_type=None)
        om=dict(research_id='A019',integration_research_id='U007',budget_policy='U001',authorization_sha256=expected['authorization_sha256'],integration_research_sha256=rh,
            software_review_sha256=expected['software_review_sha256'],source_sha256=expected['source_sha256'],work_seconds=None,stall_seconds=1800,outer_interval_id='outer',
            preconsultation=dict(u007_record_sha256=rh,u007_software_review_sha256=expected['software_review_sha256'],u001_authorization_sha256=expected['authorization_sha256']))
        sm=dict(research_id='A019',budget_policy='U001',authorization_sha256=expected['authorization_sha256'],budget_hours=None,run_id='inner')
        rows=[dict(id='outer',budget_policy='U001',integration_research_id='U007'),dict(id='inner',budget_policy='U001')]
        result=dict(budget_remaining_hours=None);good=[o,om,sm,rows,result];check_policy(*good,expected)
        faults=[]
        for index,key,value in [(0,'research_id','wrong'),(0,'integration_research_id','wrong'),(1,'integration_research_id','wrong'),(0,'budget_policy','wrong'),(1,'budget_policy','wrong'),(2,'budget_policy','wrong'),
            (1,'authorization_sha256','wrong'),(2,'authorization_sha256','wrong'),(1,'integration_research_sha256','wrong'),(1,'software_review_sha256','wrong'),(1,'source_sha256',{}),
            (1,'work_seconds',1800),(2,'budget_hours',4),(4,'budget_remaining_hours',1),(1,'stall_seconds',1),(0,'timeout_type','stall')]:
            bad=copy.deepcopy(good);bad[index][key]=value;faults.append(bad)
        for key in om['preconsultation']:
            bad=copy.deepcopy(good);bad[1]['preconsultation'][key]='wrong';faults.append(bad)
        for index,key in [(0,'budget_policy'),(1,'budget_policy'),(0,'integration_research_id')]:
            bad=copy.deepcopy(good);bad[3][index][key]='wrong';faults.append(bad)
        for bad in faults:refuses(lambda:check_policy(*bad,expected))
        checks.append(f'Null-time-policy numerical metadata accepted; {len(faults)} research/policy/qualification/consultation/interval corruptions refuse')
        fake=root/'integration';(fake/'software').mkdir(parents=True);(fake/'software/review.json').write_text('synthetic')
        pre=dict(u007_record_sha256=rh,u007_software_review_sha256=digest(fake/'software/review.json'),u001_authorization_sha256=require_record('U001')[1])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common,'BASE',fake);common.extra_consultation(pre)
            for key in pre:
                bad=dict(pre);bad[key]='wrong';refuses(lambda:common.extra_consultation(bad))
                bad=dict(pre);del bad[key];refuses(lambda:common.extra_consultation(bad))
        checks.append('Both numerical and verification consultation adapters require exact U007/U001 identities; three wrong and three missing fields refuse')
    try:inputs(stages.route(Path.cwd(),stages.ORDER[0]))
    except FileNotFoundError as exc:assert Path(exc.filename).resolve()==Path('outputs/gated/C001/l0_n3_s1/review.json').resolve()
    else:raise AssertionError('Actual training missing')
    assert not Path('outputs/gated/A019/inference/l0_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(p)==h for p,h in frozen.items())
    checks.append('Actual missing training refuses before CUDA/output; no Torch or changes to active helpers/shared ledger')
    evidence=dict(research_id='U007',research_sha256=rh,status='awaiting_literature_review',checks=checks,check_groups=len(checks),metadata_corruptions=len(faults),
        elapsed_seconds=time.monotonic()-began,source_sha256={name:digest(BASE/name) for name in HELPERS},controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
        repository_source_sha256={str(p):digest(p) for p in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{label}.json':require_record(label)[1] for label in ['A019','U001','U005','U006','U007']},
        actual_numerical_runs=0,torch_imported=False,limitation='Operational integration only; original numerical and artifact functions unchanged, not rerun. All actual extra-seed numerical gates remain absent.')
    atomic_json(out/'result.json',evidence);print(json.dumps({k:v for k,v in evidence.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='U007',status='technical_blocked',checks=checks,error=repr(exc)));raise
