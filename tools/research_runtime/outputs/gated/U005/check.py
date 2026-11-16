"""CPU integration checks; synthetic reviews never approve actual experiments."""
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
from common import BASE, ORIGINAL, HELPERS, admit, arguments, consultation, core_files, inputs, parents, progress_paths
from go2nav.research import atomic_json, digest, require_record

def refuses(call):
    with pytest.raises((ValueError, FileNotFoundError, KeyError, RuntimeError)): call()

def original_fixtures():
    tree = ast.parse((ORIGINAL/'check.py').read_text())
    names = ['fake_review','core_fixture','p26_fixture','training_fixture']
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(functions) == 4
    scope = dict(atomic_json=atomic_json, digest=digest, Path=Path)
    exec(compile(ast.Module(body=functions,type_ignores=[]), str(ORIGINAL/'check.py'), 'exec'),scope)
    return scope

def add_core_operations(root, fixtures):
    review = fixtures['fake_review']
    for kind in ['o0','o1']:
        old = root/f'outputs/gated/planning/{kind}_n3_s0'
        science = root/f'outputs/gated/planning_full/{kind}_n3_s0'
        science.parent.mkdir(parents=True,exist_ok=True); old.rename(science)
        op = root/f'outputs/gated/U001/operations/{kind}_n3_s0'
        review(op,dict(research_id='C009',budget_policy='U001',name=f'{kind}_n3_s0',
            returncode=0,child_terminal=True,inputs_unchanged=True,timed_out=False,actions=[],errors=[],
            child_artifact_sha256={'result.json':digest(science/'result.json')}))
        checked = dict(research_id='U002',checks={'cases':3000},all_task_scoring_summary_checks_passed=True,
            result_sha256=digest(science/'result.json'),operational_result_sha256=digest(op/'result.json'))
        temporary = root/'inspection_fixture'/kind
        review(temporary,checked)
        ir = json.loads((temporary/'review.json').read_text()); ir['result']='inspection.json'
        atomic_json(science/'inspection.json',checked);atomic_json(science/'inspection_review.json',ir)
    for label in ['D001/adapter_check','D002/legacy']:
        review(root/'outputs/gated'/label,dict(research_id=label[:4]))

began=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);checks=[]
(out/'source').mkdir()
for name in HELPERS: shutil.copy2(BASE/name,out/'source'/name)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
frozen={str(p):digest(p) for label in ['U001','U003','C010','A018'] for p in Path('outputs/gated',label).glob('*.py')}
try:
    record,rh=require_record('U005')
    for name in HELPERS: ast.parse((BASE/name).read_text())
    stages,supervision,core,unlimited,changed=parents()
    for name in ['route','arguments','original_data','p26_reviews','prior_training','consultation']:
        assert Path(getattr(stages,name).__code__.co_filename).resolve()==(ORIGINAL/'stages.py').resolve()
    assert Path(supervision.close_intervals.__code__.co_filename).resolve()==(ORIGINAL/'supervision.py').resolve()
    for name in ['accounting_run','accounting_remaining','wait_progress']:
        fn=getattr(unlimited,name);fn=getattr(fn,'__wrapped__',fn)
        assert Path(fn.__code__.co_filename).resolve()==Path('outputs/gated/U001/common.py').resolve()
    for path,expected in record['protocol']['sources'].items(): assert digest(path)==expected
    assert set(changed)==core.BALLS_CHANGES
    checks.append('Exact original routes/recipe/data/P26/prior-training/consultation/closure; reviewed U001 functions; only three C014 source reconciliations')
    for name in stages.ORDER:
        p=stages.route(Path.cwd(),name);old=stages.arguments('/fixture/python',p);new=arguments('/fixture/python',p,stages)
        assert new[2:]==old[2:] and new[:2]==['/fixture/python',str(BASE/'science.py')]
        assert p['science']==Path.cwd()/'outputs/gated/C001'/name and p['operations']==Path.cwd()/'outputs/gated/C010/runs'/name
        assert progress_paths(p,p['operations'])==[p['operations']/'training.log']+[p['science']/f for f in ['metrics.jsonl','checkpoint.pt','model.pt']]
    for name in ['l0_n3_s0','o1_n3_s1','l1_n1_s1','l0_n3_s3']:refuses(lambda:stages.route(Path.cwd(),name))
    checks.append('All four exact original command recipes and paths; checkpoint/log progress paths; unapproved routes refused')
    fixture=original_fixtures();review=fixture['fake_review']
    with tempfile.TemporaryDirectory(prefix='u005_') as temporary:
        root=Path(temporary);lp=root/'ledger.json';now=time.time()
        past=dict(id='past',research_id='C001',out='past',start=now-216010,end=now-10)
        atomic_json(lp,[past]);outer=admit(lp,root/'operations','C001')
        assert outer['budget_policy']=='U001' and outer['supervisor_research_id']=='C010'
        previous=digest(lp);refuses(lambda:admit(lp,root/'again','C001'));assert digest(lp)==previous
        refuses(lambda:admit(lp,root/'wrong','D001'));assert digest(lp)==previous
        science=root/'science';science.mkdir();atomic_json(science/'manifest.json',dict(run_id='inner'))
        rows=json.loads(lp.read_text())+[dict(id='inner',research_id='C001',out=str(science),start=time.time(),end=None),
            dict(id='foreign',research_id='C002',out='foreign',start=time.time(),end=None)]
        atomic_json(lp,rows)
        assert set(supervision.close_intervals(lp,outer['id'],None,science,root,'C001'))=={outer['id'],'inner'}
        closed=json.loads(lp.read_text());assert closed[0]==past and closed[-1]==rows[-1]
        checks.append('Closed60-hour ledger admits; open/wrong-scope refuse; original nested closure preserves foreign row')

        data={}
        for split in ['train','val']:
            for name in ['meta.json','labels.npz','frames.npy']:
                path=root/f'outputs/data/balls/n3/{split}/{name}';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic original data')
                data[str(path.relative_to(root))]=digest(path)
        review(root/'outputs/gated/A014/preparation',dict(research_id='A014',data_sha256=data))
        fixture['core_fixture'](root,0);add_core_operations(root,fixture)
        for label in ['C010','D003','U001','U003','U005']:
            atomic_json(root/f'docs/research/records/{label}.json',dict(synthetic=True))
            review(root/'outputs/gated'/label/'software',dict(research_id=label,source_sha256={}))
        p=stages.route(root,stages.ORDER[0]);refuses(lambda:inputs(p,stages,core))
        fixture['p26_fixture'](root)
        files=inputs(p,stages,core);assert files
        for kind in ['o0','o1']:
            for name in ['review.json','inspection_review.json']:
                assert (root/f'outputs/gated/planning_full/{kind}_n3_s0'/name).resolve() in files
            assert (root/f'outputs/gated/U001/operations/{kind}_n3_s0/review.json').resolve() in files
        refuses(lambda:inputs(stages.route(root,stages.ORDER[1]),stages,core))
        fixture['training_fixture'](root,stages.ORDER[0]);assert inputs(stages.route(root,stages.ORDER[1]),stages,core)
        fixture['training_fixture'](root,stages.ORDER[1]);refuses(lambda:inputs(stages.route(root,stages.ORDER[2]),stages,core))
        fixture['core_fixture'](root,1);assert inputs(stages.route(root,stages.ORDER[2]),stages,core)
        fixture['training_fixture'](root,stages.ORDER[2]);assert inputs(stages.route(root,stages.ORDER[3]),stages,core)
        checks.append('Complete original P26/prior-training gates and15 full core reviews; all four seeds ordered; missing P26/L0/seed1core refuse')

        dest=root/'outputs/gated/planning/l0_n3_s0';good=json.loads((dest/'result.json').read_text())
        bad=copy.deepcopy(good);bad['arms']['l0']['4']['count']=999
        review(dest,bad);refuses(lambda:inputs(p,stages,core));review(dest,good,status='measured_negative')
        assert inputs(p,stages,core)
        dest=root/'outputs/gated/binding/slots_n3_s0/cases.jsonl';saved=dest.read_bytes();dest.write_bytes(b'changed')
        refuses(lambda:inputs(p,stages,core));dest.write_bytes(saved)
        dest=root/'outputs/gated/planning_full/o0_n3_s0/inspection.json';saved=dest.read_bytes();dest.write_bytes(b'changed')
        refuses(lambda:inputs(p,stages,core));dest.write_bytes(saved)
        dest=root/f'outputs/gated/C001/{stages.ORDER[0]}/checkpoint.pt';saved=dest.read_bytes();dest.write_bytes(b'changed')
        refuses(lambda:inputs(stages.route(root,stages.ORDER[1]),stages,core));dest.write_bytes(saved)
        assert inputs(p,stages,core)
        checks.append('Measured-negative science accepted; partial counts, stale inspection, binding-case and full-checkpoint tampering refuse')

        files=inputs(p,stages,core);pre=root/'pre.json'
        value=dict(stage='train',name=p['name'],research_sha256=require_record('C010')[1],scientific_research_sha256=require_record('C001')[1],
            inputs_sha256={str(f):digest(f) for f in files},consulted_at='synthetic',sources=[dict(url='synthetic',version='synthetic',sections='synthetic',consequence='not actual consultation')],consequences='synthetic',
            u005_record_sha256=rh,u001_authorization_sha256=require_record('U001')[1],software_review_sha256=digest(root/BASE/'software/review.json'))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common,'BASE',root/BASE);atomic_json(pre,value);consultation(pre,p,files,stages)
            for key in ['u005_record_sha256','u001_authorization_sha256','software_review_sha256','inputs_sha256']:
                wrong=dict(value);wrong[key]=None;atomic_json(pre,wrong);refuses(lambda:consultation(pre,p,files,stages))
        checks.append('Fresh consultation additionally binds complete canonical inputs and exact U001/U005/software identities')
    try:inputs(stages.route(Path.cwd(),stages.ORDER[0]),stages,core)
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve()==Path('outputs/gated/D003/runs/export/review.json').resolve()
    else:raise AssertionError('Actual missing P26 must refuse')
    assert not Path('outputs/gated/C010/runs/l0_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(p)==h for p,h in frozen.items())
    checks.append('Actual missing P26 export refuses before CUDA/output; no Torch, shared ledger or active/historical helper changes')
    result=dict(research_id='U005',research_sha256=rh,status='awaiting_literature_review',checks=checks,check_groups=len(checks),
        elapsed_seconds=time.monotonic()-began,source_sha256={f:digest(BASE/f) for f in HELPERS},
        repository_source_sha256={str(p):digest(p) for p in Path('go2nav').rglob('*.py')},
        record_sha256={f'docs/research/records/{r}.json':require_record(r)[1] for r in ['C001','C010','U001','U003','U005']},
        source_reconciliation=changed,actual_GPU_runs=0,torch_imported=False,
        limitation='Prepared operational adapter only. Actual full core/P26 missing; A018 metadata integration, actual training/state audit/numerics and scientific evaluations remain.')
    atomic_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256','record_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='U005',status='technical_blocked',checks=checks,error=repr(exc)));raise
