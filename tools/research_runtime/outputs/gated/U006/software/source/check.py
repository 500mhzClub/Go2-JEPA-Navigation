"""Focused metadata/terminal integration checks without tensor execution."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import psutil
import pytest
import metadata
from metadata import BASE,ORIGINAL,HELPERS,check_policy,expected_metadata,module,qualified,training_software
from go2nav.research import atomic_json,digest,require_record

def refuses(call):
    with pytest.raises((ValueError,FileNotFoundError,KeyError,RuntimeError)):call()

def functions(source):
    return {node.name:ast.dump(node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}

began=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);checks=[]
(out/'source').mkdir()
for name in HELPERS:shutil.copy2(BASE/name,out/'source'/name)
ledger=Path('outputs/gated/gpu_intervals.json');before=digest(ledger)
frozen={str(p):digest(p) for label in ['A018','C010','U001','U005'] for p in Path('outputs/gated',label).glob('*.py')}
try:
    record,rh=require_record('U006');protocol=record['protocol']
    expected=(ORIGINAL/'audit.py').read_text()
    assert digest(ORIGINAL/'audit.py')==protocol['inherited_audit_sha256']
    for old,new in json.loads((BASE/'controlled_changes.json').read_text()):
        assert expected.count(old)==1;expected=expected.replace(old,new)
    assert (BASE/'audit.py').read_text()==expected
    old_functions=functions((ORIGINAL/'audit.py').read_text());new_functions=functions(expected)
    for name in ['need','terminal_checks','source_checks']:assert old_functions[name]==new_functions[name]
    assert digest(ORIGINAL/'state_contract.py')==protocol['inherited_state_contract_sha256']
    for name in HELPERS:ast.parse((BASE/name).read_text())
    audit=module('u006_integrated_audit',BASE/'audit.py')
    assert audit.BASE==ORIGINAL and audit.NAMES==['l0_n3_s1','l1_n3_s1','l0_n3_s2','l1_n3_s2']
    qualified(check_self=False);sw=training_software();expected_policy=expected_metadata(sw)
    checks.append('Exact controlled audit copy; original terminal/source functions and state contract unchanged; actual A018/U005 lineage and source reconciliation verified')

    with tempfile.TemporaryDirectory(prefix='u006_') as temporary:
        root=Path(temporary);sf=root/'science/l1_n3_s1';operations=root/'operations/l1_n3_s1'
        op=dict(research_id='C010',scientific_research_id='C001',stage='train',name='l1_n3_s1',
            integration_research_id='U005',budget_policy='U001',timeout_type=None,returncode=0,child_terminal=True,
            timed_out=False,actions=[],errors=[],inputs_unchanged=True,raw_status='awaiting_literature_review',closed_interval_ids=['outer'])
        om=dict(research_id='C010',scientific_research_id='C001',stage='train',name='l1_n3_s1',
            integration_research_id='U005',integration_research_sha256=expected_policy['u005_record_sha256'],
            budget_policy='U001',authorization_sha256=expected_policy['authorization_sha256'],
            software_review_sha256=expected_policy['software_review_sha256'],source_sha256=sw['source_sha256'],
            research_sha256=expected_policy['research_sha256'],scientific_research_sha256=expected_policy['scientific_research_sha256'],
            work_seconds=None,stall_seconds=1800,preconsultation=dict(u005_record_sha256=expected_policy['u005_record_sha256'],
                u001_authorization_sha256=expected_policy['authorization_sha256'],software_review_sha256=expected_policy['software_review_sha256']),
            pid=os.getpid(),creation_time=0.,cwd=str(root),outer_interval_id='outer')
        sm=dict(run_id='inner',budget_policy='U001',authorization_sha256=expected_policy['authorization_sha256'],
            research_sha256=expected_policy['scientific_research_sha256'],budget_hours=None)
        rows=[dict(id='outer',research_id='C001',supervisor_research_id='C010',integration_research_id='U005',budget_policy='U001',start=1,end=3,out=str(operations)),
              dict(id='inner',research_id='C001',budget_policy='U001',start=2,end=3,out=str(sf))]
        good=(op,om,sm,rows)
        audit.terminal_checks(*good,sf,operations);check_policy(*good,expected_policy)
        faults=[]
        for index,key,value in [(0,'integration_research_id','wrong'),(1,'integration_research_id','wrong'),
            (0,'budget_policy','wrong'),(1,'budget_policy','wrong'),(2,'budget_policy','wrong'),
            (1,'authorization_sha256','wrong'),(2,'authorization_sha256','wrong'),(1,'integration_research_sha256','wrong'),
            (1,'software_review_sha256','wrong'),(1,'source_sha256',{}),(1,'research_sha256','wrong'),
            (1,'scientific_research_sha256','wrong'),(2,'research_sha256','wrong'),(1,'work_seconds',1),
            (2,'budget_hours',1),(1,'stall_seconds',1),(0,'timeout_type','no_output_progress')]:
            bad=copy.deepcopy(good);bad[index][key]=value;faults.append(bad)
        for key in ['u005_record_sha256','u001_authorization_sha256','software_review_sha256']:
            bad=copy.deepcopy(good);bad[1]['preconsultation'][key]='wrong';faults.append(bad)
        for index,key,value in [(0,'budget_policy','wrong'),(1,'budget_policy','wrong'),(0,'integration_research_id','wrong')]:
            bad=copy.deepcopy(good);bad[3][index][key]=value;faults.append(bad)
        for bad in faults:refuses(lambda:check_policy(*bad,expected_policy))
        checks.append(f'Complete U005/null-budget metadata accepted; {len(faults)} authorization/software/source/consultation/interval corruptions refused')

        terminal_faults=[]
        for key,value in [('returncode',1),('child_terminal',False),('timed_out',True),('actions',['TERM']),('errors',['error']),('inputs_unchanged',False),('name','l1_n3_s2'),('raw_status','budget_incomplete')]:
            bad=copy.deepcopy(good);bad[0][key]=value;terminal_faults.append(bad)
        bad=copy.deepcopy(good);bad[1]['creation_time']=psutil.Process(os.getpid()).create_time();terminal_faults.append(bad)
        for key,value in [('end',None),('research_id','C005'),('supervisor_research_id','C009'),('out','foreign')]:
            bad=copy.deepcopy(good);bad[3][0][key]=value;terminal_faults.append(bad)
        bad=copy.deepcopy(good);bad[3].append(copy.deepcopy(rows[0]));terminal_faults.append(bad)
        for bad in terminal_faults:refuses(lambda:audit.terminal_checks(*bad,sf,operations))
        checks.append(f'All {len(terminal_faults)} original terminal/live-identity/closed-owned-interval fault fixtures still refuse')

        runtime=root/'runtime';audit_root=root/'audit'
        for base in [runtime,audit_root]:
            (base/'go2nav').mkdir(parents=True);(base/'go2nav/code.py').write_text('same computation')
            (base/'docs').mkdir();(base/'docs/note.md').write_text('old' if base==runtime else 'new')
        bound={str(p.relative_to(runtime)):digest(p) for p in runtime.rglob('*') if p.is_file()}
        audit.source_checks(audit_root,runtime,bound)
        (runtime/'docs/note.md').write_text('changed');refuses(lambda:audit.source_checks(audit_root,runtime,bound))
        (runtime/'docs/note.md').write_text('old');(audit_root/'go2nav/code.py').write_text('changed');refuses(lambda:audit.source_checks(audit_root,runtime,bound))
        checks.append('Original pinned-runtime and current-computation source routing retained')

        # The real integration review does not exist until these checks are reviewed.
        fake=root/'integration';(fake/'software').mkdir(parents=True);(fake/'software/review.json').write_text('synthetic-only')
        value=dict(u006_record_sha256=rh,u006_software_review_sha256=digest(fake/'software/review.json'),u001_authorization_sha256=require_record('U001')[1])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(metadata,'BASE',fake);metadata.check_consultation(value)
            for key in value:
                bad=dict(value);bad[key]='wrong';refuses(lambda:metadata.check_consultation(bad))
        checks.append('Fresh audit consultation binds U006 record/review and U001 authorization; all missing/wrong fields refuse')
    try:audit.admission('l0_n3_s1')
    except FileNotFoundError as exc:
        assert Path(exc.filename).resolve()==Path('outputs/gated/C010/runs/l0_n3_s1/result.json').resolve()
    else:raise AssertionError('Actual incomplete training must refuse')
    assert not (ORIGINAL/'runs/l0_n3_s1').exists()
    assert 'torch' not in sys.modules and digest(ledger)==before and all(digest(p)==h for p,h in frozen.items())
    checks.append('Actual missing trained result refuses before output; no Torch, actual state claim, shared ledger or active helper mutation')
    result=dict(research_id='U006',research_sha256=rh,status='awaiting_literature_review',checks=checks,check_groups=len(checks),
        metadata_corruptions=len(faults),terminal_corruptions=len(terminal_faults),elapsed_seconds=time.monotonic()-began,
        source_sha256={name:digest(BASE/name) for name in HELPERS},controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
        repository_source_sha256={str(p):digest(p) for p in Path('go2nav').rglob('*.py')},
        inherited_state_contract_sha256=protocol['inherited_state_contract_sha256'],actual_completed_extra_runs=0,torch_imported=False,
        limitation='Metadata/source/terminal integration only. Original tensor checks are byte-identical and inherited, not rerun; all actual extra-seed artifacts and audits remain absent.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='repository_source_sha256'}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='U006',status='technical_blocked',checks=checks,error=repr(exc)));raise
