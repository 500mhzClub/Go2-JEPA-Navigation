"""C015 focused CPU numerical/layout and inherited-accounting verification."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import torch
from common import BASE,HELPERS,admit,close,code_hashes,inherited,need
from costs import VARIANTS
from go2nav.eval.balls_planning import slot_distance
from go2nav.research import atomic_json,digest,require_record

start=time.monotonic();out=BASE/'software';out.mkdir(exist_ok=False);(out/'source').mkdir();checks=[]
for f in HELPERS:shutil.copy2(BASE/f,out/'source'/f)
try:
    torch.set_num_threads(2);rng=np.random.default_rng(2026111601)
    with torch.no_grad():
        for slots in [3,5,8]:
            for batch in [1,8,100]:
                p=torch.tensor(rng.normal(size=(batch,slots,128)),dtype=torch.float64);g=torch.tensor(rng.normal(size=(1,slots,128)),dtype=torch.float64)
                for fixture,(x,y) in {'random':(p,g),'zero':(p*0,g*0),'permuted':(p.flip(1),g),'tied_candidates':(p[:1].expand_as(p),g)}.items():
                    ref=slot_distance(x,y).numpy()
                    for fn in VARIANTS.values():
                        actual=fn(x,y).numpy();need(np.allclose(actual,ref,rtol=1e-12,atol=1e-12),'Cost mismatch');need(np.array_equal(np.argsort(actual,kind='stable'),np.argsort(ref,kind='stable')),'Ranking mismatch')
            checks.append(f'slots{slots}_batches1_8_100_random_zero_permuted_and_tied_costs_ranks')
        for name in ['o0_n3_s0','o1_n3_s0']:
            for split in ['train','val']:
                with np.load(Path('outputs/gated/C014')/name/f'{split}.npz') as z:
                    for prefix,goal in [('cpu','cpu_goal'),('batched','gpu_goal')]:
                        x=torch.from_numpy(z[prefix+'_rollout'][:,-1]);y=torch.from_numpy(z[goal]);ref=slot_distance(x,y).numpy()
                        for fn in VARIANTS.values():
                            actual=fn(x,y).numpy();need(np.allclose(actual,ref,rtol=1e-12,atol=1e-12),'Saved-array cost mismatch');need(np.array_equal(np.argsort(actual,kind='stable'),np.argsort(ref,kind='stable')),'Saved-array ranking mismatch')
            checks.append(name+'_saved_train_val_CPU_and_GPU_arrays_costs_ranks')
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='C015':node.value='C014'
            elif node.value=='C015: exact owned child confirmed terminal':node.value='C014: exact owned child confirmed terminal'
            elif node.value==900:node.value=1800
            return node
    old=ast.parse(Path('outputs/gated/C014/common.py').read_text());new=ast.parse((BASE/'common.py').read_text())
    for name in ['admit','close']:
        a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);need(ast.dump(a)==ast.dump(Normalize().visit(b)),'Accounting differs beyond cap/identity')
    inherited();checks.append('reviewed_owned_process_inherited_and_accounting_AST_preserved_cap900')
    def refuses(call):
        try:call()
        except ValueError:return
        raise AssertionError('Expected refusal')
    with tempfile.TemporaryDirectory() as td:
        ledger=Path(td)/'ledger.json';atomic_json(ledger,[]);row,left,work=admit(ledger,Path(td));need(work==855,'Wrong local cap');refuses(lambda:admit(ledger,Path(td)));close(ledger,row['id'],None);need(json.loads(ledger.read_text())[0]['end'] is not None,'Interval not closed');refuses(lambda:close(ledger,row['id'],None));atomic_json(ledger,[dict(id='old',research_id='C004',start=0.,end=8*3600)]);refuses(lambda:admit(ledger,Path(td)))
    checks.append('local_cap_open_interval_closed_identity_and_exhausted_reserve_refusals')
    for f in HELPERS:compile((BASE/f).read_text(),f,'exec')
    need(not torch.cuda.is_initialized(),'CUDA initialized');checks.append('all_helpers_compile_no_CUDA')
    result=dict(research_id='C015',research_sha256=require_record('C015')[1],status='awaiting_literature_review',checks=checks,elapsed_seconds=time.monotonic()-start,helper_sha256={f:digest(BASE/f) for f in HELPERS},repository_source_sha256=code_hashes(),limitation='CPU software and saved-array arithmetic only. Actual GPU/full CEM speed and exact decision equivalence unmeasured; no numerical policy, model or scientific sample changed.')
    atomic_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['repository_source_sha256']}),flush=True)
except BaseException as exc:
    atomic_json(out/'result.json',dict(research_id='C015',status='technical_blocked',checks=checks,error=repr(exc)));raise
