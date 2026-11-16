"""Freeze available before/current checkpoints through open file descriptors."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path.cwd()))
assert os.environ.get('HIP_VISIBLE_DEVICES')==os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files

base=Path('outputs/gated/A017');record,rh=require_record('A017')
assert require_review(base/'software/review.json')['status']=='passed'
plan=json.loads(Path('outputs/gated/A014/continuation_plan.json').read_text())
command=next(c for c in plan['recovery_commands'] if c['name']=='l1_n1_s0')
original=Path(command['cwd']);verify_files(original,plan['original_source_sha256'])
verify_files(Path.cwd(),command['data_sha256'])
sources=dict(before=Path('outputs/balls/recovery_A005/l1_n1_s0'),current=Path('outputs/balls/recovery_A014/l1_n1_s0'))
out=base/'snapshots';out.mkdir(exist_ok=False);snapshots={};hashes={}
for name,source in sources.items():
    folder=out/name;folder.mkdir()
    with (source/'checkpoint.pt').open('rb') as f,(folder/'checkpoint.pt').open('xb') as g:
        st=os.fstat(f.fileno());shutil.copyfileobj(f,g);g.flush();os.fsync(g.fileno())
        f.seek(0);h=hashlib.file_digest(f,'sha256').hexdigest();after=os.fstat(f.fileno())
    assert (st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert digest(folder/'checkpoint.pt')==h
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['kind']=='l1' and state['n']==1
    if name=='before':assert state['step']==105000 and h==command['checkpoint_sha256']
    else:assert 200000<=state['step']<=303000
    assert digest(source/'config.json')==command['input_sha256']['config.json']
    shutil.copyfile(source/'config.json',folder/'config.json')
    with (source/'metrics.jsonl').open() as f:text=f.read()
    complete=[line for line in text.split('\n')[:-1] if line and json.loads(line)['step']<=state['step']]
    (folder/'metrics.jsonl').write_text(''.join(line+'\n' for line in complete))
    snapshots[name]=dict(source=str(source),step=state['step'],checkpoint_sha256=h,source_inode=st.st_ino,
                         source_mtime_ns=st.st_mtime_ns,source_bytes=st.st_size,metric_rows=len(complete))
    hashes.update({str((folder/n).relative_to(out)):digest(folder/n) for n in ['checkpoint.pt','config.json','metrics.jsonl']})
assert not torch.cuda.is_initialized()
code=[Path(p) for p in ['go2nav/models/lewm.py','go2nav/training/balls.py','go2nav/data/balls.py',
    'go2nav/eval/balls_inference.py','go2nav/eval/balls_interim.py','go2nav/eval/balls_probes.py','go2nav/research.py',
    'outputs/gated/A017/analysis.py','outputs/gated/A017/check_analysis.py','outputs/gated/A017/run_stage.py','outputs/gated/A017/prepare.py']]
manifest=dict(research_id='A017',research_sha256=rh,created_at=time.time(),snapshot_sha256=hashes,snapshots=snapshots,
              data_sha256=command['data_sha256'],source_sha256={str(p):digest(p) for p in code},
              validation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              source_note='Uses unchanged A002/A003 float64-capable evaluator/model from161fe8b; original trainer remains7dfc837.',
              original_source_commit=plan['original_source_commit'],cuda_initialized=False)
atomic_json(out/'manifest.json',manifest);print(json.dumps(snapshots),flush=True)
