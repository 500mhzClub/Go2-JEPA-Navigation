from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,time,sys
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from go2nav.research import atomic_json,digest,require_record,require_review,verify_files
start=time.monotonic();out=Path('outputs/gated/U008/interim_001');out.mkdir(exist_ok=False)
_,rh=require_record('U008')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
try:
 u=module('u008_original_guard','outputs/gated/U002/verify_planning.py');original,_=u.parents()
 old=module('u008_baseline_inspector','outputs/gated/B003/inspection/v2/verify_cases.py')
 sw=Path('outputs/gated/C009/v2/planning_inspection_software')
 assert require_review(sw/'review.json')['status']=='passed'
 assert json.loads((sw/'result.json').read_text())['source_sha256']==digest('outputs/gated/C009/v2/inspect_planning.py')
 live=Path('outputs/gated/planning_full/o0_n3_s0/cases.jsonl')
 data=live.read_bytes();complete=data[:data.rfind(b'\n')+1]
 (out/'observed_cases.jsonl').write_bytes(complete)
 rows=[json.loads(x) for x in complete.splitlines()]
 keys={(r['episode'],r['horizon']) for r in rows};assert len(keys)==len(rows)
 count=0
 while all((count,h) in keys for h in [4,6,8]): count+=1
 assert count>0
 selected=[r for r in rows if r['episode']<count]
 (out/'selected_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
 source_files=[Path('docs/research/records/U008.json'),Path(__file__),Path('outputs/gated/C009/v2/inspect_planning.py'),Path('outputs/gated/B003/inspection/v2/verify_cases.py'),sw/'review.json',sw/'result.json',Path('outputs/data/balls/n3/test/labels.npz'),Path('outputs/gated/planning_full/o0_n3_s0/manifest.json')]
 all_rows={};review_status={}
 for key,path in [('controls','outputs/gated/B001/n3'),('l0','outputs/gated/planning/l0_n3_s0'),('l1','outputs/gated/planning/l1_n3_s0')]:
  p=Path(path);r=require_review(p/'review.json');assert r['status'] in ['passed','measured_negative']
  raw=json.loads((p/'result.json').read_text());assert raw['cases_sha256']==digest(p/'cases.jsonl')
  if key=='controls': assert raw['controls_passed']
  all_rows[key]=[json.loads(x) for x in (p/'cases.jsonl').read_text().splitlines()];review_status[key]=r['status']
  source_files.extend([p/'cases.jsonl',p/'result.json',p/'review.json'])
 controls={(r['episode'],r['horizon']):r for r in all_rows['controls'] if r['arm']=='recorded'}
 with np.load('outputs/data/balls/n3/test/labels.npz') as z: labels={k:z[k] for k in ['red','white']}
 checks={'o0':original.audit_rows(selected,controls,labels,3,'o0',count)}
 grouped={'o0':selected}
 for key in ['l0','l1']:
  prefix=[r for r in all_rows[key] if r['episode']<count]
  checks[key]=old.audit_rows(prefix,controls,labels,3,key,count);grouped[key]=prefix
 for arm in ['recorded','direct','true_cem','hold','random']:
  prefix=[r for r in all_rows['controls'] if r['episode']<count and r['arm']==arm]
  assert len(prefix)==3*count
  grouped[arm]=prefix
 for arm,values in grouped.items():
  lookup={(r['episode'],r['horizon']):r for r in values}
  assert len(lookup)==3*count
  for r in selected:
   s=lookup[r['episode'],r['horizon']]
   for field in ['n','episode','horizon','start_frame','goal_frame','start_position','goal_position','seed','group']:assert r[field]==s[field],(arm,field)
 summary={}
 for arm,values in grouped.items():
  summary[arm]={}
  for h in [4,6,8]:
   subset=[r for r in values if r['horizon']==h]
   summary[arm][str(h)]={'count':len(subset),'successes':sum(r['success'] for r in subset),'success_rate':sum(r['success'] for r in subset)/len(subset),'collision_free_successes':sum(r['collision_free_success'] for r in subset),'endpoint_error_mean':float(np.mean([r['endpoint_error'] for r in subset])),'endpoint_error_median':float(np.median([r['endpoint_error'] for r in subset]))}
 for path in ['outputs/gated/probes/o0_n3_s0/review.json','outputs/gated/probes/o1_n3_s0/review.json','outputs/gated/binding/slots_n3_s0/review.json']:
  p=Path(path);r=require_review(p);assert r['status']=='measured_negative';source_files.extend([p,p.parent/r['result']])
 result=dict(research_id='U008',research_sha256=rh,status='awaiting_literature_review',observed_at=datetime.now(timezone.utc).isoformat(),run='o0_n3_s0',run_complete=False,observed_rows=len(rows),selected_rows=len(selected),complete_episodes=count,excluded_incomplete_episode_rows=len(rows)-len(selected),snapshot_sha256=digest(out/'observed_cases.jsonl'),selected_sha256=digest(out/'selected_cases.jsonl'),source_sha256={str(p):digest(p) for p in source_files},independent_physical_checks=checks,paired_task_fields_match=True,prior_review_status=review_status,summary=summary,elapsed_seconds=time.monotonic()-start,limitation='Interim first completed episodes from one training seed; no final3000case provenance/terminal inspection, no O1 planning, no population or sequential significance claim. Numerical CEM internals are not replayed.')
 atomic_json(out/'result.json',result)
 print(json.dumps({k:result[k] for k in ['observed_at','observed_rows','selected_rows','complete_episodes','independent_physical_checks','summary','elapsed_seconds']},indent=2))
except BaseException as exc:
 atomic_json(out/'failure.json',dict(research_id='U008',status='technical_blocked',error=repr(exc)));raise
