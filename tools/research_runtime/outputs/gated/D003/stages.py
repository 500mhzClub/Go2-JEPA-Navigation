"""Fixed D001 stage routing and input/review bindings; no Torch imports."""
import json
from pathlib import Path
from go2nav.research import digest, require_review, reviewed_artifact, verify_files


def paths(root,stage):
    if stage not in ['export','s1','compare']:raise ValueError('Unplanned D001 stage')
    root=Path(root);base=root/'outputs/gated/D001';corpus=base/'corpus'
    prepared=json.loads((corpus/'prepare_manifest.json').read_text())
    if len(prepared['inputs'])!=1:raise ValueError('Ambiguous original feature index')
    index=Path(next(iter(prepared['inputs'])))
    if not index.is_absolute():index=root/index
    return dict(root=root,stage=stage,base=base,corpus=corpus,features=index.parent,
                feature_index=index,mazes=root/'data/mazes',s0=root/'outputs/camera_slots/s0/model.pt',
                m5=root/'outputs/object_dynamics/m5/model.pt',s1=base/'s1/model.pt',
                science={'export':corpus/'track_run','s1':base/'s1','compare':base/'comparison'}[stage],
                operations=root/'outputs/gated/D003/runs'/stage)


def arguments(python,p):
    argv=[str(python),'-m']
    c=str(p['corpus']);b=p['base']
    if p['stage']=='export':
        argv+=['go2nav.data.slot_tracks','export','--corpus',c,'--features',str(p['features']),
               '--checkpoint',str(p['s0']),'--mazes',str(p['mazes']),'--research','D001']
    elif p['stage']=='s1':
        argv+=['go2nav.training.object_jepa','train','--corpus',c,'--out',str(p['science']),
               '--config',str(p['root']/'configs/object/m5.yaml'),'--set','seed=0','--research','D001',
               '--parent-review',str(p['corpus']/'track_run/review.json')]
    else:
        argv+=['go2nav.eval.slot_dynamics','--corpus',c,'--m5',str(p['m5']),'--s1',str(p['s1']),
               '--mazes',str(p['mazes']),'--tracks-review',str(p['corpus']/'track_run/review.json'),
               '--s1-review',str(b/'s1/review.json'),'--out',str(p['science']),'--research','D001']
    return argv


def inventory(p):
    c=p['corpus'];source=json.loads((c/'index.json').read_text());features=json.loads(p['feature_index'].read_text())
    prepared=json.loads((c/'prepare_manifest.json').read_text())
    expected=next(iter(prepared['inputs'].values()))
    if digest(p['feature_index'])!=expected or source['features_index_sha256']!=expected:raise ValueError('Original feature selection changed')
    files={c/'index.json',c/'prepare_manifest.json',p['feature_index']}
    metadata=[]
    for split,count in [('train',600),('val',100)]:
        metas=sorted((c/split).glob('*.json'))
        units=[json.loads(f.read_text())['unit'] for f in metas]
        if len(metas)!=count or units!=source['units'][split] or units!=features['units']['lewm_'+split]:raise ValueError('Prepared unit selection changed')
        for f in metas:
            files.update([f,f.with_suffix('.npz')]);metadata.append(json.loads(f.read_text()))
        files.update(c/'packed'/f'{split}_{name}.npy' for name in ['pose','requested','ranges','starts'])
    maze_files=set(p['mazes'].rglob('*.json'))
    if not maze_files:raise ValueError('No existing maze registry files')
    files.update(maze_files)
    if p['stage']=='export':
        files.update([p['s0'],p['s0'].with_name('config.json')])
        for split in ['train','val']:
            files.update(p['features']/('lewm_'+split)/(Path(unit).name+'.npy') for unit in source['units'][split])
    else:
        files.update([c/'tracks_index.json',c/'tracks_config.json',c/'track_run/review.json',
                      p['root']/'outputs/gated/D003/runs/export/review.json'])
        index=json.loads((c/'tracks_index.json').read_text())
        files.update(c/f for f in index['files'])
        expected_tracks={f'{folder}/{split}/{f.stem}.npz' for folder in ['objects','engineered'] for split in ['train','val'] for f in (c/split).glob('*.json')}
        if set(index['files'])!=expected_tracks:raise ValueError('Track index incomplete')
        rasters=set((c/'objects/rasters').glob('*.npz'))
        if {int(f.stem) for f in rasters}!={m['maze'] for m in metadata}:raise ValueError('Raster coverage mismatch')
        files.update(rasters)
        if p['stage']=='s1':files.add(p['root']/'configs/object/m5.yaml')
        else:files.update([p['m5'],p['s1'],p['base']/'s1/review.json',p['base']/'s1/config.json',
                          p['root']/'outputs/gated/D003/runs/s1/review.json'])
    if any(not f.is_file() for f in files):raise ValueError('A required existing input is absent')
    return sorted(files)


def core_reviews(root):
    root=Path(root);reviews=[]
    for kind in ['l0','l1','o0','o1']:
        for phase in ['planning','probes']:
            p=root/'outputs/gated'/phase/f'{kind}_n3_s0/review.json'
            review=require_review(p);data=json.loads((p.parent/review['result']).read_text())
            ident=review.get('run_identity',{})
            actual=(data.get('n',ident.get('n')),data.get('kind',ident.get('kind')),data.get('seed',ident.get('seed')))
            if review['status'] not in ['passed','measured_negative'] or actual!=(3,kind,0):raise ValueError('n3 core review/identity missing')
            cases=p.parent/('cases.jsonl' if phase=='planning' else 'cases.npz')
            if digest(cases)!=data['cases_sha256']:raise ValueError('Core raw cases changed')
            reviews.append(p)
    p=root/'outputs/gated/binding/slots_n3_s0/review.json';review=require_review(p)
    data=json.loads((p.parent/review['result']).read_text())
    if review['status'] not in ['passed','measured_negative'] or (data['n'],data['kind'],data['seed'])!=(3,'slots',0):raise ValueError('Core binding absent')
    reviews.append(p)
    for phase in ['D001/adapter_check','D002/legacy']:
        p=root/'outputs/gated'/phase/'review.json'
        if require_review(p)['status']!='passed':raise ValueError('Existing D evidence not passed')
        reviews.append(p)
    return reviews


def stage_parents(p):
    if p['stage']=='export':return []
    c=p['corpus'];tracks=c/'track_run/review.json'
    reviewed_artifact(tracks,c/'tracks_index.json','tracks_sha256')
    verify_files(c,json.loads((c/'tracks_index.json').read_text())['files'])
    reviews=[tracks]
    for stage in (['export'] if p['stage']=='s1' else ['export','s1']):
        rp=p['root']/'outputs/gated/D003/runs'/stage/'review.json';r=require_review(rp)
        evidence=json.loads((rp.parent/r['result']).read_text())
        if r['status']!='passed' or evidence['stage']!=stage or not evidence['child_terminal'] or evidence['returncode']!=0:raise ValueError('Actual D stage operational review required')
        verify_files(p['root'],evidence['child_artifact_sha256']);reviews.append(rp)
    if p['stage']=='compare':
        sr=p['base']/'s1/review.json';evidence=reviewed_artifact(sr,p['s1'])
        if evidence['research_id']!='D001' or evidence['step']!=20000:raise ValueError('S1 did not finish declared training')
        reviews.append(sr)
        legacy=json.loads((p['root']/'outputs/gated/D002/legacy/manifest.json').read_text())
        if digest(p['m5'])!=legacy['inputs']['outputs/object_dynamics/m5/model.pt']:raise ValueError('Original M5 changed')
    return reviews


def consultation(path,p,files,reviews,record_hash,science_hash):
    value=json.loads(Path(path).read_text())
    if value.get('stage')!=p['stage'] or value.get('d003_record_sha256')!=record_hash or value.get('d001_record_sha256')!=science_hash:raise ValueError('Wrong stage/protocol consultation')
    expected={str(f.resolve()) for f in files}
    if not isinstance(value.get('inputs_sha256'),dict) or set(value['inputs_sha256'])!=expected:raise ValueError('Incomplete consultation inputs')
    expected_reviews={str(f.resolve()):digest(f) for f in reviews}
    if value.get('reviews_sha256')!=expected_reviews:raise ValueError('Stale parent reviews')
    if not value.get('consulted_at') or not value.get('sources') or not value.get('consequences'):raise ValueError('Missing substantive consultation')
    if any(not all(s.get(k) for k in ['url','version','sections','consequence']) for s in value['sources']):raise ValueError('Incomplete source consultation')
    verify_files(p['root'],value['inputs_sha256'])
    return value


def generated(p):
    files=set(f for f in p['science'].rglob('*') if f.is_file())
    if p['stage']=='export':
        files.update(f for folder in ['objects','engineered'] for f in (p['corpus']/folder).rglob('*') if f.is_file())
        files.update(f for f in [p['corpus']/'tracks_index.json',p['corpus']/'tracks_config.json'] if f.exists())
    return sorted(files)
