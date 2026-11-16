"""Inspect full U001 planning with unchanged independent C009 scoring."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import psutil

sys.path.insert(0, str(Path.cwd()))
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files
BASE = Path('outputs/gated/U002')
PARENT = Path('outputs/gated/C009/v2')
RUNTIME = Path('outputs/gated/U001')

def need(condition, message):
    if not condition: raise ValueError(message)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

def parents():
    review = require_review(PARENT/'planning_inspection_software/review.json')
    sw = json.loads((PARENT/'planning_inspection_software/result.json').read_text())
    need(review['status'] == 'passed' and sw['source_sha256'] == digest(PARENT/'inspect_planning.py'), 'Unqualified original inspector')
    verify_files(Path.cwd(), sw['repository_source_sha256'])
    original = module('u002_original_inspector', PARENT/'inspect_planning.py'); original.inherited_guard()
    runtime_review = require_review(RUNTIME/'software/review.json')
    runtime = json.loads((RUNTIME/'software/result.json').read_text())
    need(runtime_review['status'] == 'passed' and runtime['research_sha256'] == require_record('U001')[1], 'Unqualified U001 runtime')
    verify_files(RUNTIME, runtime['source_sha256']); verify_files(Path.cwd(), runtime['repository_source_sha256'])
    return original, runtime

def paths(name, root=None):
    need(name in ('o0_n3_s0', 'o1_n3_s0'), 'Unplanned arm')
    root = Path.cwd() if root is None else Path(root)
    return root/'outputs/gated/planning_full'/name, root/'outputs/gated/U001/operations'/name

def metadata(original, operation, manifest, science_manifest, name, scientific_hash, authorization_hash):
    original.operation_identity(operation, name)
    need(operation['budget_policy'] == manifest['budget_policy'] == science_manifest['budget_policy'] == 'U001', 'Wrong time policy')
    need(manifest['authorization_sha256'] == science_manifest['authorization_sha256'] == authorization_hash, 'Wrong authorization')
    need(manifest['work_seconds'] is None and science_manifest['budget_hours'] is None, 'A superseded time cap remains')
    need(manifest['stage'] == 'planning' and manifest['name'] == name and manifest['scientific_research_id'] == 'B001', 'Wrong manifest identity')
    need(manifest['scientific_research_sha256'] == science_manifest['research_sha256'] == scientific_hash, 'Wrong scientific protocol')
    need(science_manifest['research_id'] == 'B001' and manifest['source_commit'] == science_manifest['code_commit'], 'Wrong scientific source')

def absent(pid, created):
    try: live = psutil.Process(pid).create_time() == created
    except psutil.NoSuchProcess: live = False
    need(not live, 'Owned child remains live')

def intervals(rows, manifest, science_manifest, name):
    science, operation = paths(name, manifest['cwd']); selected = []
    for rid, path in [(manifest['outer_interval_id'], operation), (science_manifest['run_id'], science)]:
        found = [row for row in rows if row['id'] == rid]
        need(len(found) == 1, 'Missing or ambiguous interval')
        row = found[0]
        need(row['research_id'] == 'B001' and row['budget_policy'] == 'U001' and row['end'] is not None and row['end'] >= row['start'], 'Unclosed or wrong interval')
        need((Path(manifest['cwd'])/row['out']).resolve() == path.resolve(), 'Wrong interval output')
        selected.append(row)
    need(selected[0].get('supervisor_research_id') == 'C009' and selected[1]['start'] >= selected[0]['start'], 'Wrong outer/inner ownership')
    return selected

def expected_inputs(runtime_root, name):
    runtime_root = Path(runtime_root)
    stages = module('u002_original_routes', PARENT/'stages.py')
    files = set(stages.inputs(stages.route(runtime_root, 'planning', name)))
    probe = runtime_root/'outputs/gated/probes'/name/'review.json'
    pr = require_review(probe)
    files.update([probe, probe.parent/pr['result'], runtime_root/'docs/research/records/U001.json'])
    if name == 'o1_n3_s0':
        for path in [runtime_root/'outputs/gated/planning_full/o0_n3_s0/review.json', runtime_root/'outputs/gated/planning_full/o0_n3_s0/inspection_review.json', runtime_root/'outputs/gated/U001/operations/o0_n3_s0/review.json']:
            review = require_review(path)
            need(review['status'] in ('passed', 'measured_negative'), 'O0 predecessor unreviewed')
            files.update([path, path.parent/review['result']])
    return {str(f.resolve()):digest(f) for f in files}

def actual(name):
    import torch
    from go2nav.eval.slots_inference import require_numerical_review
    torch.set_num_threads(2); began = time.monotonic()
    _, uh = require_record('U002'); _, ah = require_record('U001'); science_record, sh = require_record('B001')
    original, runtime = parents()
    sw = json.loads((BASE/'software/result.json').read_text())
    need(require_review(BASE/'software/review.json')['status'] == 'passed' and sw['research_sha256'] == uh, 'Unqualified new inspector')
    verify_files(BASE, sw['source_sha256'])
    out, op = paths(name); r = json.loads((out/'result.json').read_text()); o = json.loads((op/'result.json').read_text())
    m = json.loads((op/'manifest.json').read_text()); sm = json.loads((out/'manifest.json').read_text())
    metadata(original, o, m, sm, name, sh, ah); absent(m['pid'], m['creation_time'])
    need(o['manifest_sha256'] == digest(op/'manifest.json'), 'Operation manifest changed')
    verify_files(out, o['child_artifact_sha256'])
    need(m['software_review_sha256'] == digest(RUNTIME/'software/review.json') and m['source_sha256'] == runtime['source_sha256'], 'Runtime software differs')
    runtime_root = Path(m['cwd']); verify_files(runtime_root, runtime['repository_source_sha256'])
    verify_files(runtime_root, sm['source_sha256']); verify_files(runtime_root, sm['inputs'])
    need(sm['config'] == science_record['protocol'] and m['integration_research_sha256'] == require_record('C014')[1], 'Configuration/inference protocol differs')
    need(digest(runtime_root/'outputs/gated/U001'/f'pre_{name}.json') == m['preconsultation_sha256'], 'Preconsultation changed')
    expected = expected_inputs(runtime_root, name)
    need(m['preconsultation']['inputs_sha256'] == expected, 'Incomplete/changed canonical input inventory')
    verify_files(runtime_root, expected)
    checkpoint = Path('outputs/gated/C001')/name/'model.pt'; evidence = require_numerical_review(checkpoint, Path('outputs/gated/C014')/name/'review.json')
    kind = name.split('_')[0]; identity = dict(n=3, kind=kind, seed=0, checkpoint_sha256=digest(checkpoint))
    need((evidence['n'], evidence['kind'], evidence['seed']) == (3, kind, 0), 'Wrong checkpoint identity')
    probe = Path('outputs/gated/probes')/name/'review.json'; pr = require_review(probe)
    need(pr['status'] in ('passed', 'measured_negative') and pr['run_identity'] == identity, 'Wrong preceding probe')
    need((r['research_id'], r['n'], r['status'], r['task_seed'], r['cem_seed']) == ('B001', 3, 'awaiting_literature_review', 2026111503, 2026111504), 'Wrong result identity')
    owned = intervals(json.loads(Path('outputs/gated/gpu_intervals.json').read_text()), m, sm, name)
    need(r['cases_sha256'] == digest(out/'cases.jsonl'), 'Raw cases changed')
    control = Path('outputs/gated/B001/n3'); cr = json.loads((control/'result.json').read_text())
    need(require_review(control/'review.json')['status'] == 'passed' and cr['controls_passed'] and cr['cases_sha256'] == digest(control/'cases.jsonl'), 'Control evidence invalid')
    controls = {(row['episode'],row['horizon']):row for row in map(json.loads,(control/'cases.jsonl').read_text().splitlines()) if row['arm'] == 'recorded'}
    need(len(controls) == 3000, 'Control coverage incomplete')
    with np.load('outputs/data/balls/n3/test/labels.npz') as z: labels = {k:z[k] for k in ['red','white']}
    rows = [json.loads(line) for line in (out/'cases.jsonl').read_text().splitlines()]
    checked = original.audit_rows(rows, controls, labels, 3, kind, 1000)
    summary = original.summary_from_rows(rows, kind); original.compare_summary(r['arms'], summary)
    need(not torch.cuda.is_initialized(), 'CUDA initialized during independent inspection')
    result = dict(research_id='U002', research_sha256=uh, scientific_research_id='B001', name=name, status='awaiting_literature_review', run_identity=identity,
        result_sha256=digest(out/'result.json'), operational_result_sha256=digest(op/'result.json'), source_sha256=digest(__file__), software_review_sha256=digest(BASE/'software/review.json'),
        original_inspector_sha256=digest(PARENT/'inspect_planning.py'), original_inspector_review_sha256=digest(PARENT/'planning_inspection_software/review.json'),
        checks=checked, summary=summary, all_task_scoring_summary_checks_passed=True, bootstrap_repeats=2000, owned_closed_intervals=owned,
        cuda_initialized=False, elapsed_seconds=time.monotonic()-began,
        limitation='Independent complete saved-action/task/scoring/summary and provenance inspection; unsaved neural outputs and CEM candidate optimization are not replayed.')
    need(not (out/'inspection.json').exists(), 'Inspection already exists'); atomic_json(out/'inspection.json', result); print(json.dumps(result), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True); args = parser.parse_args()
    try: actual(args.run)
    except BaseException as exc:
        failure = BASE/'failures'/args.run
        if not failure.exists():
            failure.mkdir(parents=True); shutil.copy2(__file__, failure/'verify_planning.py')
            atomic_json(failure/'result.json', dict(research_id='U002', name=args.run, status='technical_blocked', error=repr(exc), source_sha256=digest(__file__)))
        raise
