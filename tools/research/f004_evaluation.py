"""Serial F004 numerical/prediction execution and reviewed-input CPU scoring."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import psutil
import torch

from f004_forecast_evidence import (BASE, ROOT, base_binding, bind, implementation_sources,
                                    prediction_binding, read, reviewed, scoring_inputs)
from f004_runtime import close_interval, fresh_paths, gpu_device, gpu_idle, need, supervision
from go2nav.data.paired_forecasts import Cache
from go2nav.eval.paired_forecast_inference import (FLAVORS, MODELS, NUMERIC_POLICY, InputView,
                                                 load_model, predict, read_artifact, write_artifact)
from go2nav.eval.paired_forecast_metrics import evaluate
from go2nav.research import atomic_json, digest, require_record, verify_files
from go2nav.training.object_jepa import Rasters

EXECUTION_SOURCES = ('tools/research/f004_evaluation.py', 'tools/research/f004_evaluation_checks.py',
                     'go2nav/eval/paired_forecast_metrics.py')


def execution_sources():
    return dict(implementation_sources(), **{p: digest(ROOT/p) for p in EXECUTION_SOURCES})


def require_execution_software(bound):
    _, rh = require_record('F004')
    _, result = reviewed(bound, BASE/'evaluation_execution_software/review.json',
                          BASE/'evaluation_execution_software/result.json')
    need(result['record_sha256'] == rh and result['source_sha256'] == execution_sources(), 'Stale evaluation execution software')
    pre = BASE/'evaluation_execution_preimplementation.json'
    need(result['preimplementation_sha256'] == digest(pre), 'Changed execution consultation')
    bind(bound, pre)
    require_record('U001')
    bind(bound, ROOT/'docs/research/records/U001.json')


def stage_evidence(name, phase):
    need(name in MODELS and phase in ('numerical', 'prediction'), 'Unplanned evaluation identity')
    binding, bound = base_binding(name) if phase == 'numerical' else prediction_binding(name)
    require_execution_software(bound)
    return binding, bound


def validate_pre(pre, identity, binding, bound, code, commit):
    need(pre['run_identity'] == identity and pre['binding'] == binding, 'Wrong preconsultation identity/binding')
    need(pre['inputs_sha256'] == bound and pre['source_sha256'] == code
         and pre['source_commit'] == commit, 'Stale/incomplete preconsultation evidence/source')
    need(pre.get('consulted_at') and pre.get('sources') and pre.get('consequences'), 'Missing substantive consultation')
    for source in pre['sources']:
        need(all(source.get(k) for k in ('url', 'version', 'sections', 'consequence')), 'Incomplete consultation source')


def admit(ledger, out, identity):
    need(identity['model'] in MODELS and identity['phase'] in ('numerical', 'prediction'), 'Unplanned interval identity')
    ledger = Path(ledger)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read(ledger)
        need(all(r['end'] is not None for r in rows), 'Open GPU interval')
        interval = dict(id=uuid.uuid4().hex, research_id='F004', budget_policy='U001',
                        **identity, out=str(out), start=time.time(), end=None)
        rows.append(interval)
        atomic_json(ledger, rows)
    return interval


def child_admission(path):
    admission = read(path)
    need(admission['cwd'] == str(ROOT), 'Wrong child checkout')
    identity = admission['run_identity']
    binding, bound = stage_evidence(identity['model'], identity['phase'])
    need(admission['binding'] == binding and admission['inputs_sha256'] == bound
         and admission['source_sha256'] == execution_sources(), 'Stale child binding/source/evidence')
    parent = psutil.Process(os.getppid())
    need(parent.pid == admission['supervisor_pid'] and parent.create_time() == admission['supervisor_created']
         and Path(parent.cwd()).resolve() == ROOT.resolve(), 'Not the admitted supervisor child')
    rows = read(admission['ledger'])
    need([r for r in rows if r['id'] == admission['interval']['id']] == [admission['interval']]
         and admission['interval']['end'] is None, 'Unadmitted child interval')
    verify_files(ROOT, bound)
    verify_files(ROOT, admission['source_sha256'])
    return admission


def compute(admission, view, raster_factory, device_factory=gpu_device):
    """Dispatch only the fixed deployed numerical/prediction modes; no CLI CPU fallback."""
    identity, binding = admission['run_identity'], admission['binding']
    phase = identity['phase']
    need(phase in ('numerical', 'prediction') and identity['model'] == binding['model']['name'], 'Wrong computation identity')
    need(binding['numerical_policy'] == NUMERIC_POLICY, 'Changed numerical policy')
    if phase == 'prediction':
        need(binding.get('numerical_result_sha256') and binding.get('numerical_review_sha256')
             and binding.get('prediction_policy') == 'GPU_float32_math_batch32', 'Missing reviewed numerical admission')
    device = device_factory()
    need(torch.device(device).type == 'cuda', 'GPU evaluation required; CPU fallback disabled')
    operation = Path(admission['operation_out'])
    retained = operation/'completed_flavors'
    retained.mkdir(exist_ok=False)
    values = {}
    keys = None
    for family in (('cpu', 'gpu') if phase == 'numerical' else ('gpu',)):
        target = torch.device('cpu') if family == 'cpu' else device
        rasters = raster_factory(target)
        model = load_model(ROOT/binding['model']['path'], binding['model'], target)
        for batch in ((1, 32) if phase == 'numerical' else (32,)):
            flavor = f'{family}{batch}'
            callback = lambda completed, total: atomic_json(operation/'progress.json',
                dict(flavor=flavor, completed=completed, total=total, at=time.time()))
            prediction, current_keys = predict(model, view, rasters, batch,
                scenes=NUMERIC_POLICY['scenes'] if phase == 'numerical' else None, progress=callback)
            need(keys is None or np.array_equal(keys, current_keys), 'Numerical flavor changed case ordering')
            keys = current_keys
            for frame, array in prediction.items():
                # Keep completed flavors if a later device/rollout fails or stalls.
                with (retained/f'{flavor}_{frame}.npy').open('xb') as file:
                    np.save(file, array, allow_pickle=False)
                values[f'{flavor}_{frame}'] = array
            print(json.dumps(dict(flavor=flavor, trajectories=len(keys), status='completed')), flush=True)
        del model, rasters
    arrays = values if phase == 'numerical' else {'world': values['gpu32_world']}
    result = write_artifact(admission['science_out'], arrays, keys, binding, phase)
    return result


def child_main(path):
    admission = child_admission(path)
    torch.set_num_threads(2)
    view = InputView(Cache(BASE/'data', 'val').arrays)
    layouts = np.unique(view.arrays['maze']).tolist()
    rasters = lambda device: Rasters(ROOT/'outputs/data/ring_corpus_v1/objects/rasters', layouts, device)
    result = compute(admission, view, rasters)
    print(json.dumps(dict(phase=result['phase'], status=result['status'])), flush=True)


def run(name, phase, pre_path):
    need(Path.cwd().resolve() == ROOT.resolve(), 'Run from intended checkout')
    identity = dict(model=name, phase=phase)
    binding, bound = stage_evidence(name, phase)
    code = execution_sources()
    need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True), 'Source checkout is dirty')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pre = read(pre_path)
    validate_pre(pre, identity, binding, bound, code, commit)
    out = BASE/'evaluation'/phase/name
    fresh_paths(out)
    common = supervision()
    _, _, owned = common.inherited()
    gpu_report = subprocess.run(['amd-smi', 'process', '--json'], capture_output=True, text=True, check=True, timeout=10)
    gpu_idle(json.loads(gpu_report.stdout))
    ledger = ROOT/'outputs/gated/gpu_intervals.json'
    operation, science = out/'operations', out/'science'
    supervisor = psutil.Process()
    supervisor_created = supervisor.create_time()
    pre_hash = digest(pre_path)
    interval = admit(ledger, out, identity)
    child = process = created = None
    actions, errors = [], []
    stalled, unchanged = False, False
    began = time.monotonic()
    admission = dict(research_id='F004', run_identity=identity, binding=binding, inputs_sha256=bound,
                     source_sha256=code, source_commit=commit, preconsultation=pre,
                     preconsultation_sha256=pre_hash, cwd=str(ROOT), science_out=str(science),
                     operation_out=str(operation), ledger=str(ledger), interval=interval,
                     supervisor_pid=supervisor.pid, supervisor_created=supervisor_created,
                     stall_seconds=1800, gpu_hours_limit=None, admitted_at=datetime.now(timezone.utc).isoformat())
    argv = [sys.executable, str(ROOT/'tools/research/f004_evaluation.py'), 'child', '--admission', str(operation/'admission.json')]
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTORCH_TUNABLEOP_ENABLED='0')
    try:
        operation.mkdir(parents=True, exist_ok=False)
        atomic_json(operation/'admission.json', admission)
        (operation/'process_before.json').write_text(gpu_report.stdout)
        with (operation/'evaluation.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            created = process.create_time()
            atomic_json(operation/'child.json', dict(pid=child.pid, creation_time=created, argv=argv, cwd=str(ROOT)))
            stalled = common.wait_progress(child, process, created, argv, ROOT,
                [operation/'progress.json', operation/'evaluation.log'], actions,
                heartbeat=lambda quiet: atomic_json(operation/'monitor_latest.json', dict(at=time.time(),
                    pid=child.pid, creation_time=created, seconds_without_progress=quiet)))
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                owned.stop_owned(child, process, created, argv, ROOT, actions)
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            close_interval(ledger, interval, child)
        except BaseException as exc:
            errors.append(repr(exc))
    terminal = child is None or child.poll() is not None
    code_return = None if child is None else child.returncode
    artifacts, raw = {}, {}
    if terminal:
        try:
            verify_files(ROOT, bound)
            verify_files(ROOT, code)
            need(stage_evidence(name, phase) == (binding, bound), 'Terminal evaluation evidence changed')
            need(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit, 'Pinned source moved')
            view = InputView(Cache(BASE/'data', 'val').arrays)
            keys = view.keys(NUMERIC_POLICY['scenes'] if phase == 'numerical' else None)
            raw, _ = read_artifact(science, binding, keys, phase)
            unchanged = True
            artifacts = {str(p.relative_to(science)): digest(p) for p in science.rglob('*') if p.is_file()}
        except BaseException as exc:
            errors.append(repr(exc))
    okay = code_return == 0 and terminal and not errors and not stalled and unchanged
    result = dict(research_id='F004', phase='operations', run_identity=identity,
                  record_sha256=binding['record_sha256'], status='awaiting_literature_review' if okay else 'technical_blocked',
                  returncode=code_return, child_terminal=terminal, inputs_unchanged=unchanged, stalled=stalled,
                  actions=actions, errors=errors, interval_id=interval['id'],
                  admission_sha256=digest(operation/'admission.json') if (operation/'admission.json').is_file() else None,
                  elapsed_seconds=time.monotonic()-began, child_artifact_sha256=artifacts,
                  scientific_status=raw.get('status'), source_sha256=code,
                  next_step='Review actual science and operations separately; no automatic prediction/scoring/successor.')
    atomic_json(operation/'result.json', result)
    print(json.dumps(dict(run_identity=identity, status=result['status'], scientific_status=raw.get('status'), errors=errors)), flush=True)
    need(okay, 'Evaluation operations require technical review')


def score(pre_path):
    data, predictions, bound = scoring_inputs()
    require_execution_software(bound)
    _, rh = require_record('F004')
    binding = dict(record_sha256=rh, models=list(MODELS), data_result_sha256=digest(BASE/'data/result.json'))
    code = execution_sources()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True), 'Scoring source is dirty')
    pre = read(pre_path)
    validate_pre(pre, dict(model='all', phase='scoring'), binding, bound, code, commit)
    out = BASE/'evaluation/scoring'
    fresh_paths(out)
    out.mkdir(parents=True)
    began = time.monotonic()
    atomic_json(out/'manifest.json', dict(binding=binding, inputs_sha256=bound, source_sha256=code,
                source_commit=commit, preconsultation=pre, preconsultation_sha256=digest(pre_path)))
    report = evaluate(data, predictions)
    verify_files(ROOT, bound)
    verify_files(ROOT, code)
    need(not torch.cuda.is_initialized(), 'CPU scoring initialized GPU')
    result = dict(research_id='F004', phase='scoring', record_sha256=rh, status='awaiting_literature_review',
                  binding=binding, source_sha256=code, inputs_sha256=bound, source_commit=commit,
                  elapsed_seconds=time.monotonic()-began, report=report,
                  next_step='Consult primary literature and review actual physical/reaction evidence, coverage and uncertainty; no automatic benefit claim.')
    atomic_json(out/'result.json', result)
    print(json.dumps(dict(path=str(out/'result.json'), sha256=digest(out/'result.json'), status=result['status'])))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--model', choices=MODELS, required=True)
    run_parser.add_argument('--phase', choices=('numerical', 'prediction'), required=True)
    run_parser.add_argument('--preconsultation', required=True)
    child_parser = commands.add_parser('child')
    child_parser.add_argument('--admission', required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--preconsultation', required=True)
    args = ap.parse_args()
    if args.command == 'child':
        child_main(args.admission)
    elif args.command == 'score':
        score(args.preconsultation)
    else:
        run(args.model, args.phase, args.preconsultation)


if __name__ == '__main__':
    main()
