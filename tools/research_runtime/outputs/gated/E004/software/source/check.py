"""Verify bounded evidence capacity and P26 metadata selection; no model execution."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import shutil
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
import pytest
from go2nav.research import atomic_json, digest, require_record, require_review, verify_files

BASE = Path('outputs/gated/E004')
OUT = BASE/'software'
OUT.mkdir(exist_ok=False)
(OUT/'source').mkdir()
for name in ('report.py', 'check.py'):
    shutil.copy2(BASE/name, OUT/'source'/name)
START = time.monotonic()
CHECKS = []
LEDGER = Path('outputs/gated/gpu_intervals.json')
LEDGER_BEFORE = digest(LEDGER)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def rejects(call, error, pattern, absent):
    with pytest.raises(error, match=pattern):
        call()
    assert not absent.exists()


try:
    record, rh = require_record('E004')
    protocol = record['protocol']
    verify_files(Path.cwd(), protocol['source_sha256'])
    assert require_review('outputs/gated/U004/software/review.json')['status'] == 'passed'
    expected = Path('outputs/gated/U004/report.py').read_text()
    for before, after in json.loads((BASE/'controlled_changes.json').read_text()):
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    assert (BASE/'report.py').read_text() == expected
    reporter = module('e004_report', BASE/'report.py')
    original = module('e004_previous_report', Path('outputs/gated/U004/report.py'))
    assert (reporter.FILE_LIMIT, reporter.TOTAL_LIMIT) == (64 << 20, 2 << 30)
    assert (original.FILE_LIMIT, original.TOTAL_LIMIT) == (64 << 20, 512 << 20)
    CHECKS.append('Exact four substitutions; all declared source/review hashes verify; fixed64MiB individual/2GiB aggregate limits')

    suite = module('e004_inherited_tests', Path('tests/unit/test_research_report.py'))
    suite.report = reporter
    cases = 0
    for name in sorted(n for n in vars(suite) if n.startswith('test_')):
        function = getattr(suite, name)
        faults = ['corrupt', 'duplicate', 'missing', 'unlisted', 'unsafe'] if 'fault' in inspect.signature(function).parameters else [None]
        for fault in faults:
            with tempfile.TemporaryDirectory(prefix='e004_integrity_') as directory, pytest.MonkeyPatch.context() as patch:
                args = dict(tmp_path=Path(directory), monkeypatch=patch, fault=fault)
                function(**{key: args[key] for key in inspect.signature(function).parameters})
            cases += 1
    assert cases == 9
    CHECKS.append('All nine inherited status/review/capture-race/historical-blob/archive-corruption cases pass')

    with tempfile.TemporaryDirectory(prefix='e004_p26_') as directory, pytest.MonkeyPatch.context() as patch:
        temporary = Path(directory)
        root, repo = temporary/'runs', temporary/'repo'
        root.mkdir(); repo.mkdir()
        patch.setattr(reporter, 'REPO', repo)
        patch.setattr(original, 'REPO', repo)
        corpus = root/'D001/corpus'
        export = corpus/'track_run'
        export.mkdir(parents=True)
        atomic_json(export/'result.json', dict(research_id='D001', status='awaiting_literature_review', fixture=True))
        atomic_json(export/'review.json', dict(research_id='D001', result='result.json', result_sha256=digest(export/'result.json'),
            reviewed_at='synthetic fixture', status='passed', sources=[dict(url='fixture', version='fixture', sections='fixture', consequence='fixture')],
            measurements='fixture only', uncertainty='no actual export', explanations=['fixture'], next_step='fixture only'))
        atomic_json(export/'manifest.json', dict(fixture=True))
        (export/'diagnostic.npz').write_bytes(b'small synthetic diagnostic')
        for name in ('index.json', 'prepare_manifest.json', 'tracks_index.json', 'tracks_config.json'):
            atomic_json(corpus/name, dict(fixture=name))
        atomic_json(corpus/'train/unit.json', dict(unit='fixture', external_array='unit.npz'))
        (corpus/'train/unit.npz').write_bytes(b'bulk array excluded')
        (corpus/'packed').mkdir()
        (corpus/'packed/train_pose.npy').write_bytes(b'bulk array excluded')
        (export/'model.pt').write_bytes(b'checkpoint excluded')
        atomic_json(root/'other/corpus/result.json', dict(status='unrelated corpus excluded'))
        old = original.build(root, temporary/'old', bundle=True)
        assert old == []
        rows = reporter.build(root, temporary/'new', bundle=True)
        assert len(rows) == 1 and rows[0]['run'] == 'D001/corpus/track_run'
        assert rows[0]['status'] == 'passed' and rows[0]['result_status'] == 'awaiting_literature_review'
        archive = temporary/'new/reproducibility.tar.gz'
        reporter.verify_bundle(archive)
        selected = [export/'result.json', export/'review.json', export/'manifest.json', export/'diagnostic.npz']
        selected += [corpus/name for name in ('index.json', 'prepare_manifest.json', 'tracks_index.json', 'tracks_config.json', 'train/unit.json')]
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
            for path in selected:
                assert tar.extractfile('runs/'+path.relative_to(root).as_posix()).read() == path.read_bytes()
            assert not any(name.endswith(('.pt', '.npy', '/train/unit.npz')) or '/other/corpus/' in name for name in names)
        (export/'result.json').write_text('{"research_id":"D001","changed":true}')
        rejects(lambda: reporter.build(root, temporary/'stale'), ValueError, 'stale', temporary/'stale')
        CHECKS.append('Original reporter omits export fixture; E004 includes reviewed export and all nine metadata/diagnostic files, preserves bulk exclusions and refuses stale export review')

    with tempfile.TemporaryDirectory(prefix='e004_capacity_') as directory, pytest.MonkeyPatch.context() as patch:
        temporary = Path(directory)
        root, repo = temporary/'runs', temporary/'repo'
        root.mkdir(); repo.mkdir()
        patch.setattr(reporter, 'REPO', repo)
        patch.setattr(original, 'REPO', repo)
        atomic_json(root/'gpu_intervals.json', [dict(research_id='fixture', start=1, end=180001)])
        # Sparse files still yield their full logical bytes to the unchanged
        # read-once capture and archive hash checks.
        for index in range(9):
            with (root/f'payload_{index:02}.log').open('wb') as stream:
                stream.truncate(64 << 20)
        rejects(lambda: original.build(root, temporary/'old'), ValueError,
                'capture limit exceeded', temporary/'old')
        assert reporter.build(root, temporary/'new', bundle=True) == []
        archive = temporary/'new/reproducibility.tar.gz'
        verified = reporter.verify_bundle(archive)
        with tarfile.open(archive) as tar:
            manifest = json.load(tar.extractfile('bundle_manifest.json'))
            for index in range(9):
                name = f'runs/payload_{index:02}.log'
                assert tar.getmember(name).size == 64 << 20
                assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == manifest[name] == digest(root/f'payload_{index:02}.log')
        evidence = json.loads((temporary/'new/evidence.json').read_text())
        assert evidence['new_gpu_hours'] == 50 and evidence['allowance_hours'] is None
        assert evidence['budget_policy'] == 'U001' and evidence['authorization_sha256'] == require_record('U001')[1]
        boundary = dict(payload_bytes=576 << 20, files=9, verified=verified)
        CHECKS.append('Actual576MiB fixture: old512MiB limit refuses; all nine64MiB files captured and independently verified;50hour accounting has null allowance')
        per_file = temporary/'per_file'
        per_file.mkdir()
        with (per_file/'oversize.log').open('wb') as stream:
            stream.truncate(65 << 20)
        rejects(lambda: reporter.build(per_file, temporary/'file_overflow'), ValueError,
                'file limit exceeded', temporary/'file_overflow')
        for index in range(9, 33):
            with (root/f'payload_{index:02}.log').open('wb') as stream:
                stream.truncate(64 << 20)
        rejects(lambda: reporter.build(root, temporary/'total_overflow'), ValueError,
                'capture limit exceeded', temporary/'total_overflow')
        CHECKS.append('Actual65MiB individual file and2112MiB aggregate fixture refuse before output directory creation')

    verify_files(Path.cwd(), protocol['source_sha256'])
    assert digest(LEDGER) == LEDGER_BEFORE and 'torch' not in sys.modules
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak_rss_bytes < 8 << 30
    CHECKS.append('Inherited sources and shared ledger unchanged; no Torch/GPU; peak RSS below declared8GiB')
    result = dict(research_id='E004', research_sha256=rh, status='awaiting_literature_review',
        checks=CHECKS, check_groups=len(CHECKS), inherited_cases=cases, large_fixture=boundary,
        elapsed_seconds=time.monotonic()-START, peak_rss_bytes=peak_rss_bytes,
        source_sha256={name: digest(BASE/name) for name in ('report.py', 'check.py')},
        controlled_changes_sha256=digest(BASE/'controlled_changes.json'),
        torch_imported=False, gpu_seconds=0,
        limitation='Synthetic capacity, integrity and P26 evidence selection only. No actual final archive, completed P26 export or final scientific coverage is established. Actual selected evidence may still exceed either bound.')
    atomic_json(OUT/'result.json', result)
    print(json.dumps(result), flush=True)
except BaseException as exc:
    atomic_json(OUT/'result.json', dict(research_id='E004', status='technical_blocked',
        error=repr(exc), completed_checks=CHECKS, elapsed_seconds=time.monotonic()-START))
    raise
