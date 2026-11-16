"""Focused E003 capacity and inherited integrity checks; no research execution."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
import pytest
from go2nav.research import atomic_json, digest, require_record, require_review
from go2nav.eval import research_report as original

BASE = Path('outputs/gated/E003')
OUT = BASE/'software'
OUT.mkdir(exist_ok=False)
START = time.monotonic()
CHECKS = []


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def rejects(call, error, pattern, absent):
    with pytest.raises(error, match=pattern):
        call()
    assert not absent.exists()


def run():
    record, rh = require_record('E003')
    p = record['protocol']
    source = Path('go2nav/eval/research_report.py')
    tests = Path('tests/unit/test_research_report.py')
    assert digest(source) == p['inherited_source_sha256']
    assert digest(tests) == p['inherited_tests_sha256']
    review = Path('outputs/gated/E002/software/review.json')
    assert require_review(review)['status'] == 'passed'
    assert digest(review) == p['inherited_software_review_sha256']
    expected = source.read_text().replace("require_record('E002')", "require_record('E003')")
    expected = expected.replace('FILE_LIMIT, TOTAL_LIMIT = 64 << 20, 256 << 20',
                                'FILE_LIMIT, TOTAL_LIMIT = 64 << 20, 512 << 20')
    expected = expected.replace('from go2nav.research import REPO,',
        'import sys\nsys.path.insert(0, str(Path.cwd()))\n\nfrom go2nav.research import REPO,')
    assert (BASE/'report.py').read_text() == expected
    reporter = module('e003_report', BASE/'report.py')
    assert reporter.FILE_LIMIT == 64*2**20 and reporter.TOTAL_LIMIT == 512*2**20
    assert original.FILE_LIMIT == 64*2**20 and original.TOTAL_LIMIT == 256*2**20
    CHECKS.append('Exact controlled transformation; immutable inherited source/tests/review; fixed64/512MiB limits')

    suite = module('e003_inherited_tests', tests)
    suite.report = reporter
    inherited_cases = 0
    for name in sorted(n for n in vars(suite) if n.startswith('test_')):
        function = getattr(suite, name)
        faults = ['corrupt', 'duplicate', 'missing', 'unlisted', 'unsafe'] if 'fault' in inspect.signature(function).parameters else [None]
        for fault in faults:
            with tempfile.TemporaryDirectory(prefix='e003_test_') as directory, pytest.MonkeyPatch.context() as patch:
                available = dict(tmp_path=Path(directory), monkeypatch=patch, fault=fault)
                function(**{k: available[k] for k in inspect.signature(function).parameters})
            inherited_cases += 1
    assert inherited_cases == 9
    CHECKS.append('All9 inherited report/status/capture-race/historical-source/archive-corruption cases pass against E003')

    with tempfile.TemporaryDirectory(prefix='e003_capacity_') as directory, pytest.MonkeyPatch.context() as patch:
        temporary = Path(directory)
        root, repo = temporary/'runs', temporary/'repo'
        root.mkdir(); repo.mkdir()
        patch.setattr(reporter, 'REPO', repo)
        block = bytes(2**20)
        for i in range(5):
            with (root/f'payload_{i}.log').open('wb') as stream:
                for _ in range(56): stream.write(block)
        rejects(lambda: original.build(root, temporary/'old_report'), ValueError,
                'capture limit exceeded', temporary/'old_report')
        result = reporter.build(root, temporary/'new_report', bundle=True)
        assert result == []
        archive = temporary/'new_report/reproducibility.tar.gz'
        verified = reporter.verify_bundle(archive)
        with tarfile.open(archive) as tar:
            manifest = json.load(tar.extractfile('bundle_manifest.json'))
            for i in range(5):
                name = f'runs/payload_{i}.log'
                assert tar.getmember(name).size == 56*2**20
                assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == digest(root/f'payload_{i}.log') == manifest[name]
        CHECKS.append('Actual280MiB fixture: E002 refuses; E003 captures all5 payloads and verifies full archive')
        boundary_result = dict(payload_bytes=280*2**20, files=5, verified=verified)

        per_file = temporary/'per_file'; per_file.mkdir()
        with (per_file/'oversize.log').open('wb') as stream:
            stream.truncate(65*2**20)
        rejects(lambda: reporter.build(per_file, temporary/'too_large_file'), ValueError,
                'file limit exceeded', temporary/'too_large_file')
        CHECKS.append('65MiB individual file refused before any output directory')

        for i in range(5, 10):
            with (root/f'payload_{i}.log').open('wb') as stream:
                for _ in range(56): stream.write(block)
        rejects(lambda: reporter.build(root, temporary/'too_large_total'), ValueError,
                'capture limit exceeded', temporary/'too_large_total')
        CHECKS.append('Actual560MiB fixture refused at512MiB aggregate limit before any output directory')

    assert digest(source) == p['inherited_source_sha256']
    assert digest(tests) == p['inherited_tests_sha256']
    assert 'torch' not in sys.modules
    ledger = json.loads(Path('outputs/gated/gpu_intervals.json').read_text())
    assert not any(r['research_id'] == 'E003' for r in ledger)
    CHECKS.append('Original reporter/tests unchanged; no Torch import or E003 GPU interval')
    return dict(research_id='E003', status='awaiting_literature_review', research_sha256=rh,
                checks=CHECKS, check_groups=len(CHECKS), inherited_cases=inherited_cases,
                large_fixture=boundary_result, elapsed_seconds=time.monotonic()-START,
                source_sha256={name:digest(BASE/name) for name in ['report.py','check.py']},
                inherited_source_sha256=p['inherited_source_sha256'],
                inherited_tests_sha256=p['inherited_tests_sha256'],
                capacity_observation_sha256=p['capacity_observation_sha256'],
                torch_imported=False, gpu_seconds=0,
                limitation='Synthetic capacity/integrity verification, not the final evidence archive. '
                           'Future actual packaging can still exceed the declared limits and must be separately inspected and reviewed.')


try:
    result = run()
except BaseException as exc:
    atomic_json(OUT/'result.json', dict(research_id='E003', status='technical_blocked',
        error=repr(exc), completed_checks=CHECKS, elapsed_seconds=time.monotonic()-START,
        source_sha256={name:digest(BASE/name) for name in ['report.py','check.py']}))
    raise
atomic_json(OUT/'result.json', result)
print(json.dumps(result), flush=True)
