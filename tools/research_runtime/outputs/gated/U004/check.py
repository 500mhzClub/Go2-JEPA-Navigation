"""Reuse the archive suite and check accounting metadata without GPU execution."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path.cwd()))
import pytest
from go2nav.research import atomic_json, digest, require_record, require_review

BASE = Path('outputs/gated/U004')
OUT = BASE/'software'
OUT.mkdir(exist_ok=False)
START = time.monotonic()
CHECKS = []

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

try:
    record, rh = require_record('U004')
    protocol = record['protocol']
    parent = Path('outputs/gated/E003')
    assert require_review(parent/'software/review.json')['status'] == 'passed'
    assert digest(parent/'software/review.json') == protocol['inherited_software_review_sha256']
    assert digest(parent/'report.py') == protocol['inherited_source_sha256']
    assert require_record('U001')[1] == protocol['authorization_sha256']
    expected = (parent/'report.py').read_text().replace("require_record('E003')", "require_record('U004')")
    expected = expected.replace('new_gpu_hours=hours, allowance_hours=48',
        'new_gpu_hours=hours, allowance_hours=None, budget_policy="U001", authorization_sha256=require_record("U001")[1]')
    expected = expected.replace("f'Additional GPU device time: {hours:.4f} / 48 hours.'",
        "f'Additional GPU device time: {hours:.4f} hours; no GPU-hour limit (U001).'")
    assert (BASE/'report.py').read_text() == expected
    reporter = module('u004_report', BASE/'report.py')
    assert (reporter.FILE_LIMIT, reporter.TOTAL_LIMIT) == (64 << 20, 512 << 20)
    CHECKS.append('Exact controlled metadata-only transformation; reviewed E003 and U001 hashes; unchanged file/archive limits')

    tests = Path('tests/unit/test_research_report.py')
    assert digest(tests) == protocol['inherited_tests_sha256']
    suite = module('u004_inherited_tests', tests)
    suite.report = reporter
    cases = 0
    for name in sorted(n for n in vars(suite) if n.startswith('test_')):
        function = getattr(suite, name)
        faults = ['corrupt', 'duplicate', 'missing', 'unlisted', 'unsafe'] if 'fault' in inspect.signature(function).parameters else [None]
        for fault in faults:
            with tempfile.TemporaryDirectory(prefix='u004_') as directory, pytest.MonkeyPatch.context() as patch:
                args = dict(tmp_path=Path(directory), monkeypatch=patch, fault=fault)
                function(**{key:args[key] for key in inspect.signature(function).parameters})
            cases += 1
    assert cases == 9
    CHECKS.append('All nine inherited review/capture/source/archive fault cases pass')

    with tempfile.TemporaryDirectory(prefix='u004_hours_') as directory:
        root = Path(directory)/'runs'; root.mkdir()
        atomic_json(root/'gpu_intervals.json', [dict(research_id='fixture', start=1, end=180001)])
        reporter.build(root, Path(directory)/'report')
        data = json.loads((Path(directory)/'report/evidence.json').read_text())
        assert data['new_gpu_hours'] == 50 and data['allowance_hours'] is None
        assert data['budget_policy'] == 'U001' and data['authorization_sha256'] == protocol['authorization_sha256']
        text = (Path(directory)/'report/report.md').read_text()
        assert '50.0000 hours; no GPU-hour limit (U001).' in text and '/ 48' not in text
    CHECKS.append('Synthetic 50-hour ledger retained exactly, with null allowance and correct authorization')
    assert 'torch' not in sys.modules
    assert digest(parent/'report.py') == protocol['inherited_source_sha256']
    CHECKS.append('Original reporter unchanged; no Torch import or GPU execution')
    result = dict(research_id='U004', status='awaiting_literature_review', research_sha256=rh,
        checks=CHECKS, check_groups=len(CHECKS), inherited_cases=cases,
        elapsed_seconds=time.monotonic()-START, torch_imported=False, gpu_seconds=0,
        source_sha256={name:digest(BASE/name) for name in ['report.py','check.py']},
        limitation='Reporter metadata and inherited integrity only; final archive and remaining scientific experiments are incomplete.')
except BaseException as exc:
    atomic_json(OUT/'result.json', dict(research_id='U004', status='technical_blocked',
        error=repr(exc), completed_checks=CHECKS, elapsed_seconds=time.monotonic()-START))
    raise
atomic_json(OUT/'result.json', result)
print(json.dumps(result), flush=True)
