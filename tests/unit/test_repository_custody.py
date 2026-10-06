"""Custody guard: this public repository must never contain sealed benchmark material, even by name."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEALED = re.compile(r'(^|/)(sealed_test\.json$|sealed/|sealed_[^/]*/)')


def test_no_sealed_paths_tracked():
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    assert not [p for p in tracked if SEALED.search(p)]


def test_no_absolute_home_paths_in_code():
    tracked = subprocess.run(['git', 'ls-files', '*.py'], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    offenders = [p for p in tracked if '/home/' in (REPO/p).read_text() and not p.startswith('tests/')]
    assert not offenders, offenders
