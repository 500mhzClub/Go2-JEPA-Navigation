"""Custody guard: this public repository must never contain sealed benchmark material, even by name."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEALED = re.compile(r'(^|/)(sealed_test\.json$|sealed/|sealed_[^/]*/)')


def test_no_sealed_paths_tracked():
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    assert not [p for p in tracked if SEALED.search(p)]


MACHINE_PATH = re.compile(r'/home/[\w.-]+/|/mnt/\w+/')
TEXT = ('.py', '.md', '.json', '.yaml', '.yml', '.txt', '.sh', '.toml')


def test_no_machine_paths_tracked():
    """No tracked text file names a home directory or a mount point: original files are named by their root
    (`tools/porting/original.py`), shipped files by their artefact name."""
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    offenders = [p for p in tracked if p.endswith(TEXT) and MACHINE_PATH.search((REPO/p).read_text(errors='ignore'))]
    assert not offenders, offenders
