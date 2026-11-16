"""Build a local evidence index and reproducibility archive without inventing result reviews."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

import sys
sys.path.insert(0, str(Path.cwd()))

from go2nav.research import REPO, atomic_json, device_hours, digest, require_record, require_review


SOURCE_ROOTS = ('go2nav', 'configs', 'docs', 'tests', 'env')
SOURCE_FILES = ('AGENTS.md', 'README.md', 'pyproject.toml', 'THIRD_PARTY_NOTICES.md', 'LICENSE', 'NOTICE')
SOURCE_SUFFIXES = {'.py', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.sh'}
RUN_SUFFIXES = {'.json', '.jsonl', '.npz', '.txt', '.log', '.py', '.sh'}
FILE_LIMIT, TOTAL_LIMIT = 64 << 20, 512 << 20


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned_sources(revision):
    """Capture exact Git blobs; export-ignore/subst must not change executed code."""
    if not re.fullmatch('[0-9a-fA-F]{7,40}', revision):
        raise ValueError('Pinned source must name a hexadecimal commit')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', revision+'^{commit}'],
                                     cwd=REPO, text=True).strip()
    listing = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', commit, '--',
                                      *SOURCE_ROOTS, *SOURCE_FILES], cwd=REPO)
    entries = []
    for entry in listing.split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        name = name.decode()
        if kind == b'blob' and mode in (b'100644', b'100755') and (
                Path(name).suffix in SOURCE_SUFFIXES or name in SOURCE_FILES):
            entries.append((oid, name))
    raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=REPO,
                                  input=b''.join(oid+b'\n' for oid, _ in entries))
    stream, files = io.BytesIO(raw), {}
    for expected_oid, name in entries:
        oid, kind, size = stream.readline().split()
        if oid != expected_oid or kind != b'blob':
            raise ValueError('Unexpected Git blob response')
        data = stream.read(int(size))
        if len(data) != int(size) or stream.read(1) != b'\n':
            raise ValueError('Incomplete Git blob response')
        files[f'pinned_sources/{commit}/{name}'] = data
    return commit, files


def verify_bundle(path):
    """Verify every payload, without extracting archive paths to the filesystem."""
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        for m in members:
            p = PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe or nonregular archive member')
        manifest = json.load(tar.extractfile('bundle_manifest.json'))
        if set(names) != set(manifest) | {'bundle_manifest.json'}:
            raise ValueError('Archive payload does not match manifest')
        for name, expected in manifest.items():
            if hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() != expected:
                raise ValueError(f'Archive digest mismatch: {name}')
    return dict(files=len(manifest), archive_sha256=digest(path))


def build(root, out, bundle=False, source_commits=()):
    require_record('U004')
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise ValueError('Evidence root does not exist')
    if out.exists():
        raise FileExistsError('Use a new report directory; existing evidence is preserved')
    # Read each selected file once. Evidence, review checks and archive hashes all
    # use this capture, even if a live log changes while the archive is written.
    payload, captured_bytes = {}, 0

    def capture(name, data):
        nonlocal captured_bytes
        if name in payload:
            raise ValueError(f'Duplicate payload: {name}')
        if len(data) > FILE_LIMIT or captured_bytes+len(data) > TOTAL_LIMIT:
            raise ValueError(f'Bundle capture limit exceeded: {name}')
        payload[name] = data
        captured_bytes += len(data)

    report_dirs = {p.parent for p in root.rglob('bundle_manifest.json')}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or path.is_relative_to(out)
                or any(path.is_relative_to(p) for p in report_dirs)
                or relative.parts[0] in ('sources', 'report', 'tools') or 'corpus' in relative.parts
                or path.suffix not in RUN_SUFFIXES):
            continue
        if path.stat().st_size > FILE_LIMIT:
            raise ValueError(f'Bundle file limit exceeded: {relative}')
        capture('runs/'+relative.as_posix(), path.read_bytes())

    evidence = []
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for name, raw in payload.items():
            if PurePosixPath(name).name != 'result.json':
                continue
            data, review = json.loads(raw), None
            review_name = str(PurePosixPath(name).with_name('review.json'))
            if review_name in payload:
                (staging/'result.json').write_bytes(raw)
                (staging/'review.json').write_bytes(payload[review_name])
                review = require_review(staging/'review.json', staging/'result.json')
            evidence.append(dict(run=str(PurePosixPath(name).parent.relative_to('runs')),
                                 research_id=data.get('research_id'), result_sha256=sha(raw),
                                 result_status=data.get('status', 'not_reported'),
                                 status=review['status'] if review else 'awaiting_literature_review',
                                 next_step=review['next_step'] if review else
                                 'Literature review required before a dependent experiment'))
    ledger = payload.get('runs/gpu_intervals.json')
    hours = device_hours(json.loads(ledger)) if ledger else 0.
    capture('evidence.json', encoded(dict(evidence=evidence, new_gpu_hours=hours, allowance_hours=None, budget_policy="U001", authorization_sha256=require_record("U001")[1])))
    lines = ['# P25/P26 evidence', '', f'Additional GPU device time: {hours:.4f} hours; no GPU-hour limit (U001).', '',
             'This is a snapshot of available evidence. Missing experiments remain incomplete; '
             'large datasets and checkpoints remain local at their recorded paths and digests.', '',
             '| Run | Research | Result status | Reviewed status | Next step |', '|---|---|---|---|---|']
    for row in evidence:
        lines.append('| '+' | '.join(str(row[k]).replace('|', '/').replace('\n', ' ')
                                     for k in ('run', 'research_id', 'result_status', 'status', 'next_step'))+' |')
    capture('report.md', ('\n'.join(lines)+'\n').encode())
    if bundle:
        paths = [p for base in SOURCE_ROOTS for p in (REPO/base).rglob('*')
                 if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts
                 and p.suffix in SOURCE_SUFFIXES]
        paths += [REPO/p for p in SOURCE_FILES if (REPO/p).is_file()]
        for path in sorted(paths):
            capture(str(path.relative_to(REPO)), path.read_bytes())
        commits = []
        for revision in source_commits:
            commit, files = pinned_sources(revision)
            if commit in commits:
                continue
            commits.append(commit)
            for name, raw in files.items():
                capture(name, raw)
        capture('source_versions.json', encoded(dict(pinned_commits=commits,
                current_source='Working tree bytes in this bundle; file manifest is authoritative')))
        capture('bundle_manifest.json', encoded({name: sha(raw) for name, raw in payload.items()}))
    out.mkdir(parents=True, exist_ok=False)
    for name in ('evidence.json', 'report.md'):
        (out/name).write_bytes(payload[name])
    if bundle:
        partial = out/'reproducibility.tar.gz.partial'
        with tarfile.open(partial, 'w:gz') as tar:
            for name, raw in sorted(payload.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(raw), 0o644
                tar.addfile(member, io.BytesIO(raw))
        verified = verify_bundle(partial)
        partial.replace(out/'reproducibility.tar.gz')
        (out/'bundle_manifest.json').write_bytes(payload['bundle_manifest.json'])
        atomic_json(out/'verification.json', verified)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bundle', action='store_true')
    p.add_argument('--source-commit', action='append', default=[], help='Historical commit to include as exact source blobs')
    a = p.parse_args()
    print(json.dumps(build(a.root, a.out, a.bundle, a.source_commit), indent=1))


if __name__ == '__main__':
    main()
