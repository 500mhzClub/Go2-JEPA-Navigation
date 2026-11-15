import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from go2nav.eval import research_report as report


def evidence(root, *, reviewed=False):
    run = root/'trial'
    run.mkdir(parents=True)
    result = run/'result.json'
    result.write_text('{"research_id":"fixture", "status":"passed"}')
    if reviewed:
        review = dict(research_id='fixture', result='result.json',
                      result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                      reviewed_at='fixture', sources=[dict(url='fixture', version='fixture',
                      sections='fixture', consequence='fixture')], measurements='fixture',
                      uncertainty='fixture', explanations=['fixture'], next_step='fixture', status='passed')
        (run/'review.json').write_text(json.dumps(review))
    return run


def test_unreviewed_pass_is_never_reported_as_reviewed(tmp_path):
    root = tmp_path/'runs'
    evidence(root)
    rows = report.build(root, tmp_path/'out')
    assert rows[0]['result_status'] == 'passed'
    assert rows[0]['status'] == 'awaiting_literature_review'
    assert 'awaiting_literature_review' in (tmp_path/'out/report.md').read_text()


def test_stale_review_and_existing_output_fail_without_overwrite(tmp_path):
    root = tmp_path/'runs'
    run = evidence(root, reviewed=True)
    (run/'result.json').write_text('{"status":"measured_negative"}')
    with pytest.raises(ValueError, match='stale'):
        report.build(root, tmp_path/'out')
    assert not (tmp_path/'out').exists()
    (tmp_path/'out').mkdir()
    keep = tmp_path/'out/keep.txt'
    keep.write_text('previous evidence')
    with pytest.raises(FileExistsError):
        report.build(root, tmp_path/'out')
    assert keep.read_text() == 'previous evidence'


def test_archive_contains_captured_bytes_despite_live_file_change(tmp_path, monkeypatch):
    repo, root, out = tmp_path/'repo', tmp_path/'runs', tmp_path/'report'
    repo.mkdir()
    monkeypatch.setattr(report, 'REPO', repo)
    run = evidence(root, reviewed=True)
    (run/'preflight.py').write_text('print("diagnostic")\n')
    trace = run/'stacks.txt'
    trace.write_text('captured stack')
    (run/'checkpoint.pt').write_bytes(b'large model omitted')
    (root/'tools').mkdir()
    (root/'tools/install.json').write_text('{}')
    (root/'previous').mkdir()
    (root/'previous/bundle_manifest.json').write_text('{}')
    (root/'previous/result.json').write_text('{"status":"obsolete"}')
    real_add = tarfile.TarFile.addfile

    def mutate_then_add(self, info, fileobj=None):
        if info.name == 'runs/trial/stacks.txt':
            trace.write_text('new live stack')
        return real_add(self, info, fileobj)

    monkeypatch.setattr(tarfile.TarFile, 'addfile', mutate_then_add)
    rows = report.build(root, out, bundle=True)
    assert len(rows) == 1 and rows[0]['status'] == 'passed'
    report.verify_bundle(out/'reproducibility.tar.gz')
    with tarfile.open(out/'reproducibility.tar.gz') as tar:
        assert tar.extractfile('runs/trial/stacks.txt').read() == b'captured stack'
        assert tar.extractfile('runs/trial/preflight.py').read() == b'print("diagnostic")\n'
        assert not any('checkpoint.pt' in n or '/tools/' in n or '/previous/' in n for n in tar.getnames())
    assert trace.read_text() == 'new live stack'


def test_pinned_source_is_historical_blob_not_worktree_or_export_filter(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    (repo/'go2nav').mkdir(parents=True)
    monkeypatch.setattr(report, 'REPO', repo)
    code = repo/'go2nav/train.py'
    code.write_text('original training code\n')
    (repo/'.gitattributes').write_text('go2nav/train.py export-ignore\n')
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=repo, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    code.write_text('changed current code\n')
    root, out = tmp_path/'runs', tmp_path/'out'
    evidence(root)
    report.build(root, out, bundle=True, source_commits=[commit[:8]])
    with tarfile.open(out/'reproducibility.tar.gz') as tar:
        assert tar.extractfile(f'pinned_sources/{commit}/go2nav/train.py').read() == b'original training code\n'
        assert tar.extractfile('go2nav/train.py').read() == b'changed current code\n'


@pytest.mark.parametrize('fault', ['corrupt', 'duplicate', 'missing', 'unlisted', 'unsafe'])
def test_archive_verifier_rejects_changed_payload_or_members(tmp_path, fault):
    data = b'evidence'
    manifest = {'data.txt': hashlib.sha256(data).hexdigest()}
    entries = [('bundle_manifest.json', json.dumps(manifest).encode())]
    if fault != 'missing':
        entries.append(('data.txt', b'changed' if fault == 'corrupt' else data))
    if fault in ('duplicate', 'unlisted', 'unsafe'):
        entries.append(({'duplicate': 'data.txt', 'unlisted': 'extra.txt', 'unsafe': '../escape'}[fault], data))
    path = tmp_path/'bad.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, raw in entries:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    with pytest.raises(ValueError):
        report.verify_bundle(path)
