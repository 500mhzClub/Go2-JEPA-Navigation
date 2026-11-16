"""Independent read-only audit of the E002 interim archive after A017."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time


case = Path(__file__).resolve().parent
root = Path.cwd()
output = root / 'outputs/gated/report_checkpoint_progress_20261115'
archive = output / 'reproducibility.tar.gz'
started = time.monotonic()
matched_files = []
matched_blobs = []

with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    names = [member.name for member in members]
    assert len(names) == len(set(names))
    assert all(member.isfile() and not PurePosixPath(member.name).is_absolute()
               and '..' not in PurePosixPath(member.name).parts for member in members)
    manifest = json.load(tar.extractfile('bundle_manifest.json'))
    assert set(names) == set(manifest) | {'bundle_manifest.json'}
    for name, expected in manifest.items():
        assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected, name

    for name in names:
        if name.startswith('runs/A017/'):
            local = root / 'outputs/gated' / name.removeprefix('runs/')
            assert tar.extractfile(name).read() == local.read_bytes(), name
            matched_files.append(name)
    for name in [
        'docs/research/CHECKPOINT_PROGRESS_2026-11-15.md',
        'docs/research/records/A017.json',
        'runs/A014/run_original.py',
        'runs/A014/continuation_plan.json',
        'runs/A014/runs/l1_n1_s0/execution.json',
        'runs/A015/state_contract.py',
        'runs/A015/verify_final.py',
        'runs/A015/software/review.json',
        'runs/A016/supervision.py',
        'runs/A016/run_a003.py',
        'runs/A016/software/review.json',
    ]:
        local = (root / 'outputs/gated' / name.removeprefix('runs/')) if name.startswith('runs/') else root / name
        assert tar.extractfile(name).read() == local.read_bytes(), name
        matched_files.append(name)

    for revision, paths in [
        ('7dfc837', ['go2nav/training/balls.py']),
        ('161fe8b', ['go2nav/eval/balls_inference.py']),
        ('98e6fcf', ['go2nav/eval/slot_measures.py', 'go2nav/models/slots.py', 'go2nav/training/slots.py']),
    ]:
        commit = subprocess.check_output(['git', 'rev-parse', revision], text=True).strip()
        for path in paths:
            name = f'pinned_sources/{commit}/{path}'
            expected = subprocess.check_output(['git', 'cat-file', 'blob', f'{commit}:{path}'])
            assert tar.extractfile(name).read() == expected, name
            matched_blobs.append(name)

    evidence = json.load(tar.extractfile('evidence.json'))
    reviewed, unreviewed = 0, 0
    for row in evidence['evidence']:
        prefix = 'runs/' + row['run'] + '/'
        raw = tar.extractfile(prefix + 'result.json').read()
        assert hashlib.sha256(raw).hexdigest() == row['result_sha256']
        review_name = prefix + 'review.json'
        if review_name in manifest:
            review = json.load(tar.extractfile(review_name))
            assert review['result_sha256'] == row['result_sha256']
            assert review['status'] == row['status']
            reviewed += 1
        else:
            assert row['status'] == 'awaiting_literature_review'
            unreviewed += 1
    assert not any(name.endswith(('.pt', '.npy', '.png', '.pdf')) for name in manifest)
    assert not any(name.startswith('runs/tools/') or '/reproducibility.tar.gz' in name for name in manifest)

archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert json.loads((output / 'verification.json').read_text()) == {'files': len(manifest), 'archive_sha256': archive_sha}
result = {
    'research_id': 'E002', 'status': 'artifact_verified',
    'archive': str(archive.relative_to(root)), 'archive_sha256': archive_sha,
    'compressed_bytes': archive.stat().st_size, 'payload_files': len(manifest),
    'evidence_rows': len(evidence['evidence']), 'reviewed_rows': reviewed, 'unreviewed_rows': unreviewed,
    'independently_matched_local_files': matched_files,
    'independently_matched_historical_blobs': matched_blobs,
    'additional_gpu_hours': evidence['new_gpu_hours'], 'verification_elapsed_seconds': time.monotonic() - started,
    'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limitations': 'Interim evidence only. Large datasets/checkpoints and rendered PNG/PDF figures are excluded. Live monitor files capture individual read times, not an atomic system snapshot. This verifier, post-build result and literature review are subsequent artifacts and are not inside this archive. Full original training and remaining P25/P26 scientific gates remain pending.'
}
assert not (case / 'result.json').exists()
(case / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['payload_files', 'evidence_rows', 'reviewed_rows', 'unreviewed_rows', 'compressed_bytes', 'archive_sha256', 'verification_elapsed_seconds']}, indent=2))
