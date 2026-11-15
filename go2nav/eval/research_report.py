"""Build a local evidence index and reproducibility archive without inventing result reviews."""
import argparse
import json
from pathlib import Path
import tarfile

from go2nav.research import REPO, atomic_json, device_hours, digest, require_review


def build(root, out, bundle=False):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    for result in sorted(root.rglob('result.json')):
        data = json.loads(result.read_text())
        review_path = result.with_name('review.json')
        review = require_review(review_path, result) if review_path.exists() else None
        evidence.append(dict(run=str(result.parent.relative_to(root)), research_id=data.get('research_id'),
                             result_sha256=digest(result), status=review['status'] if review else data.get('status'),
                             next_step=review['next_step'] if review else 'Literature review required before a dependent experiment'))
    ledger = root/'gpu_intervals.json'
    hours = device_hours(json.loads(ledger.read_text())) if ledger.exists() else 0.
    atomic_json(out/'evidence.json', dict(evidence=evidence, new_gpu_hours=hours, allowance_hours=48))
    lines = ['# P25/P26 evidence', '', f'Additional GPU device time: {hours:.4f} / 48 hours.', '',
             '| Run | Research | Reviewed status | Next step |', '|---|---|---|---|']
    for row in evidence:
        lines.append('| '+' | '.join(str(row[k]).replace('|', '/') for k in ('run', 'research_id', 'status', 'next_step'))+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if bundle:
        paths = [(p, str(p.relative_to(REPO))) for base in ('go2nav', 'configs', 'docs/research', 'tests', 'env')
                 for p in (REPO/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix in ('.py', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.sh')]
        paths += [(REPO/p, p) for p in ('AGENTS.md', 'README.md', 'pyproject.toml', 'THIRD_PARTY_NOTICES.md')]
        for path in root.rglob('*'):
            relative = path.relative_to(root)
            if path.is_file() and relative.parts[0] not in ('sources', 'report') and 'corpus' not in relative.parts \
                    and path.suffix in ('.json', '.jsonl', '.npz'):
                paths.append((path, 'runs/'+str(relative)))
        paths += [(out/'evidence.json', 'evidence.json'), (out/'report.md', 'report.md')]
        atomic_json(out/'bundle_manifest.json', {name: digest(path) for path, name in paths})
        with tarfile.open(out/'reproducibility.tar.gz', 'w:gz') as tar:
            for path, name in paths:
                tar.add(path, arcname=name, recursive=False)
            tar.add(out/'bundle_manifest.json', arcname='bundle_manifest.json', recursive=False)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bundle', action='store_true')
    a = p.parse_args()
    print(json.dumps(build(a.root, a.out, a.bundle), indent=1))


if __name__ == '__main__':
    main()
