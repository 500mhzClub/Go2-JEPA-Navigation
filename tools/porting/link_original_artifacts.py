"""Link the original machine's copies of the shipped checkpoints into the go2nav cache, until they are hosted.

    python tools/porting/link_original_artifacts.py

For every manifest entry with a `source` (artifacts/manifest.yaml; external artefacts are fetched by
artifacts/fetch.py instead), links its cache path (`go2nav.artifacts.cached`) to the original file
(`tools/porting/original.py`) after checking its size and SHA-256 against the manifest.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2nav import artifacts  # noqa: E402
from tools.porting import original  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    linked = {}
    for name, entry in artifacts.entries().items():
        if entry['kind'] == 'external':
            continue
        source = original.resolve(entry['source'])
        if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise ValueError(f'{name}: {source} does not match the manifest')
        target = artifacts.cached(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        linked[name] = str(target)
    print(json.dumps(linked, indent=1))


if __name__ == '__main__':
    main()
