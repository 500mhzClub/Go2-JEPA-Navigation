"""Fetch external artefacts into the go2nav cache ($GO2NAV_CACHE, default ~/.cache/go2nav), checking SHA-256.

    python artifacts/fetch.py vjepa [--link-weights PATH] [--link-source DIR]

`vjepa`: the V-JEPA 2.1 source (github.com/facebookresearch/vjepa2 at the pinned commit; MIT licence, some portions
Apache-2.0) and the ViT-L (distilled from ViT-G, 384 px) weights from Meta's download server. With --link-weights or
--link-source, existing local copies are linked instead of downloaded (they are still hash-checked).

Project checkpoints and datasets are listed in artifacts/manifest.yaml; their download is added when they are uploaded.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import urllib.request

CACHE = Path(os.environ.get('GO2NAV_CACHE', Path.home()/'.cache/go2nav'))
VJEPA_REPO = 'https://github.com/facebookresearch/vjepa2.git'
VJEPA_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
VJEPA_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
VJEPA_WEIGHTS_SHA256 = '7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_vjepa(link_weights=None, link_source=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE/f'vjepa2-{VJEPA_COMMIT}'
    if not source.exists():
        if link_source:
            source.symlink_to(Path(link_source).resolve())
        else:
            subprocess.run(['git', 'clone', '--quiet', VJEPA_REPO, str(source)], check=True)
            subprocess.run(['git', '-C', str(source), 'checkout', '--quiet', VJEPA_COMMIT], check=True)
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if commit != VJEPA_COMMIT:
        raise ValueError(f'V-JEPA source at {commit}, expected {VJEPA_COMMIT}')
    weights = CACHE/'vjepa2_1_vitl_dist_vitG_384.pt'
    if not weights.exists():
        if link_weights:
            weights.symlink_to(Path(link_weights).resolve())
        else:
            urllib.request.urlretrieve(VJEPA_WEIGHTS_URL, weights)
    if sha256(weights) != VJEPA_WEIGHTS_SHA256:
        raise ValueError(f'{weights} does not match the expected SHA-256')
    print(f'V-JEPA 2.1 ready: {source}, {weights}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['vjepa'])
    p.add_argument('--link-weights')
    p.add_argument('--link-source')
    a = p.parse_args()
    fetch_vjepa(a.link_weights, a.link_source)


if __name__ == '__main__':
    main()
