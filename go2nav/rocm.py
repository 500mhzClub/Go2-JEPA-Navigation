"""ROCm environment for Genesis's GPU backend.

Genesis compiles its GPU kernels with the ROCm LLVM linker (ld.lld), which must be on PATH, and finds ROCm through
ROCM_PATH. Call `configure()` before `gs.init(backend=gs.gpu)`. It changes nothing when they are already set.
"""
import os
from pathlib import Path


def configure():
    rocm = os.environ.get('ROCM_PATH') or next((str(p) for p in [Path('/opt/rocm'), *sorted(Path('/opt').glob('rocm-*'))]
                                                if p.is_dir()), None)
    if rocm is None:
        return
    os.environ.setdefault('ROCM_PATH', rocm)
    for sub in ('lib/llvm/bin', 'llvm/bin', 'bin'):
        path = Path(rocm)/sub
        if path.is_dir() and str(path) not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/go2nav_mplconfig')
