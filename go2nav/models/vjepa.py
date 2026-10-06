"""The frozen V-JEPA 2.1 ViT-L image encoder: RGB frames to a 24 x 32 grid of 1024-d tokens.

The encoder is Meta's released V-JEPA 2.1 ViT-L (distilled from ViT-G, 384 px), built with the official source
(facebookresearch/vjepa2 at a pinned commit) and its EMA encoder weights. Single frames go through the official image
tokenizer (temporal size 1), so no frame is duplicated to make a clip. It is never trained here.

Frames are resized to 512 x 384 (bicubic) and ImageNet-normalised: 16 divides both sides, so every token covers image
content. Token features are layer-normalised before use (`normalise`). Many consumers pool the grid 2 x 2 to 12 x 16
(`pool_tokens`).

Weights and source are fetched by `artifacts/fetch.py`; see `artifacts/manifest.yaml` for URLs, hashes and licence.
"""
from contextlib import contextmanager
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

GRID = (24, 32)
TOKENS = GRID[0]*GRID[1]
TOKEN_DIM = 1024
INPUT_SIZE = (512, 384)  # (width, height)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SOURCE_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
CACHE = Path(os.environ.get('GO2NAV_CACHE', Path.home()/'.cache/go2nav'))
SOURCE_DIR = CACHE/f'vjepa2-{SOURCE_COMMIT}'
WEIGHTS = CACHE/'vjepa2_1_vitl_dist_vitG_384.pt'


def preprocess(image):
    """An RGB image (PIL or HxWx3 uint8) to the encoder input: (3, 384, 512), ImageNet-normalised."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    resized = image.convert('RGB').resize(INPUT_SIZE, Image.Resampling.BICUBIC)
    chw = torch.from_numpy(np.asarray(resized, dtype=np.float32)/255.0).permute(2, 0, 1).contiguous()
    mean = chw.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = chw.new_tensor(IMAGENET_STD)[:, None, None]
    return (chw-mean)/std


def preprocess_may_corpus_frame(image):
    """A 224 x 224 May-corpus frame (T1 training data) to the encoder input.

    The May corpus was rendered square at the same focal length as the Go2 camera, so cropping 28 rows top and bottom
    (224 x 168) gives exactly the Go2 camera's field of view (78.3 x 62.8 degrees); the crop is then preprocessed like
    any frame.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    if image.size != (224, 224):
        raise ValueError(f'224 x 224 May-corpus frame required, got {image.size}')
    return preprocess(image.convert('RGB').crop((0, 28, 224, 196)))


def normalise(tokens):
    """Layer-normalise token features (no affine), in float32."""
    return F.layer_norm(tokens.float(), (TOKEN_DIM,))


def pool_tokens(tokens):
    """(B, 768, 1024) tokens to (B, 192, 1024): average 2 x 2 neighbouring patches into a 12 x 16 grid."""
    if tokens.ndim != 3 or tokens.shape[1:] != (TOKENS, TOKEN_DIM):
        raise ValueError('(B, 768, 1024) tokens required')
    grid = tokens.transpose(1, 2).reshape(-1, TOKEN_DIM, *GRID)
    return F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2).contiguous()


@contextmanager
def _legacy_timm_drop_path():
    """The V-JEPA source imports `timm.models.layers.drop_path`/`DropPath`; provide just those (drop_path is inactive)."""
    names = ('timm', 'timm.models', 'timm.models.layers')
    saved = {n: sys.modules.get(n) for n in names}

    def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
        if drop_prob == 0.0 or not training:
            return x
        keep = 1-drop_prob
        mask = x.new_empty((x.shape[0],)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x*mask/keep if keep > 0.0 and scale_by_keep else x*mask

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=0.0, scale_by_keep=True):
            super().__init__()
            self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    modules = {}
    for n in names:
        module = types.ModuleType(n)
        module.__package__, module.__path__ = n, []
        module.__spec__ = importlib.machinery.ModuleSpec(name=n, loader=None, is_package=True)
        modules[n] = module
    modules['timm.models.layers'].drop_path, modules['timm.models.layers'].DropPath = drop_path, DropPath
    modules['timm'].models, modules['timm.models'].layers = modules['timm.models'], modules['timm.models.layers']
    sys.modules.update(modules)
    try:
        yield
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


class VJepaEncoder:
    def __init__(self, device='cuda:0', dtype=torch.float32, weights=WEIGHTS, source=SOURCE_DIR):
        if not Path(weights).is_file() or not Path(source).is_dir():
            raise FileNotFoundError('V-JEPA 2.1 source or weights missing; run `python artifacts/fetch.py vjepa`')
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        with _legacy_timm_drop_path():
            backbones = importlib.import_module('src.hub.backbones')
            encoder, _ = backbones.vjepa2_1_vit_large_384(pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=False)
        encoder.load_state_dict(backbones._clean_backbone_key(state['ema_encoder']), strict=True)
        self.module = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def tokens(self, pixels):
        """(B, 3, 384, 512) preprocessed frames to (B, 768, 1024) raw tokens (image tokenizer: temporal size 1)."""
        return self.module(pixels.unsqueeze(2))
