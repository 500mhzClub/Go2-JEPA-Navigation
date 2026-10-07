"""Corner flow (the visual odometry's fallback associations) on synthetic shifted images."""
from types import SimpleNamespace

import cv2
import numpy as np

from go2nav.perception.features import FeatureFrame, chain, patches_agree, track

DEPTH_M = 2.


def texture(seed=3):
    noise = np.random.default_rng(seed).integers(0, 256, (480, 640)).astype(np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2.)


def frame(gray):
    depth = np.full(gray.shape, DEPTH_M, np.float32)
    return FeatureFrame(np.dstack([gray]*3), depth, np.ones(gray.shape, bool))


def shifted(gray, dx, dy):
    return np.roll(gray, (dy, dx), axis=(0, 1))


def interior(pixels, margin=30):
    """Tracks whose LK window stays clear of the wrapped image border."""
    return ((pixels[:, 0] > margin)&(pixels[:, 0] < 639-margin)&(pixels[:, 1] > margin)&(pixels[:, 1] < 479-margin))


def test_track_follows_a_known_shift():
    gray = texture()
    a, b, ua, ub = track(frame(gray), frame(shifted(gray, 3, 2)))
    inner = interior(ua)
    assert inner.sum() > 50
    assert np.allclose((ub-ua)[inner], (3, 2), atol=.1)
    assert np.allclose(a[:, 0], b[:, 0], atol=.05)  # same depth along the view: equal forward distance


def test_track_keeps_one_track_per_half_pixel():
    gray = texture()
    reference = frame(gray)
    pixels = [np.asarray(k.pt, float) for k in reference.keypoints]
    _, _, ua, _ = track(reference, frame(gray), pixels+pixels)
    assert len(ua) == len({tuple(np.rint(p*2).astype(int)) for p in ua})


def test_chain_follows_original_corners_through_frames():
    gray = texture()
    images = [shifted(gray, 2*i, i) for i in range(4)]
    reference, current = frame(images[0]), frame(images[-1])
    between = [SimpleNamespace(gray=g, depth_m=reference.depth_m, valid=reference.valid) for g in images[1:-1]]
    a, b, ua, ub = chain([reference, *between, current])
    corners = {tuple(np.float32(k.pt)) for k in reference.keypoints}
    assert all(tuple(p) in corners for p in ua)
    inner = interior(ua)
    assert inner.sum() > 50
    assert np.allclose((ub-ua)[inner], (6, 3), atol=.15)


def test_patches_agree_rejects_flat_and_unrelated_patches():
    gray = texture()
    flat = np.full_like(gray, 128)
    p = np.asarray([[100., 100.], [200., 300.]], np.float32)
    assert patches_agree(gray, gray, p, p).all()
    assert not patches_agree(gray, flat, p, p).any()
    assert not patches_agree(gray, texture(seed=4), p, p).any()
