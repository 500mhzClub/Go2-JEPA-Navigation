"""Image features for the visual odometry: measured corners with SIFT descriptors, and their matching.

A `FeatureFrame` holds one camera's grey image, its depth, and up to 150 corners:
- Shi–Tomasi corners (quality 0.01, at least 5 px apart, 3 x 3 block) on depth-valid pixels, kept only where the
  depth can be lifted (`cameras.lift`), ordered by corner strength (minimum eigenvalue), then y, then x;
- if more than 150 are liftable, at most 13 per cell of a 4 x 3 grid of 160-px cells, then the first 150;
- if 150 or fewer, all of them, completed up to 150 with weaker corners (quality 0.001) not already present;
- upright SIFT descriptors (size 8 px, angle 0).

`match` pairs the corners of a reference and a current frame (mutual SIFT ratio test, one corner per half-pixel,
forward-backward Lucas–Kanade check) and lifts both ends to body-frame points with the locally smoothed depth.

The visual odometry's fallbacks use corner flow instead of descriptors:
- `track` follows the reference corners into the current image by pyramidal Lucas–Kanade from the same pixel; a track
  is kept if the backward flow returns within 0.5 px, the 11 x 11 patches around both ends agree (zero-normalised
  correlation at least 0.90, both standard deviations at least 2 grey levels), and its end is the first at its
  half-pixel;
- `chain` follows a reference frame's corners through consecutive frames, link by link; a track survives a link only
  if both of its ends have valid sensor depth, and lost tracks never return.
"""
import cv2
import numpy as np

from go2nav.perception import cameras

BUDGET = 150
PER_CELL = 13
QUALITY, COMPLETION_QUALITY = .01, .001
MIN_DISTANCE_PX, BLOCK = 5., 3
DESCRIPTOR_SIZE_PX, DESCRIPTOR_ANGLE = 8., 0.
RATIO = .7
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01),
          flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
FORWARD_BACKWARD_PX, FLOW_AGREEMENT_PX = .5, 1.
FLOW_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
PATCH_WIDTH, MIN_PATCH_STD, MIN_PATCH_ZNCC = 11, 2., .90


def cell_index(x, y):
    """Cell of a 4 x 3 grid of 160-px cells."""
    return int(y//160)*4+int(x//160)


def grid_cells(uv):
    """Number of distinct 160-px grid cells covered by pixel locations."""
    return len(set((int(x//160), int(y//160)) for x, y in np.asarray(uv)))


class FeatureFrame:
    """One camera image's corners and descriptors (see the module docstring)."""

    def __init__(self, rgb, depth_m, valid):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.depth_m, self.valid = depth_m, valid
        self._local_depth = None
        response = cv2.cornerMinEigenVal(self.gray, blockSize=BLOCK)
        corners = self._liftable_corners(QUALITY, response)
        if len(corners) > BUDGET:
            selected, counts = [], [0]*12
            for x, y in corners:
                cell = cell_index(x, y)
                if counts[cell] < PER_CELL:
                    selected.append((x, y))
                    counts[cell] += 1
            keypoints, descriptors = self._describe(selected)
            self.keypoints = keypoints[:BUDGET]
            self.descriptors = None if descriptors is None else descriptors[:BUDGET].copy()
            return
        self.keypoints, self.descriptors = self._describe(corners)
        if len(self.keypoints) == BUDGET:
            return
        existing = {tuple(k.pt) for k in self.keypoints}
        extra = [p for p in self._liftable_corners(COMPLETION_QUALITY, response)
                 if tuple(p) not in existing][:BUDGET-len(self.keypoints)]
        if extra:
            keys, descriptors = self._describe(extra)
            self.keypoints = list(self.keypoints)+list(keys)
            self.descriptors = descriptors if self.descriptors is None else np.concatenate((self.descriptors, descriptors))

    def _liftable_corners(self, quality, response):
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=0, qualityLevel=quality, minDistance=MIN_DISTANCE_PX,
                                         mask=self.valid.astype(np.uint8)*255, blockSize=BLOCK, useHarrisDetector=False)
        points = np.empty((0, 2), np.float32) if points is None else points[:, 0]
        _, liftable = cameras.lift(self.depth_m, self.valid, points)
        return sorted(points[liftable].tolist(), key=lambda p: (-float(response[int(p[1]), int(p[0])]), p[1], p[0]))

    def _describe(self, points):
        if not points:
            return (), None
        keypoints = [cv2.KeyPoint(x, y, DESCRIPTOR_SIZE_PX, DESCRIPTOR_ANGLE) for x, y in points]
        return cv2.SIFT_create().compute(self.gray, keypoints)

    def local_depth(self):
        """The locally smoothed depth used to lift matched corners (computed once per frame)."""
        if self._local_depth is None:
            depth_m, supported = cameras.local_inverse_depth(self.depth_m, self.valid)
            self._local_depth = depth_m.astype(np.float32), supported
        return self._local_depth


def match(reference, current):
    """Matched corners of two frames: (reference points, current points, reference pixels, current pixels).

    Points are body-frame (primary-camera geometry, see `cameras`), lifted with each frame's local depth.
    """
    kp0, kp1, d0, d1 = reference.keypoints, current.keypoints, reference.descriptors, current.descriptors
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    if d0 is None or d1 is None or min(len(d0), len(d1)) < 2:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(a, b):
        return {x.queryIdx: x.trainIdx for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2
                for x, y in [pair] if x.distance < RATIO*y.distance}
    forward, backward = ratio_matches(d0, d1), ratio_matches(d1, d0)
    pairs, seen0, seen1 = [], set(), set()
    for i, j in sorted(forward.items()):
        if backward.get(j) != i:
            continue
        left = tuple(np.rint(np.asarray(kp0[i].pt)*2).astype(int))
        right = tuple(np.rint(np.asarray(kp1[j].pt)*2).astype(int))
        if left in seen0 or right in seen1:
            continue
        pairs.append((i, j))
        seen0.add(left)
        seen1.add(right)
    if not pairs:
        return empty
    p = np.float32([kp0[i].pt for i, j in pairs]).reshape(-1, 1, 2)
    initial = np.float32([kp1[j].pt for i, j in pairs]).reshape(-1, 1, 2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(reference.gray, current.gray, p, initial.copy(), **LK)
    back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(current.gray, reference.gray, q, p.copy(), **LK)
    good = ok.ravel().astype(bool)&reverse_ok.ravel().astype(bool)
    good &= np.isfinite(q).all((1, 2))&np.isfinite(back).all((1, 2))
    good &= np.linalg.norm(back[:, 0]-p[:, 0], axis=1) <= FORWARD_BACKWARD_PX
    good &= np.linalg.norm(q[:, 0]-initial[:, 0], axis=1) <= FLOW_AGREEMENT_PX
    ua, ub = p[good, 0], q[good, 0]
    a, va = cameras.lift(*reference.local_depth(), ua)
    b, vb = cameras.lift(*current.local_depth(), ub)
    use = va&vb
    return a[use], b[use], ua[use], ub[use]


# -- corner flow (the visual odometry's fallback associations) ------------------------------------------------------

def half_pixel(point):
    return tuple(np.rint(np.asarray(point)*2).astype(int))


def _patch_agrees(left, right, p, q):
    half = PATCH_WIDTH//2
    if any(not (half <= x <= 639-half and half <= y <= 479-half) for x, y in (p, q)):
        return False
    a = cv2.getRectSubPix(left, (PATCH_WIDTH, PATCH_WIDTH), tuple(map(float, p))).astype(float)
    b = cv2.getRectSubPix(right, (PATCH_WIDTH, PATCH_WIDTH), tuple(map(float, q))).astype(float)
    a -= a.mean()
    b -= b.mean()
    if min(float(a.std()), float(b.std())) < MIN_PATCH_STD:
        return False
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))) >= MIN_PATCH_ZNCC


def patches_agree(left, right, p, q):
    """Which tracks' end patches agree. Batched; a decision within 1e-10 of a threshold is recomputed alone, as the
    original did, so that the summation order cannot change it."""
    half = PATCH_WIDTH//2
    inside = np.ones(len(p), bool)
    for pixels in (p, q):
        inside &= (pixels[:, 0] >= half)&(pixels[:, 0] <= 639-half)&(pixels[:, 1] >= half)&(pixels[:, 1] <= 479-half)
    indices = np.flatnonzero(inside)
    result = np.zeros(len(p), bool)
    if not len(indices):
        return result
    size = (PATCH_WIDTH, PATCH_WIDTH)
    a = np.stack([cv2.getRectSubPix(left, size, tuple(map(float, p[i]))) for i in indices]).astype(float)
    b = np.stack([cv2.getRectSubPix(right, size, tuple(map(float, q[i]))) for i in indices]).astype(float)
    a -= a.mean(axis=(1, 2), keepdims=True)
    b -= b.mean(axis=(1, 2), keepdims=True)
    spread = np.minimum(a.std(axis=(1, 2)), b.std(axis=(1, 2)))
    denominator = np.sqrt(np.sum(a*a, axis=(1, 2))*np.sum(b*b, axis=(1, 2)))
    correlation = np.divide(np.sum(a*b, axis=(1, 2)), denominator, out=np.zeros(len(indices)), where=denominator > 0)
    result[indices] = (spread >= MIN_PATCH_STD)&(correlation >= MIN_PATCH_ZNCC)
    near = (np.abs(spread-MIN_PATCH_STD) <= 1e-10)|(np.abs(correlation-MIN_PATCH_ZNCC) <= 1e-10)
    for i in indices[near]:
        result[i] = _patch_agrees(left, right, p[i], q[i])
    return result


def track(reference, current, pixels=None, *, local=True):
    """Corner flow from `reference` to `current`: (reference points, current points, reference pixels, current
    pixels), as `match` returns. `pixels` are the reference pixels to follow (default: its corners). Both ends are
    lifted with their frame's local depth, or with its sensor depth if `local` is False (frames then need only `gray`,
    `depth_m` and `valid`)."""
    if pixels is None:
        pixels = [np.asarray(k.pt, float) for k in reference.keypoints]
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    points, seen = [], set()
    for point in pixels:
        if half_pixel(point) not in seen:
            points.append(point)
            seen.add(half_pixel(point))
    if not points:
        return empty
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(reference.gray, current.gray, p, None, **FLOW_LK)
    good = ok.ravel().astype(bool)&np.isfinite(q).all((1, 2))
    p, q = p[good], q[good]
    if not len(p):
        return empty
    back, ok, _ = cv2.calcOpticalFlowPyrLK(current.gray, reference.gray, q, None, **FLOW_LK)
    good = ok.ravel().astype(bool)&np.isfinite(back).all((1, 2))
    good &= np.linalg.norm(back[:, 0]-p[:, 0], axis=1) <= FORWARD_BACKWARD_PX
    p, q = p[good, 0], q[good, 0]
    good = patches_agree(reference.gray, current.gray, p, q)
    p, q = p[good], q[good]
    keep, seen = [], set()
    for i, point in enumerate(q):
        if half_pixel(point) not in seen:
            keep.append(i)
            seen.add(half_pixel(point))
    p, q = p[keep], q[keep]
    if local:
        a, va = cameras.lift(*reference.local_depth(), p)
        b, vb = cameras.lift(*current.local_depth(), q)
    else:
        a, va = cameras.lift(reference.depth_m, reference.valid, p)
        b, vb = cameras.lift(current.depth_m, current.valid, q)
    valid = va&vb&np.isfinite(a).all(1)&np.isfinite(b).all(1)
    return a[valid], b[valid], p[valid], q[valid]


def chain(frames):
    """Corner flow from the first to the last of consecutive frames (see the module docstring), as `match` returns.

    The first frame's corners seed the tracks and every link is a `track` with sensor depth; the surviving tracks'
    original reference pixels and final pixels are then lifted with local depth. The first and last frames are
    FeatureFrames; the frames between need only `gray`, `depth_m` and `valid`."""
    reference = frames[0]
    origins = {}
    for keypoint in reference.keypoints:
        point = np.asarray(keypoint.pt, np.float32)
        origins.setdefault(half_pixel(point), point.copy())
    view, pixels = reference, None
    origin_pixels = current_pixels = np.empty((0, 2), np.float32)
    for current in frames[1:]:
        _, _, before, after = track(view, current, pixels, local=False)
        origin_pixels = np.asarray([origins[half_pixel(p)] for p in before], np.float32).reshape(-1, 2)
        current_pixels = np.asarray(after, np.float32).reshape(-1, 2)
        origins = {half_pixel(p): origin.copy() for p, origin in zip(current_pixels, origin_pixels)}
        view, pixels = current, list(current_pixels)
    a, va = cameras.lift(*reference.local_depth(), origin_pixels)
    b, vb = cameras.lift(*frames[-1].local_depth(), current_pixels)
    valid = va&vb&np.isfinite(a).all(1)&np.isfinite(b).all(1)
    return a[valid], b[valid], origin_pixels[valid], current_pixels[valid]
