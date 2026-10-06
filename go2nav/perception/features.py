"""Image features for the visual odometry: measured corners with SIFT descriptors, and their matching.

A `FeatureFrame` holds one camera's grey image, its depth, and up to 150 corners:
- Shi–Tomasi corners (quality 0.01, at least 5 px apart, 3 x 3 block) on depth-valid pixels, kept only where the
  depth can be lifted (`cameras.lift`), ordered by corner strength (minimum eigenvalue), then y, then x;
- if more than 150 are liftable, at most 13 per cell of a 4 x 3 grid of 160-px cells, then the first 150;
- if 150 or fewer, all of them, completed up to 150 with weaker corners (quality 0.001) not already present;
- upright SIFT descriptors (size 8 px, angle 0).

`match` pairs the corners of a reference and a current frame (mutual SIFT ratio test, one corner per half-pixel,
forward-backward Lucas–Kanade check) and lifts both ends to body-frame points with the locally smoothed depth.
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
            self._local_depth = cameras.local_inverse_depth(self.depth_m, self.valid)
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
