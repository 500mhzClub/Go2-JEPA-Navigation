"""The Go2's two simulated RGB-D cameras: geometry, depth lifting and projection.

Both cameras are 640 x 480 pinhole cameras with the same focal length (78.3° horizontal field of view) and depth
valid from 0.2 to 5 m along the optical axis. The **primary** camera looks forward from the head; the **auxiliary**
camera sits 0.35 m ahead of the body origin, pitched 45° down at the floor.

Body frame: x forward, y left, z up. Optical frame: x right, y down, z along the view.

`lift` and `project` always use the primary camera's optical-to-body transform. Auxiliary-camera points are first
expressed in that "primary-shaped" frame (the *reference adapter* frame) and moved into the body frame with
`AUXILIARY_FROM_ADAPTER`; `adapter_rotation` and `body_pose_from_adapter` convert rotations and poses.
"""
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FOCAL = 392.8974952372412
MIN_DEPTH_M, MAX_DEPTH_M = .2, 5.

# Body from primary optical frame (homogeneous).
PRIMARY = np.asarray(((0., 0., 1., .326), (-1., 0., 0., 0.), (0., -1., 0., .043), (0., 0., 0., 1.)))

# Lifting rules: a bilinear depth sample is rejected if any of its four neighbours is invalid or they spread more
# than 2 cm + 1% of their mean (a surface jump).
DEPTH_SPREAD_BASE_M, DEPTH_SPREAD_FRACTION = .02, .01
LOCAL_DEPTH_WINDOW = 5


def _auxiliary():
    c, s = math.cos(math.pi/4), math.sin(math.pi/4)
    pitch = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    E = np.eye(4)
    E[:3, :3] = pitch@PRIMARY[:3, :3]
    E[:3, 3] = (.35, 0., .08)
    return E


# Body from auxiliary optical frame (homogeneous).
AUXILIARY = _auxiliary()


def _adapter():
    R = AUXILIARY[:3, :3]@PRIMARY[:3, :3].T
    return R, AUXILIARY[:3, 3]-R@PRIMARY[:3, 3]


# Rotation and offset taking reference-adapter coordinates of auxiliary points to body coordinates.
ADAPTER_ROTATION, ADAPTER_OFFSET = _adapter()


def lift(depth_m, valid, uv):
    """Body-frame points (primary-camera geometry) at sub-pixel locations, by bilinear depth; and their validity."""
    uv = np.asarray(uv, float)
    x, y = uv.T
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    inside = (ix >= 0)&(iy >= 0)&(ix < 639)&(iy < 479)
    ix, iy = np.clip(ix, 0, 638), np.clip(iy, 0, 478)
    corners = ((0, 0), (1, 0), (0, 1), (1, 1))
    z = np.stack([depth_m[iy+dy, ix+dx] for dx, dy in corners], 1)
    ok = np.stack([valid[iy+dy, ix+dx] for dx, dy in corners], 1).all(1)&inside
    ok &= np.ptp(z, axis=1) <= DEPTH_SPREAD_BASE_M+DEPTH_SPREAD_FRACTION*np.mean(z, axis=1)
    a, b = x-ix, y-iy
    weights = np.column_stack(((1-a)*(1-b), a*(1-b), (1-a)*b, a*b))
    distance = np.sum(z*weights, axis=1)
    optical = np.column_stack((distance*(x+.5-320)/FOCAL, distance*(y+.5-240)/FOCAL, distance))
    return optical@PRIMARY[:3, :3].T+PRIMARY[:3, 3], ok


def project(points):
    """Pixel coordinates of body-frame points (primary-camera geometry), and whether each is in front (z > 0.2 m)."""
    optical = (np.asarray(points)-PRIMARY[:3, 3])@PRIMARY[:3, :3]
    z = optical[:, 2]
    return optical[:, :2]/np.maximum(z[:, None], 1e-12)*FOCAL+[319.5, 239.5], z > .2


def project_views(points, split):
    """`project` for stacked body-frame points: the first `split` from the primary camera, the rest auxiliary."""
    p, valid_p = project(points[:split])
    q, valid_q = project((points[split:]-ADAPTER_OFFSET)@ADAPTER_ROTATION)
    return np.concatenate((p, q)), np.concatenate((valid_p, valid_q))


def adapter_rotation(rotation_body):
    """A body-frame relative rotation expressed in the reference-adapter frame."""
    return ADAPTER_ROTATION.T@rotation_body@ADAPTER_ROTATION


def body_pose_from_adapter(rotation, translation):
    """A relative pose fitted in the reference-adapter frame, expressed in the body frame."""
    A, b = ADAPTER_ROTATION, ADAPTER_OFFSET
    body_rotation = A@rotation@A.T
    return body_rotation, b+A@np.asarray(translation, float)-body_rotation@b


def local_inverse_depth(depth_m, valid):
    """Depth smoothed as the inverse of the 5 x 5 mean inverse depth, where the window is fully valid and flat.

    Returns (float64 depth, support); unsupported pixels are 0 and invalid.
    """
    d, v = np.asarray(depth_m), np.asarray(valid)
    kernel = np.ones((LOCAL_DEPTH_WINDOW, LOCAL_DEPTH_WINDOW), np.uint8)
    supported = cv2.erode(v.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    lo = cv2.erode(d.astype(np.float64), kernel)
    hi = cv2.dilate(d.astype(np.float64), kernel)
    mean = cv2.boxFilter(d.astype(np.float64), -1, (LOCAL_DEPTH_WINDOW, LOCAL_DEPTH_WINDOW), normalize=True)
    supported &= hi-lo <= DEPTH_SPREAD_BASE_M+DEPTH_SPREAD_FRACTION*mean
    inverse = np.divide(1., d, out=np.zeros(d.shape, np.float64), where=v)
    average = cv2.boxFilter(inverse, -1, (LOCAL_DEPTH_WINDOW, LOCAL_DEPTH_WINDOW), normalize=True)
    return np.divide(1., average, out=np.zeros_like(average), where=supported), supported
