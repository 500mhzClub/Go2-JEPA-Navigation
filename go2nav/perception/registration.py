"""Floor registration: correct the visual pose's tilt and height against the floor seen at the start.

The floor is assumed flat and static. Each frame, both cameras' floor points (`floor.paired_floor_points`, selected
along the visual pose's up direction) give the current floor plane in body coordinates. The visual pose is then
corrected so that this plane coincides with the first frame's plane:
- the minimal rotation aligning the current floor normal (seen through the visual rotation) with the reference normal;
- a twist about the reference normal restoring the visual pose's forward heading (so only tilt changes);
- a translation along the normal correcting the height (the in-plane position is kept).
Corrections are bounded (0.1 rad tilt, 5 cm height).

When a frame has too few floor points for a plane (fewer than 100), the last registered frame's correction is carried
forward instead (the visual motion since then, composed onto its registered pose; bounded to 5 cm and 0.1 rad).
"""
from dataclasses import dataclass

import numpy as np

from go2nav.perception import floor
from go2nav.perception.rigid import UnportedBranch
from go2nav.perception.rotations import angle

MAX_TILT_RAD, MAX_HEIGHT_M = .10, .05
MAX_TRANSPORT_M, MAX_TRANSPORT_RAD = .05, .10


@dataclass
class RegisteredPose:
    """A floor-registered (or floor-transported) pose and the visual pose it corrects."""
    frame: int
    measured_ns: int
    position: np.ndarray
    rotation: np.ndarray
    visual_position: np.ndarray
    visual_rotation: np.ndarray
    transported: bool
    plane: dict  # the current frame's joint floor plane (unavailable when transported)


def register_pose(p, R, reference_plane, current_plane):
    """Corrected (position, rotation) of a visual pose (see the module docstring)."""
    n = np.asarray(reference_plane['normal_body'], float)
    b = np.asarray(current_plane['normal_body'], float)
    d0, d = float(reference_plane['offset_body_m']), float(current_plane['offset_body_m'])
    a = R@b
    v = np.cross(a, n)
    c = float(a@n)
    tilt = float(np.arctan2(np.linalg.norm(v), c))
    height = d-d0-float(n@p)
    if tilt > MAX_TILT_RAD or abs(height) > MAX_HEIGHT_M:
        raise UnportedBranch('floor registration correction out of bounds (the original re-acquires the floor)')
    K = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    aligned = (np.eye(3)+K+K@K/(1.+c))@R
    old = R[:, 0]-n*(n@R[:, 0])
    new = aligned[:, 0]-n*(n@aligned[:, 0])
    if min(np.linalg.norm(old), np.linalg.norm(new)) < .2:
        raise UnportedBranch('forward axis nearly along the floor normal')
    old /= np.linalg.norm(old)
    new /= np.linalg.norm(new)
    twist = float(np.arctan2(n@np.cross(new, old), new@old))
    N = np.array([[0., -n[2], n[1]], [n[2], 0., -n[0]], [-n[1], n[0], 0.]])
    return p+height*n, (np.eye(3)+np.sin(twist)*N+(1.-np.cos(twist))*(N@N))@aligned


class FloorRegistration:
    """See the module docstring. `observe` takes one frame's depth images and visual pose."""

    def __init__(self):
        self.initial_up = None
        self.reference_plane = None
        self.anchor = None  # the last registered RegisteredPose

    def observe(self, depth, auxiliary_depth, visual, specific_force):
        """`depth`/`auxiliary_depth`: (depth_m, valid) pairs; `visual`: the visual odometry's Pose;
        `specific_force`: the (20, 3) accelerometer history (used once, for the initial up direction)."""
        if self.initial_up is None:
            up = np.asarray(specific_force).mean(0)
            self.initial_up = up/np.linalg.norm(up)
        p, R = np.asarray(visual.position, float), np.asarray(visual.rotation, float)
        up = R.T@self.initial_up
        clouds, _ = floor.paired_floor_points(depth, auxiliary_depth, up)
        plane = floor.joint_plane(*clouds, up)
        if not plane['available']:
            if plane['reason'] == 'insufficient_combined_measured_candidates' and self.anchor is not None:
                return self._transport(visual, p, R, plane)
            raise UnportedBranch(f"floor plane unavailable for registration: {plane['reason']}")
        if self.reference_plane is None:
            self.reference_plane = plane
        position, rotation = register_pose(p, R, self.reference_plane, plane)
        self.anchor = RegisteredPose(visual.frame, visual.measured_ns, position, rotation, p, R, False, plane)
        return self.anchor

    def _transport(self, visual, p, R, plane):
        anchor = self.anchor
        correction = anchor.rotation@anchor.visual_rotation.T
        transported_p = anchor.position+correction@(p-anchor.visual_position)
        transported_R = correction@R
        if float(np.linalg.norm(transported_p-p)) > MAX_TRANSPORT_M or angle(R.T@transported_R) > MAX_TRANSPORT_RAD:
            raise UnportedBranch('floor transport correction out of bounds')
        return RegisteredPose(visual.frame, visual.measured_ns, transported_p, transported_R, p, R, True, plane)
