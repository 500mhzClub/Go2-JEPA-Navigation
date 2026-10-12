"""Floor registration: correct the visual pose's tilt and height against the floor seen at the start.

The floor is assumed flat and static. Each frame, both cameras' floor points (`floor.paired_floor_points`, selected
along the visual pose's up direction) give the current floor plane in body coordinates. The visual pose is then
corrected so that this plane coincides with the first frame's plane:
- the minimal rotation aligning the current floor normal (seen through the visual rotation) with the reference normal;
- a twist about the reference normal restoring the visual pose's forward heading (so only tilt changes);
- a translation along the normal correcting the height (the in-plane position is kept).
Corrections are bounded (0.1 rad tilt, 5 cm height).

When a frame's floor points cannot give a plane, the last registered frame's correction is carried forward instead
(*transport*: the visual motion since then, composed onto its registered pose; bounded to 5 cm and 0.1 rad):
- fewer than 100 points: the transported pose as it is;
- at least 100 points but too narrow a patch for a normal: the transported pose with its height corrected so that the
  points' mean lies on the transported floor (the update and the total correction bounded to 5 cm). Such a patch is
  first pruned along the transported floor normal (`floor.condition_narrow_patch`).
Every current floor point must then lie within 3 mm of the transported floor. If one does not, the frame's floor
measurement is rejected: `observe` returns None (no pose for this frame) and the controller holds until the floor is
measured again (`NavigationController`, floor re-acquisition).
"""
from dataclasses import dataclass

import numpy as np

from go2nav.perception import floor
from go2nav.perception.rigid import UnportedBranch
from go2nav.perception.rotations import angle

MAX_TILT_RAD, MAX_HEIGHT_M = .10, .05
MAX_TRANSPORT_M, MAX_TRANSPORT_RAD = .05, .10
MAX_TRANSPORT_RESIDUAL_M = .003


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
    """See the module docstring. `observe` takes one frame's depth images and visual pose and returns the
    `RegisteredPose`, or None if the floor measurement is rejected."""

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
        carried = None if self.anchor is None else self._carried_rotation(R).T@self.reference_plane['normal_body']
        clouds, _ = floor.paired_floor_points(depth, auxiliary_depth, up, narrow_normal=carried, per_camera_mean=True)
        plane = floor.joint_plane(*clouds, up)
        if not plane['available']:
            if plane['reason'] in (floor.TOO_FEW_POINTS, floor.TOO_NARROW) and self.anchor is not None:
                return self._transport(visual, p, R, plane, clouds)
            raise UnportedBranch(f"floor plane unavailable for registration: {plane['reason']}")
        if self.reference_plane is None:
            self.reference_plane = plane
        position, rotation = register_pose(p, R, self.reference_plane, plane)
        self.anchor = RegisteredPose(visual.frame, visual.measured_ns, position, rotation, p, R, False, plane)
        return self.anchor

    def _carried_rotation(self, R):
        """The visual rotation `R` with the last registered frame's correction."""
        return self.anchor.rotation@self.anchor.visual_rotation.T@R

    def _transport(self, visual, p, R, plane, clouds):
        anchor = self.anchor
        correction = anchor.rotation@anchor.visual_rotation.T
        position = anchor.position+correction@(p-anchor.visual_position)
        rotation = correction@R
        if float(np.linalg.norm(position-p)) > MAX_TRANSPORT_M or angle(R.T@rotation) > MAX_TRANSPORT_RAD:
            raise UnportedBranch('floor transport correction out of bounds')
        normal = np.asarray(self.reference_plane['normal_body'], float)
        floor_normal = rotation.T@normal  # the reference floor in the current body frame, through the transported pose
        floor_offset = float(self.reference_plane['offset_body_m']+normal@position)
        if plane['reason'] == floor.TOO_NARROW:  # enough points for the height, not for a normal
            update = -float(floor_normal@plane['mean_body']+floor_offset)
            position = position+update*normal
            if float(np.linalg.norm(position-p)) > MAX_TRANSPORT_M or abs(update) > MAX_TRANSPORT_M:
                raise UnportedBranch('partial floor height correction out of bounds')
            floor_offset = float(self.reference_plane['offset_body_m']+normal@position)
        for points in clouds:
            if len(points) and float(np.abs(points@floor_normal+floor_offset).max()) > MAX_TRANSPORT_RESIDUAL_M:
                return None
        return RegisteredPose(visual.frame, visual.measured_ns, position, rotation, p, R, True, plane)
