"""Visual support: remember a well-supported view, and ask for it again when the tracker's support drops low.

The support of a frame is the larger of the two cameras' corner counts (`visual_odometry.Pose.feature_counts`); it is
a recovery heuristic, not a calibrated tracking confidence. `SupportedView.advance` is called with every registered
frame (10 Hz):

- a frame with at least 96 corners becomes the *good view* (its registered pose);
- with fewer than 48 corners and a good view within 20 cm, the good view becomes the *active view*: plans then turn
  to its heading instead of following the route (`planning.planner`);
- the active view is released once the heading is within 0.1 rad of it and the support is back to 48 corners;
- an active view held within 0.1 rad for a full second of consecutive frames without that is *retired*: it and the
  good view are dropped, and a later strong frame starts again.
"""
import math

import numpy as np

LOW_FEATURES, STRONG_FEATURES = 48, 96
MAX_VIEW_DISTANCE_M = .20
HEADING_TOLERANCE_RAD = .1
ATTAINED_VIEW_DWELL_NS = 1_000_000_000
MAX_FRAME_GAP_NS = 100_000_000


def heading_error(target, rotation):
    """The yaw from `rotation` to `target` (rotation matrices), wrapped."""
    difference = math.atan2(target[1, 0], target[0, 0])-math.atan2(rotation[1, 0], rotation[0, 0])
    return math.atan2(math.sin(difference), math.cos(difference))


class SupportedView:
    """See the module docstring. `retirements` lists the retired views."""

    def __init__(self):
        self.good = self.active = None
        self.aligned_since_ns = self.aligned_trigger_ns = self.last_ns = None
        self.retirements = []

    def advance(self, counts, position, rotation, now_ns):
        """The active view after this frame (dict: position, rotation, measured_ns of the good view and trigger_ns,
        the time the recovery started), or None."""
        if self.last_ns is None or not 0 < now_ns-self.last_ns <= MAX_FRAME_GAP_NS:
            self.aligned_since_ns = self.aligned_trigger_ns = None
        self.last_ns = now_ns
        strength = max(counts)
        if self.active is not None and strength >= LOW_FEATURES \
                and abs(heading_error(self.active['rotation'], rotation)) <= HEADING_TOLERANCE_RAD:
            self.active = None
        if strength >= STRONG_FEATURES:
            self.good = dict(position=position.copy(), rotation=rotation.copy(), measured_ns=now_ns)
        if self.active is None and strength < LOW_FEATURES and self.good is not None \
                and np.linalg.norm(position-self.good['position']) <= MAX_VIEW_DISTANCE_M:
            self.active = self.good | dict(trigger_ns=now_ns)
        active = self.active
        if active is None:
            self.aligned_since_ns = self.aligned_trigger_ns = None
            return None
        error = heading_error(active['rotation'], rotation)
        if abs(error) > HEADING_TOLERANCE_RAD:
            self.aligned_since_ns = self.aligned_trigger_ns = None
            return active
        if self.aligned_trigger_ns != active['trigger_ns']:
            self.aligned_since_ns, self.aligned_trigger_ns = now_ns, active['trigger_ns']
        if now_ns-self.aligned_since_ns < ATTAINED_VIEW_DWELL_NS:
            return active
        self.retirements.append(dict(measured_ns=now_ns, trigger_ns=active['trigger_ns'],
                                     reference_measured_ns=active['measured_ns'], aligned_since_ns=self.aligned_since_ns,
                                     heading_error_rad=error, feature_counts=list(counts)))
        self.good = self.active = None
        self.aligned_since_ns = self.aligned_trigger_ns = None
        return None
