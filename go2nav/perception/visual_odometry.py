"""Visual odometry: the robot's pose from its two RGB-D cameras and its gyro.

Every frame (10 Hz) is measured against retained **reference** frames (keyframes): matched corners
(`features.match`), the gyro's relative rotation, and both frames' floor planes give a relative pose
(`rigid` steps 1–3); composing it with the reference's pose gives the current pose in the initial body frame.

**Which reference.** Up to 8 references are kept, one of which is the *stable anchor*. The anchor is tried first,
then the others from newest to oldest; if the anchor fails, the qualified alternative with the best support (grid
cells, inlier fraction, inliers, residual) is used. The previous frame is also measured, unless it is the chosen
reference, and must agree with the anchor measurement within 2 cm and 0.1 rad. Every fourth frame, a remembered view
from at least 1 s ago, within 20 cm and 0.2 rad of the current estimate, temporarily stands in as the anchor
(revisiting a place re-anchors to the old view rather than accumulating drift).

**Which camera.** The forward camera first; if neither the anchor nor the previous frame can be measured with it,
the downward auxiliary camera; and if that fails against the previous frame, both cameras jointly.

**When to keep a reference.** The current frame becomes a reference if it was measured against an alternative
rather than the anchor, if it moved at least 0.4 m or 0.35 rad from its reference, if its inlier support covers at
most 7 grid cells, if at most half of the reference's corners were inliers, or if the newest reference is at least
0.4 s old. A new reference replaces the stable anchor when the anchor was not the reference used, or is at least
0.4 m or 0.35 rad away; otherwise the anchor is kept beside the 7 most recent references. One remembered view per
45° heading bin (replaced when 30 cm away) serves the revisits.
"""
from copy import copy
from dataclasses import dataclass

import numpy as np

from go2nav.perception import cameras, floor, rigid
from go2nav.perception.features import FeatureFrame, match
from go2nav.perception.rigid import Candidate, Registration, Rejected, UnportedBranch
from go2nav.perception.rotations import GyroAttitude, angle

MAX_REFERENCES = 8
REVISIT_PERIOD_FRAMES = 4
REVISIT_MIN_AGE_NS, REVISIT_MAX_DISTANCE_M, REVISIT_MAX_ROTATION_RAD = 1_000_000_000, .20, .20
VIEW_BIN_REPLACE_DISTANCE_M = .30
PROMOTE_TRANSLATION_M, PROMOTE_ROTATION_RAD = .4, .35
SUPPORT_MARGIN_CELLS = 7
MAX_REFERENCE_AGE_NS = 400_000_000


@dataclass(frozen=True)
class Reference:
    """A retained frame: its features (both cameras), when it was taken, and its accepted pose."""
    frame: int
    measured_ns: int
    features: dict
    rotation: np.ndarray
    gyro: np.ndarray
    position: np.ndarray


@dataclass
class FloorRecord:
    """A frame's floor: the joint plane, the selected raw floor points of both cameras, and their candidate pool."""
    plane: dict
    points: np.ndarray
    pool: int


@dataclass
class Pose:
    """The visual odometry's output for one frame (initial-body frame: x forward, y left, z up at the start)."""
    frame: int
    measured_ns: int
    position: np.ndarray
    rotation: np.ndarray
    gyro_rotation: np.ndarray
    reference_frame: int
    promoted_keyframe: bool
    promotion_reason: str | None
    keyframe_count: int
    camera: str | None
    feature_counts: tuple  # corners kept in this frame (primary, auxiliary)


def view_bin(gyro):
    """45° heading bin of a rotation."""
    import math
    return int(math.floor((math.atan2(gyro[1, 0], gyro[0, 0])+math.pi/8)/(math.pi/4)))%8


class VisualOdometry:
    """See the module docstring. `observe` takes one frame's sensor packet and returns its `Pose`."""

    def __init__(self):
        self.gyro = GyroAttitude()
        self.frame = -1
        self.initial_up = None
        self.references = []
        self.stable = None
        self.previous = None
        self.last_R, self.last_p = np.eye(3), np.zeros(3)
        self.keyframes = 0
        self.floors = {}         # frame -> (features, FloorRecord) for retained references and the previous frame
        self.pending = None      # (frame, features, FloorRecord) of the current frame
        self.view_bank = {}      # heading bin -> (Reference, (features, FloorRecord))
        self.selected_reference = None
        self.camera = 'primary'

    # -- one frame ------------------------------------------------------------------------------------------------

    def observe(self, rgb, depth, auxiliary_rgb, auxiliary_depth, gyro_rates, specific_force, now):
        """`depth` and `auxiliary_depth` are (depth_m, valid) pairs; `gyro_rates` the frame's (51, 3) 500 Hz block;
        `specific_force` the (20, 3) body accelerometer history (used once, for the initial up direction); `now` the
        frame's acquisition time (ns)."""
        if self.initial_up is None:
            up = np.asarray(specific_force).mean(0)
            self.initial_up = up/np.linalg.norm(up)
        G = self.gyro.update(gyro_rates).copy()
        current = dict(primary=FeatureFrame(rgb, *depth), auxiliary=FeatureFrame(auxiliary_rgb, *auxiliary_depth))
        self.frame += 1
        reason, measured = None, False
        if self.previous is None:
            R, p, ref_frame = np.eye(3), np.zeros(3), 0
            self.camera = None
            self._remember(current, R, G, p, now)
            self.keyframes += 1
            self.selected_reference = 0
        else:
            candidate, fallback = self._measure(current, G, now)
            measured = True
            R, p, reg, ref = candidate.R, candidate.p, candidate.registration, candidate.reference
            ref_frame = ref.frame
            moved = np.linalg.norm(candidate.t) >= PROMOTE_TRANSLATION_M or angle(candidate.local_R) >= PROMOTE_ROTATION_RAD
            reason = ('qualified_recent_reference' if fallback else 'motion_threshold' if moved
                      else 'accepted_support_margin' if min(reg.grid_cells) <= SUPPORT_MARGIN_CELLS else None)
            if reg.joint_retained_anchor:
                reference_features = sum(len(ref.features[c].keypoints) for c in ('primary', 'auxiliary'))
            else:
                reference_features = len(ref.features[self.camera].keypoints)
            if reason is None and 2*reg.inliers <= reference_features:
                reason = 'accepted_half_feature_overlap'
            if reason is not None:
                self.keyframes += 1
                self._remember(current, R, G, p, now)
        self.last_R, self.last_p = R.copy(), p.copy()
        self.previous = Reference(self.frame, now, current, R.copy(), G.copy(), p.copy())
        self.floors[self.frame] = (current, self.pending[2])
        retained = {r.frame for r in self.references} | {self.frame}
        self.floors = {f: v for f, v in self.floors.items() if f in retained}
        if measured and reason is None and self.references and \
                now-max(r.measured_ns for r in self.references) >= MAX_REFERENCE_AGE_NS:
            reason = 'accepted_anchor_recent_reference_age'
            self._remember(current, R, G, p, now)
            self.keyframes += 1
        counts = (len(current['primary'].keypoints), len(current['auxiliary'].keypoints))
        return Pose(self.frame, now, p, R, G, ref_frame, reason is not None, reason, self.keyframes, self.camera, counts)

    # -- references -----------------------------------------------------------------------------------------------

    def _remember(self, current, R, G, p, now):
        record = self._floor(current, G, now)
        self.references = (self.references+[Reference(self.frame, now, current, R.copy(), G.copy(), p.copy())])[-MAX_REFERENCES:]
        self.floors[self.frame] = (current, record)
        stable = self.stable
        if (stable is None or self.selected_reference != stable.frame
                or np.linalg.norm(p-stable.position) >= PROMOTE_TRANSLATION_M
                or angle(stable.rotation.T@R) >= PROMOTE_ROTATION_RAD):
            self.stable = self.references[-1]
        else:
            recent = [r for r in self.references if r is not stable][-(MAX_REFERENCES-1):]
            self.references = sorted([stable]+recent, key=lambda r: r.frame)
        reference = next(r for r in self.references if r.frame == self.frame)
        key = view_bin(G)
        old = self.view_bank.get(key)
        if old is None or np.linalg.norm(old[0].position-p) > VIEW_BIN_REPLACE_DISTANCE_M:
            self.view_bank[key] = (reference, self.floors[self.frame])

    def _floor(self, current, G, now):
        """The current frame's FloorRecord (computed once per frame)."""
        if self.pending is not None and self.pending[0] == self.frame:
            return self.pending[2]
        seed_up = np.asarray(G).T@self.initial_up
        up = self.last_R.T@self.initial_up
        clouds, pool = floor.paired_floor_points((current['primary'].depth_m, current['primary'].valid),
                                                 (current['auxiliary'].depth_m, current['auxiliary'].valid), seed_up)
        plane = floor.joint_plane(*clouds, up)
        if not plane['available'] and (self.frame == 0 or plane['reason'] not in (
                'insufficient_combined_measured_candidates', 'insufficient_combined_two_axis_extent')):
            raise UnportedBranch(f"floor plane unavailable at frame {self.frame}: {plane['reason']}")
        plane['seed_up'] = seed_up
        record = FloorRecord(plane, np.concatenate(clouds), pool)
        self.pending = (self.frame, current, record)
        return record

    # -- measurement ----------------------------------------------------------------------------------------------

    def _measure(self, current, G, now):
        """(candidate, measured against an alternative) for the current frame, or the visual odometry fails."""
        revisit = self._revisit(G, now) if self.frame%REVISIT_PERIOD_FRAMES == 0 else None
        saved = self.references, self.stable, self.floors
        if revisit is not None:
            reference, record = revisit
            self.references = self.references[-(MAX_REFERENCES-1):]+[reference]
            self.stable = reference
            self.floors = self.floors | {reference.frame: record}
        try:
            # The stable anchor is tried first.
            self.references = [r for r in self.references if r is not self.stable]+[self.stable]
            self._floor(current, G, now)
            for camera in ('primary', 'auxiliary'):
                self.camera = camera
                try:
                    return self._measure_with(current, G)
                except Rejected:
                    if camera == 'auxiliary':
                        raise UnportedBranch('visual odometry failure: no camera measures the current frame')
        finally:
            self.references, self.stable, self.floors = saved

    def _revisit(self, G, now):
        active = {r.frame for r in self.references}
        entries = [e for e in self.view_bank.values() if e[0].frame not in active
                   and now-e[0].measured_ns >= REVISIT_MIN_AGE_NS
                   and np.linalg.norm(e[0].position-self.last_p) <= REVISIT_MAX_DISTANCE_M
                   and angle(e[0].gyro.T@G) <= REVISIT_MAX_ROTATION_RAD]
        return min(entries, key=lambda e: e[0].frame) if entries else None

    def _measure_with(self, current, G):
        """Anchor and previous-frame measurement with the current camera."""
        anchor = None
        try:
            anchor, alternative = self._choose(current, G)
        except Rejected:
            pass
        if anchor is not None and anchor.reference.frame == self.previous.frame:
            incremental = anchor
        else:
            try:
                incremental = self._candidate(self.previous, current, G)
            except Rejected:
                incremental = None
        if anchor is not None:
            if incremental is not None and (
                    float(np.linalg.norm(anchor.p-incremental.p)) > rigid.MAX_MEASUREMENT_DISAGREEMENT_M
                    or angle(anchor.R.T@incremental.R) > rigid.MAX_MEASUREMENT_DISAGREEMENT_RAD):
                raise UnportedBranch('anchor and previous-frame measurements disagree')
            return anchor, alternative
        if incremental is None:
            raise Rejected('neither a retained reference nor the previous frame supports the current pose')
        raise UnportedBranch('measured-increment bridge (previous frame only)')

    def _choose(self, current, G):
        self.selected_reference = None
        anchor = self.references[-1]
        try:
            candidate = self._candidate(anchor, current, G)
            self.selected_reference = anchor.frame
            return candidate, False
        except Rejected:
            pass
        candidates = []
        for ref in reversed(self.references[:-1]):
            try:
                candidates.append(self._candidate(ref, current, G))
            except Rejected:
                pass
        if not candidates:
            raise Rejected('no retained reference qualifies')
        for i, a in enumerate(candidates):
            for b in candidates[i+1:]:
                if np.linalg.norm(a.p-b.p) > .02 or angle(a.R.T@b.R) > .10:
                    raise UnportedBranch('qualified alternatives disagree')

        def rank(c):
            q = c.registration
            return (min(q.grid_cells), q.inlier_fraction, q.inliers, -q.residual_rms_m, c.reference.frame)
        chosen = max(candidates, key=rank)
        self.selected_reference = chosen.reference.frame
        return chosen, True

    def _candidate(self, ref, current, G):
        """Steps 1–3 of `rigid` against one reference with the current camera (and the joint fallback)."""
        previous = self.previous
        retained_previous = (ref is not previous and any(ref is r for r in self.references)
                             and ref.frame == previous.frame == self.frame-1)
        measured_ref = previous if retained_previous else ref
        if measured_ref.frame not in self.floors:
            raise UnportedBranch('reference without a measured floor')
        reference_floor = self.floors[measured_ref.frame][1]
        current_floor = self.pending[2]
        candidate = self._register(measured_ref, current, G, retained_anchor=retained_previous or any(
            measured_ref is r for r in self.references))
        gyro = measured_ref.gyro.T@G
        if reference_floor.plane['available'] and current_floor.plane['available']:
            candidate = rigid.refine_if_supported(candidate, reference_floor.plane, current_floor.plane,
                                                  camera=self.camera, gyro=gyro, last_p=self.last_p, last_R=self.last_R)
            constraint = floor.floor_constraint(
                reference_floor.points, current_floor.points, gyro,
                reference_pool=reference_floor.pool, current_pool=current_floor.pool,
                reference_up=reference_floor.plane['up_body'], current_up=current_floor.plane['up_body'],
                min_second_eigenvalue_m2=max(reference_floor.plane['min_second_eigenvalue_m2'],
                                             current_floor.plane['min_second_eigenvalue_m2']))
            candidate.registration.floor_constraint = constraint
        else:
            candidate.registration.plane_refinement_applied = False
        candidate = rigid.gyro_consensus_refit(candidate, gyro, camera=self.camera, last_p=self.last_p,
                                               last_R=self.last_R)
        candidate.reference = ref
        return candidate

    def _register(self, ref, current, G, *, retained_anchor):
        """Step 1: matching and gyro-conditioned consensus, single camera or (fallback) both cameras."""
        relative_gyro = ref.gyro.T@G
        try:
            if self.camera == 'primary':
                a, b, ua, ub = match(ref.features['primary'], current['primary'])
                local_R, t, mask, residual = rigid.consensus(a, b, ua, ub, relative_gyro, self.frame)
                R, p = ref.rotation@local_R, ref.position+ref.rotation@t
                points = a[mask], b[mask]
            else:
                a, b, ua, ub = match(ref.features['auxiliary'], current['auxiliary'])
                Q, v, mask, residual = rigid.consensus(a, b, ua, ub, cameras.adapter_rotation(relative_gyro), self.frame)
                local_R, t = cameras.body_pose_from_adapter(Q, v)
                R, p = ref.rotation@local_R, ref.position+ref.rotation@t
                if np.linalg.norm(t) > rigid.MAX_REFERENCE_TRANSLATION_M:
                    raise Rejected('auxiliary measurement too far from its reference')
                A, offset = cameras.ADAPTER_ROTATION, cameras.ADAPTER_OFFSET
                points = a[mask]@A.T+offset, b[mask]@A.T+offset
            rigid.check_step(np.zeros(3), R, p, self.last_R, self.last_p)
            registration = Registration(*points, np.asarray(ua[mask], float), np.asarray(ub[mask], float),
                                        lifted_matches=len(a), residual_rms_m=float(np.sqrt(np.mean(residual[mask]**2))))
            return Candidate(ref, R, p, local_R, t, registration)
        except Rejected:
            if not (ref is self.previous and self.frame-ref.frame == 1 and self.camera == 'auxiliary'):
                raise
        return self._register_joint(ref, current, relative_gyro, retained_anchor)

    def _register_joint(self, ref, current, relative_gyro, retained_anchor):
        views = [match(ref.features[camera], current[camera]) for camera in ('primary', 'auxiliary')]
        A, offset = cameras.ADAPTER_ROTATION, cameras.ADAPTER_OFFSET
        (pa, pb, pu, pv), (aa, ab, au, av) = views
        a = np.concatenate((pa, aa@A.T+offset))
        b = np.concatenate((pb, ab@A.T+offset))
        ua, ub = np.concatenate((pu, au)), np.concatenate((pv, av))
        split = len(pa)
        try:
            local_R, t, mask, residual = rigid.consensus(a, b, ua, ub, relative_gyro, self.frame,
                                                         project=lambda points: cameras.project_views(points, split))
        except Rejected:
            raise UnportedBranch('joint registration failed: the original tries chained flow association')
        counts = [int(mask[:split].sum()), int(mask[split:].sum())]
        if min(counts) < 3:
            raise UnportedBranch('joint registration with fewer than three points in one camera')
        R, p = ref.rotation@local_R, ref.position+ref.rotation@t
        if (np.linalg.norm(p-self.last_p) > rigid.MAX_STEP_TRANSLATION_M
                or angle(self.last_R.T@R) > rigid.MAX_STEP_ROTATION_RAD):
            raise Rejected('joint-camera consecutive displacement envelope rejected')
        registration = Registration(a[mask], b[mask], np.asarray(ua[mask], float), np.asarray(ub[mask], float),
                                    lifted_matches=len(a), residual_rms_m=float(np.sqrt(np.mean(residual[mask]**2))),
                                    split=counts[0], joint_retained_anchor=retained_anchor)
        return Candidate(ref, R, p, local_R, t, registration)
