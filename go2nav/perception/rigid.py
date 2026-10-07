"""Rigid registration of matched RGB-D points, with the gyro rotation and the measured floor plane.

The visual odometry's relative pose between a reference frame and the current frame comes from matched, depth-lifted
corners (`features.match`), in three steps:
1. `consensus`: robust translation with the rotation fixed to the gyro's relative rotation (best of the full set and
   128 seeded three-point proposals, then pruning to a stable inlier set; a strict inlier majority is required);
2. `refine_on_planes`: the inlier set refitted with both frames' measured floor planes as constraints (rotation and
   translation); used for its admission checks and its pose, against which step 3 must agree. If the refit loses an
   image inlier, the image fit is kept when it agrees with the plane fit (`refine_if_supported`); if it then
   contradicts the measured floor height by more than 3 mm, `plane_consensus` prunes the inliers to a plane-consistent
   set and refines that;
3. `gyro_consensus_refit`: the final pose, with the gyro rotation and the translation's height fixed by the
   gyro-coherent floor constraint (`floor.FloorConstraint`), after re-pruning the inliers.

An inlier is within 2 cm in 3-D and reprojects within 1 px in both images. Any failed gate raises `Rejected`.

Points are in the body frame; for the auxiliary camera alone they are in the reference-adapter frame (see `cameras`)
and the relative pose is converted to the body frame afterwards.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from go2nav.perception import cameras
from go2nav.perception.features import grid_cells
from go2nav.perception.rotations import angle


class Rejected(Exception):
    """A measurement fails a gate (not an error: the caller tries another reference or camera)."""


class LosesImageInlier(Rejected):
    """The plane-constrained refit no longer explains every image inlier."""


class PlaneImageConflict(Rejected):
    """The retained image fit disagrees with the measured floor height."""


class PoseLost(RuntimeError):
    """The visual pose is lost (terminal, as in the original tracker): the mission ends as a pose loss."""


class UnportedBranch(NotImplementedError):
    """A branch of the original runtime that no traced mission reached, so it was not ported.

    Reaching it means this run diverges from the original: port the branch (see docs/porting/RUNTIME_PORT.md)."""


MIN_MATCHES = 12
MIN_INLIER_FRACTION = .5
INLIER_RESIDUAL_M, REPROJECTION_PX = .02, 1.
PROPOSALS, PROPOSAL_SEED = 128, 2026090631
MIN_SECOND_SCATTER_M, MIN_SCATTER_RATIO = .02, .05
MAX_GYRO_DISAGREEMENT_RAD = .10
MAX_REFERENCE_TRANSLATION_M = 3.
MAX_STEP_TRANSLATION_M, MAX_STEP_ROTATION_RAD = .15, .20
MAX_MEASUREMENT_DISAGREEMENT_M, MAX_MEASUREMENT_DISAGREEMENT_RAD = .02, .10
MAX_FLOOR_OFFSET_RESIDUAL_M = .003


@dataclass
class Registration:
    """A fitted correspondence set between a reference and the current frame."""
    reference_points: np.ndarray
    current_points: np.ndarray
    reference_pixels: np.ndarray
    current_pixels: np.ndarray
    lifted_matches: int
    residual_rms_m: float
    split: int | None = None  # joint fits: the number of primary-camera points (the rest are auxiliary)
    association: str = 'descriptor'  # or 'direct_flow', 'chained_flow' (see features)
    floor_constraint: dict | None = None
    plane_refinement_applied: bool | None = None
    joint_retained_anchor: bool = False

    @property
    def inliers(self):
        return len(self.reference_points)

    @property
    def inlier_fraction(self):
        return float(self.inliers/self.lifted_matches)

    @property
    def grid_cells(self):
        return grid_cells(self.reference_pixels), grid_cells(self.current_pixels)

    def projector(self):
        return cameras.project if self.split is None else (lambda points: cameras.project_views(points, self.split))


@dataclass
class Candidate:
    """A relative pose measurement against one reference: local (reference-body) and global (initial-body) pose."""
    reference: object
    R: np.ndarray
    p: np.ndarray
    local_R: np.ndarray
    t: np.ndarray
    registration: Registration
    extra: dict = field(default_factory=dict)


def scatter(points):
    """Singular values / sqrt(n) of centred points; rejects nearly collinear or tightly clustered supports."""
    p = np.asarray(points, float)
    rms = np.linalg.svd(p-p.mean(0), compute_uv=False)/np.sqrt(len(p))
    if rms[1] < MIN_SECOND_SCATTER_M or rms[1] < MIN_SCATTER_RATIO*rms[0]:
        raise Rejected('noncollinear well-spread point support required')
    return rms


def gyro_fit(a, b, G):
    """Translation with the rotation fixed to G (both supports must be well spread)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scatter(a), scatter(b)
    return G, a.mean(0)-G@b.mean(0)


def inliers(a, b, ua, ub, R, t, project=cameras.project):
    """Which correspondences fit (R, t): 3-D residual and reprojection into both images; and the residuals."""
    residual = np.linalg.norm(a-b@R.T-t, axis=1)
    current, cvalid = project((a-t)@R)
    reference, rvalid = project(b@R.T+t)
    keep = (residual <= INLIER_RESIDUAL_M)&cvalid&rvalid
    keep &= np.linalg.norm(current-ub, axis=1) <= REPROJECTION_PX
    keep &= np.linalg.norm(reference-ua, axis=1) <= REPROJECTION_PX
    return keep, residual


def _proposals(a, b, G, frame):
    """The full-set gyro fit, then the 128 seeded three-point gyro fits whose supports are well spread."""
    try:
        yield gyro_fit(a, b, G)
    except Rejected:
        pass
    rng = np.random.default_rng(np.random.SeedSequence([PROPOSAL_SEED, frame]))
    indices = np.array([rng.choice(len(a), 3, replace=False) for _ in range(PROPOSALS)])
    aa, bb = a[indices], b[indices]
    ma, mb = aa.mean(axis=1), bb.mean(axis=1)
    sa = np.linalg.svd(aa-ma[:, None, :], compute_uv=False)/np.sqrt(3)
    sb = np.linalg.svd(bb-mb[:, None, :], compute_uv=False)/np.sqrt(3)
    valid = np.ones(len(indices), dtype=bool)
    for s in (sa, sb):
        valid &= s[:, 1] >= MIN_SECOND_SCATTER_M
        valid &= s[:, 1] >= MIN_SCATTER_RATIO*s[:, 0]
    for i in np.flatnonzero(valid):
        yield G, ma[i]-G@mb[i]


def consensus(a, b, ua, ub, G, frame, project=cameras.project):
    """Robust gyro-conditioned translation: (R, t, inlier mask, residuals)."""
    a, b, ua, ub = [np.asarray(x, float) for x in (a, b, ua, ub)]
    if len(a) < MIN_MATCHES:
        raise Rejected('insufficient rigid-pose matches')
    best = rank = None
    for R, t in _proposals(a, b, G, frame):
        mask, residual = inliers(a, b, ua, ub, R, t, project)
        count = int(mask.sum())
        score = (count, -float(np.sum(residual[mask]**2)))
        if best is None or score > rank:
            best, rank = mask, score
        if count == len(a):
            break
    if best is None:
        raise Rejected('no conditioned rigid-pose proposal')
    mask = best.copy()
    while True:
        if mask.sum() < MIN_MATCHES:
            raise Rejected('insufficient rigid consensus after pruning')
        R, t = gyro_fit(a[mask], b[mask], G)
        good, residual = inliers(a, b, ua, ub, R, t, project)
        use = mask&good
        if np.array_equal(use, mask):
            break
        mask = use
    if mask.mean() < MIN_INLIER_FRACTION or np.linalg.norm(t) > MAX_REFERENCE_TRANSLATION_M:
        raise Rejected('rigid consensus fraction or displacement rejected')
    if 2*int(mask.sum()) <= len(mask):
        raise Rejected('strict inlier majority required')
    return R, t, mask, residual


def check_step(t, R_global, p_global, last_R, last_p):
    """The envelope every measurement must pass: displacement from its reference, and step from the last pose."""
    if (np.linalg.norm(t) > MAX_REFERENCE_TRANSLATION_M
            or np.linalg.norm(p_global-last_p) > MAX_STEP_TRANSLATION_M
            or angle(last_R.T@R_global) > MAX_STEP_ROTATION_RAD):
        raise Rejected('measurement outside the motion envelope')


def plane_fit(a, b, reference_plane, current_plane):
    """Least-squares rigid fit of all correspondences with both measured floor planes imposed.

    The rotation maps the current floor normal onto the reference normal and fits the in-plane yaw; the translation is
    the centroid fit corrected so the floor heights agree."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scatter(a), scatter(b)
    na, nb = np.asarray(reference_plane['normal_body'], float), np.asarray(current_plane['normal_body'], float)
    da, db = float(reference_plane['offset_body_m']), float(current_plane['offset_body_m'])

    def basis(n):
        e = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = e-n*float(n@e)
        u /= np.linalg.norm(u)
        return np.column_stack((u, np.cross(n, u), n))
    A, B = basis(na), basis(nb)
    ac, bc = a-a.mean(0), b-b.mean(0)
    x, y = ac@A[:, :2], bc@B[:, :2]
    cosine = float(np.sum(x*y))
    sine = float(np.sum(y[:, 0]*x[:, 1]-y[:, 1]*x[:, 0]))
    scale = float(np.linalg.norm(x)*np.linalg.norm(y))
    magnitude = float(np.hypot(cosine, sine))
    if not np.isfinite([scale, magnitude]).all() or scale <= 0 or magnitude <= 1e-12*scale:
        raise Rejected('in-plane rotation unidentifiable')
    c, s = cosine/magnitude, sine/magnitude
    R = A@np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])@B.T
    unconstrained_t = a.mean(0)-R@b.mean(0)
    return R, unconstrained_t+na*(db-da-float(na@unconstrained_t))


def _image_frame(a, b, R, t, camera):
    """Points and pose as the image projection expects them (auxiliary fits are checked in the adapter frame)."""
    if camera == 'primary':
        return a, b, R, t
    A, offset = cameras.ADAPTER_ROTATION, cameras.ADAPTER_OFFSET
    return (a-offset)@A, (b-offset)@A, A.T@R@A, A.T@(t-offset+R@offset)


def refine_on_planes(candidate, reference_plane, current_plane, *, camera, gyro, last_p, last_R):
    """Step 2: refit the candidate's inliers with both floor planes; every image inlier must survive."""
    reg = candidate.registration
    a, b, ua, ub = reg.reference_points, reg.current_points, reg.reference_pixels, reg.current_pixels
    R, t = plane_fit(a, b, reference_plane, current_plane)
    if reg.split is not None:
        camera = 'primary'  # joint fits are already in body coordinates
    image_a, image_b, image_R, image_t = _image_frame(a, b, R, t, camera)
    keep, residual = inliers(image_a, image_b, ua, ub, image_R, image_t, reg.projector())
    if not keep.all():
        raise LosesImageInlier('plane refinement loses an original image inlier')
    ref = candidate.reference
    global_R, global_p = ref.rotation@R, ref.position+ref.rotation@t
    check_step(t, global_R, global_p, last_R, last_p)
    if angle(gyro.T@R) > MAX_GYRO_DISAGREEMENT_RAD:
        raise Rejected('plane refinement disagrees with the gyro')
    if (float(np.linalg.norm(global_p-candidate.p)) > MAX_MEASUREMENT_DISAGREEMENT_M
            or angle(candidate.R.T@global_R) > MAX_MEASUREMENT_DISAGREEMENT_RAD):
        raise Rejected('plane refinement moves the image fit too far')
    refined = replace(reg, residual_rms_m=float(np.sqrt(np.mean(residual**2))), plane_refinement_applied=True)
    return Candidate(ref, global_R, global_p, R, t, refined)


def refine_if_supported(candidate, reference_plane, current_plane, *, camera, gyro, last_p, last_R):
    """Step 2, keeping the image fit if the plane refit loses an inlier but the image fit agrees with the floor height."""
    try:
        return refine_on_planes(candidate, reference_plane, current_plane, camera=camera, gyro=gyro,
                                last_p=last_p, last_R=last_R)
    except LosesImageInlier:
        pass
    reg = candidate.registration
    R, t = plane_fit(reg.reference_points, reg.current_points, reference_plane, current_plane)
    ref = candidate.reference
    global_R, global_p = ref.rotation@R, ref.position+ref.rotation@t
    if (float(np.linalg.norm(global_p-candidate.p)) > MAX_MEASUREMENT_DISAGREEMENT_M
            or angle(candidate.R.T@global_R) > MAX_MEASUREMENT_DISAGREEMENT_RAD):
        raise Rejected('qualified image and measured-plane estimates conflict')
    check_step(t, global_R, global_p, last_R, last_p)
    if angle(gyro.T@R) > MAX_GYRO_DISAGREEMENT_RAD:
        raise Rejected('plane refit disagrees with the gyro')
    na = np.asarray(reference_plane['normal_body'])
    offset_residual = abs(float(na@candidate.t-current_plane['offset_body_m']+reference_plane['offset_body_m']))
    if offset_residual > MAX_FLOOR_OFFSET_RESIDUAL_M:
        try:
            return plane_consensus(candidate, reference_plane, current_plane, camera=camera, gyro=gyro,
                                   last_p=last_p, last_R=last_R)
        except Rejected:
            raise Rejected('retained image fit conflicts with measured floor height') from None
    return replace(candidate, registration=replace(reg, plane_refinement_applied=False))


def plane_consensus(candidate, reference_plane, current_plane, *, camera, gyro, last_p, last_R):
    """Step 2 for an image fit that conflicts with the measured floor height: prune its inliers to those the
    plane-constrained fit explains (refitting until stable; at least 12 points, a strict majority of the original
    matches and, for joint fits, 3 per camera), then refine on the pruned set with all of step 2's checks."""
    reg = candidate.registration
    a, b, ua, ub = reg.reference_points, reg.current_points, reg.reference_pixels, reg.current_pixels
    check = image_check(reg, camera)
    mask = np.ones(len(a), dtype=bool)
    while True:
        count = int(mask.sum())
        if count < MIN_MATCHES or 2*count <= reg.lifted_matches:
            raise Rejected('plane/image consensus requires the match count and a strict majority')
        if reg.split is not None and min(int(mask[:reg.split].sum()), int(mask[reg.split:].sum())) < 3:
            raise Rejected('plane/image consensus needs support from both cameras')
        R, t = plane_fit(a[mask], b[mask], reference_plane, current_plane)
        good, _ = check(a, b, ua, ub, R, t)
        use = mask&good
        if np.array_equal(use, mask):
            break
        mask = use
    split = None if reg.split is None else int(mask[:reg.split].sum())
    pruned = replace(reg, reference_points=a[mask], current_points=b[mask], reference_pixels=ua[mask],
                     current_pixels=ub[mask], split=split)
    return refine_on_planes(replace(candidate, registration=pruned), reference_plane, current_plane, camera=camera,
                            gyro=gyro, last_p=last_p, last_R=last_R)


def image_check(reg, camera):
    """`inliers` for a registration's body-frame points: joint fits project each point into its own camera; auxiliary
    fits are checked in the adapter frame."""
    if reg.split is not None:
        project = reg.projector()
        return lambda a, b, ua, ub, R, t: inliers(a, b, ua, ub, R, t, project)
    if camera == 'auxiliary':
        A, offset = cameras.ADAPTER_ROTATION, cameras.ADAPTER_OFFSET
        return lambda a, b, ua, ub, R, t: inliers((a-offset)@A, (b-offset)@A, ua, ub, A.T@R@A, A.T@(t-offset+R@offset))
    return inliers


def constrain_translation(t, constraint):
    """Impose the floor constraint's height equation on a translation."""
    normal = np.asarray(constraint['reference_normal_body'])
    return t+normal*(constraint['current_offset_m']-constraint['reference_offset_m']-normal@t)


def _gyro_refit(candidate, reg, gyro, *, camera, last_p, last_R):
    """Refit all of `reg`'s correspondences with the gyro rotation and the floor constraint."""
    a, b, ua, ub = reg.reference_points, reg.current_points, reg.reference_pixels, reg.current_pixels
    R, t = gyro_fit(a, b, gyro)
    if reg.floor_constraint is not None:
        t = constrain_translation(t, reg.floor_constraint)
    elif reg.plane_refinement_applied:
        raise UnportedBranch('measured-plane height constraint without a gyro-coherent floor constraint')
    if reg.split is not None:
        keep, residual = inliers(a, b, ua, ub, R, t, reg.projector())
    elif camera == 'auxiliary':
        A, offset = cameras.ADAPTER_ROTATION, cameras.ADAPTER_OFFSET
        keep, residual = inliers((a-offset)@A, (b-offset)@A, ua, ub, A.T@R@A, A.T@(t-offset+R@offset))
    else:
        keep, residual = inliers(a, b, ua, ub, R, t)
    if not keep.all():
        raise Rejected('gyro refit loses an image inlier')
    ref = candidate.reference
    p, global_R = ref.position+ref.rotation@t, ref.rotation@R
    check_step(t, global_R, p, last_R, last_p)
    if (np.linalg.norm(p-candidate.p) > MAX_MEASUREMENT_DISAGREEMENT_M
            or angle(candidate.R.T@global_R) > MAX_MEASUREMENT_DISAGREEMENT_RAD):
        raise Rejected('gyro refit moves the measurement too far')
    return Candidate(ref, global_R, p, R, t, replace(reg, residual_rms_m=float(np.sqrt(np.mean(residual**2)))))


def gyro_consensus_refit(candidate, gyro, *, camera, last_p, last_R):
    """Step 3: prune the inliers to those consistent with the gyro rotation and floor height, then refit.

    The pruned set must keep at least 12 points, a strict majority of the original matches and, for joint fits, at
    least 3 points per camera."""
    reg = candidate.registration
    a, b, ua, ub = reg.reference_points, reg.current_points, reg.reference_pixels, reg.current_pixels
    check = image_check(reg, camera)
    mask = np.ones(len(a), dtype=bool)
    while True:
        count = int(mask.sum())
        if count < MIN_MATCHES or 2*count <= reg.lifted_matches:
            raise Rejected('gyro consensus requires the match count and a strict majority')
        if reg.split is not None and min(int(mask[:reg.split].sum()), int(mask[reg.split:].sum())) < 3:
            raise Rejected('gyro consensus needs support from both cameras')
        R, t = gyro_fit(a[mask], b[mask], gyro)
        if reg.floor_constraint is not None:
            t = constrain_translation(t, reg.floor_constraint)
        elif reg.plane_refinement_applied:
            raise UnportedBranch('measured-plane height constraint without a gyro-coherent floor constraint')
        good, _ = check(a, b, ua, ub, R, t)
        use = mask&good
        if np.array_equal(use, mask):
            break
        mask = use
    split = None if reg.split is None else int(mask[:reg.split].sum())
    pruned = replace(reg, reference_points=a[mask], current_points=b[mask], reference_pixels=ua[mask],
                     current_pixels=ub[mask], split=split)
    return _gyro_refit(candidate, pruned, gyro, camera=camera, last_p=last_p, last_R=last_R)
