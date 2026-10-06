"""The floor as both depth cameras measure it: candidate points, the joint floor plane and the paired floor constraint.

**Candidates.** Each depth image is sampled every 4 px. A sample is a candidate if its 3 x 3 neighbourhood is valid and
it lies more than 15 cm below the body (along the given up direction). The candidates of both cameras are pooled
and the floor is taken as the dominant height cluster: the densest 6-mm height slab, refined by plane fits until
every point is within 3 mm (at most 8 rounds; at least a quarter of the pool must remain).

**Joint plane.** One plane through the selected points of both cameras (counts-weighted moments): available if it has
at least 100 points, a second covariance eigenvalue of at least (2 cm)², a normal within acos(0.97) of up, and every
point within 3 mm.

**Paired floor constraint.** For a reference and a current frame related by a gyro rotation: one common normal fitted
to both frames' raw floor points (the current frame's rotated into the reference), refined until every point is within
3 mm. It fixes the relative height (translation along the normal) of the visual odometry's fit.
"""
import numpy as np

from go2nav.perception import cameras
from go2nav.perception.rigid import Rejected

SAMPLE_ROWS, SAMPLE_COLUMNS = np.arange(2, 480, 4), np.arange(2, 640, 4)
BELOW_BODY_M = -.15
SLAB_RESIDUAL_M = .003
MIN_CLUSTER_FRACTION = .25
MAX_REFINEMENTS = 8
MIN_PLANE_POINTS = 100
# 2 cm second extent: the original configures this at start-up (two_cm_floor_extent_development.configure).
MIN_SECOND_EXTENT_M = .02
MIN_UP_ALIGNMENT = .97
CAMERAS = (('primary', cameras.PRIMARY), ('auxiliary', cameras.AUXILIARY))


def candidates(depth_m, valid, mount, up):
    """Sampled points below the body whose 3 x 3 neighbourhood is valid (body frame)."""
    yy, xx = np.meshgrid(SAMPLE_ROWS+.5, SAMPLE_COLUMNS+.5, indexing='ij')
    z = depth_m[np.ix_(SAMPLE_ROWS, SAMPLE_COLUMNS)]
    optical = np.stack((z*(xx-320)/cameras.FOCAL, z*(yy-240)/cameras.FOCAL, z), axis=-1)
    xyz = optical@mount[:3, :3].T+mount[:3, 3]
    good = xyz@up < BELOW_BODY_M
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            good &= valid[np.ix_(SAMPLE_ROWS+dr, SAMPLE_COLUMNS+dc)]
    return xyz[good]


def dominant_height_cluster(clouds, up):
    """Masks (one per cloud) selecting the pooled floor cluster; empty masks if there is none."""
    points = np.concatenate([np.asarray(p, float) for p in clouds])
    keep = np.zeros(len(points), bool)
    if len(points):
        height = points@up
        order = np.argsort(height, kind='stable')
        h = height[order]
        right = np.searchsorted(h, h+2*SLAB_RESIDUAL_M, side='right')
        left = int(np.argmax(right-np.arange(len(h))))
        keep[order[left:right[left]]] = True
    initial = int(keep.sum())
    if initial >= 3 and initial >= MIN_CLUSTER_FRACTION*len(points):
        for _ in range(MAX_REFINEMENTS):
            selected = points[keep]
            center = selected.mean(0)
            delta = selected-center
            _, vectors = np.linalg.eigh(delta.T@delta/len(selected))
            normal = vectors[:, 0]
            if normal@up < 0:
                normal = -normal
            if normal@up < MIN_UP_ALIGNMENT:
                keep[:] = False
                break
            residual = np.abs(selected@normal-float(normal@center))
            if residual.max() <= SLAB_RESIDUAL_M:
                break
            ids = np.flatnonzero(keep)
            keep[ids[residual > SLAB_RESIDUAL_M]] = False
            if keep.sum() < 3 or keep.sum() < MIN_CLUSTER_FRACTION*len(points):
                keep[:] = False
                break
        else:
            keep[:] = False
    else:
        keep[:] = False
    split = len(clouds[0])
    return keep[:split], keep[split:]


def paired_floor_points(primary, auxiliary, up):
    """Selected floor points of both cameras' depth ((depth_m, valid) pairs) and the size of the pooled candidates."""
    clouds = [candidates(depth_m, valid, mount, up) for (depth_m, valid), (_, mount) in zip((primary, auxiliary), CAMERAS)]
    masks = dominant_height_cluster(clouds, up)
    return [cloud[keep] for cloud, keep in zip(clouds, masks)], sum(map(len, clouds))


def _compose(statistics, up):
    populated = [(count, mean, covariance) for count, mean, covariance in statistics if count]
    total = sum(count for count, _, _ in statistics)
    plane = dict(available=False, candidate_count=total, min_second_eigenvalue_m2=MIN_SECOND_EXTENT_M**2)
    if total < MIN_PLANE_POINTS:
        return plane | dict(reason='insufficient_combined_measured_candidates')
    mean = sum(count*center for count, center, _ in populated)/total
    covariance = sum(count*(cov+np.outer(center-mean, center-mean)) for count, center, cov in populated)/total
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    if normal@up < 0:
        normal = -normal
    plane.update(normal_body=normal, offset_body_m=-float(normal@mean))
    if values[1] < MIN_SECOND_EXTENT_M**2:
        return plane | dict(reason='insufficient_combined_two_axis_extent')
    if normal@up < MIN_UP_ALIGNMENT:
        return plane | dict(reason='combined_plane_up_disagreement')
    return plane | dict(available=True, reason=None)


def joint_plane(primary_points, auxiliary_points, up):
    """The floor plane through both cameras' selected points (dict: available, reason, normal_body, offset_body_m,
    candidate_count, min_second_eigenvalue_m2, up_body)."""
    statistics, clouds = [], []
    for points in (primary_points, auxiliary_points):
        points = np.asarray(points, float)
        if len(points):
            mean = points.mean(0)
            statistics.append((len(points), mean, (points-mean).T@(points-mean)/len(points)))
        else:
            statistics.append((0, None, None))
        clouds.append(points)
    plane = _compose(statistics, up) | dict(up_body=np.asarray(up, float))
    if not plane['available']:
        return plane
    for points in clouds:
        if len(points) and float(np.abs(points@plane['normal_body']+plane['offset_body_m']).max()) > SLAB_RESIDUAL_M:
            return plane | dict(available=False, reason='combined_points_not_one_coherent_plane')
    return plane


def floor_constraint(reference_points, current_points, gyro, *, reference_pool, current_pool, reference_up, current_up,
                     min_second_eigenvalue_m2):
    """The paired floor constraint between two frames (see the module docstring), or `Rejected`.

    Returns the common normal and the floor offset in each frame's body coordinates."""
    G = gyro
    arrays = [np.asarray(p, float) for p in (reference_points, current_points)]
    pools = (reference_pool, current_pool)
    ups = [np.asarray(u, float) for u in (reference_up, current_up)]
    masks = [np.ones(len(p), bool) for p in arrays]
    for _ in range(MAX_REFINEMENTS):
        selected = [p[m] for p, m in zip(arrays, masks)]
        if any(len(p) < max(100, MIN_CLUSTER_FRACTION*pool) for p, pool in zip(selected, pools)):
            raise Rejected('insufficient paired floor support after pruning')
        means, covs = [], []
        for chosen in selected:
            mean = chosen.mean(0)
            covariance = (chosen-mean).T@(chosen-mean)/len(chosen)
            if np.linalg.eigvalsh(covariance)[1] < min_second_eigenvalue_m2:
                raise Rejected('paired floor patch lacks two-axis extent')
            means.append(mean)
            covs.append(covariance)
        total = sum(map(len, selected))
        _, vectors = np.linalg.eigh((len(selected[0])*covs[0]+len(selected[1])*G@covs[1]@G.T)/total)
        n = vectors[:, 0]
        if n@ups[0] < 0:
            n = -n
        normals = (n, G.T@n)
        if any(normal@up < MIN_UP_ALIGNMENT for normal, up in zip(normals, ups)):
            raise Rejected('paired floor normal disagrees with measured up')
        residuals = [np.abs((p-m)@normal) for p, m, normal in zip(selected, means, normals)]
        if all(r.max() <= SLAB_RESIDUAL_M for r in residuals):
            return dict(reference_normal_body=n, reference_offset_m=-float(n@means[0]),
                        current_offset_m=-float((G.T@n)@means[1]))
        for mask, residual in zip(masks, residuals):
            mask[np.flatnonzero(mask)[residual > SLAB_RESIDUAL_M]] = False
    raise Rejected('bounded paired floor refinement did not converge')
