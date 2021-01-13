import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

# Decompose an essential matrix into the possible rotations and translations.
#
# The first pose is assumed to be P = [I | 0] and the set of four other
# possible second poses are defined as: {[R1 | t], [R2 | t],
#                                        [R1 | -t], [R2 | -t]}
#
# @param E          3x3 essential matrix.
# @param R1         First possible 3x3 rotation matrix.
# @param R2         Second possible 3x3 rotation matrix.
# @param t          3x1 possible translation vector (also -t possible).
def decompose_essential_matrix(E):
    assert E.shape == (3, 3)
    U, S, VH = np.linalg.svd(E)
    V = VH.T
    if np.linalg.det(U) < 0.0:
        U *= -1.0
    if np.linalg.det(V) < 0.0:
        V *= -1.0
    W = np.array([0, 1, 0, -1, 0, 0, 0, 0, 1], dtype=np.float).reshape(3, 3)
    R1 = U.dot(W).dot(V)
    R2 = U.dot(W.T).dot(V)
    t = U[:, 2]
    norm = np.linalg.norm(t)
    t /= norm
    return R1, R2, t

# Recover the most probable pose from the given essential matrix.
#
# The pose of the first image is assumed to be P = [I | 0].
#
# @param E            3x3 essential matrix.
# @param points1      First set of corresponding points.
# @param points2      Second set of corresponding points.
# @param inlier_mask  Only points with `true` in the inlier mask are
#                     considered in the cheirality test. Size of the
#                     inlier mask must match the number of points N.
# @param R            Most probable 3x3 rotation matrix.
# @param t            Most probable 3x1 translation vector.
# @param points3D     Triangulated 3D points infront of camera.
def pose_from_essential_matrix(E, points1, points2):
    assert points1.shape == points2.shape
    R1, R2, t = decompose_essential_matrix(E)
    R_cmbs = [R1, R2, R1, R2]
    t_cmbs = [t, t, -t, -t]

    points3D = None
    R = None
    t = None
    for i in range(4):
        points3D_cmb = pyVO.base.CheckCheirality(R_cmbs[i], t_cmbs[i], points1, points2)
        if points3D is None or len(points3D_cmb) > len(points3D):
            points3D = points3D_cmb
            R = R_cmbs[i]
            t = t_cmbs[i]

    points3D = np.array(points3D)
    return R, t, points3D

# Compose essential matrix from relative camera poses.
#
# Assumes that first camera pose has projection matrix P = [I | 0], and
# pose of second camera is given as transformation from world to camera system.
#
# @param R             3x3 rotation matrix.
# @param t             3x1 translation vector.
#
# @return              3x3 essential matrix.
def essential_matrix_from_pose(R, t):
    assert R.shape == (3, 3) and t.shape[0] == 3
    tnorm = np.linalg.norm(t)
    t /= tnorm
    tcross = np.zeros([3, 3])
    tcross[0, 1], tcross[0, 2] = -t[2], t[1]
    tcross[1, 0], tcross[1, 2] = t[2], -t[0]
    tcross[2, 0], tcross[2, 1] = -t[1], t[0]
    return tcross.dot(R)

# Compute the location of the epipole in homogeneous coordinates.
#
# @param E           3x3 essential matrix.
# @param left_image  If true, epipole in left image is computed,
#                    else in right image.
#
# @return            Epipole in homogeneous coordinates.
def epipole_from_essential_matrix(E, is_left):
    assert type(is_left) == type(True) and E.shape == (3, 3)
    if is_left:
        U, S, VH = np.linalg.svd(E)
        return VH[:, 2]
    U, S, VH = np.linalg.svd(E.T)
    return VH[:, 2]