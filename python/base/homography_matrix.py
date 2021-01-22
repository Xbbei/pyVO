import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

# Decompose an homography matrix into the possible rotations, translations,
# and plane normal vectors, according to:
#
#    Malis, Ezio, and Manuel Vargas. "Deeper understanding of the homography
#    decomposition for vision-based control." (2007): 90.
#
# The first pose is assumed to be P = [I | 0]. Note that the homography is
# plane-induced if `R.size() == t.size() == n.size() == 4`. If `R.size() ==
# t.size() == n.size() == 1` the homography is pure-rotational.
#
# @param H          3x3 homography matrix.
# @param K          3x3 calibration matrix.
# @param Rs         Possible 3x3 rotation matrices.
# @param ts         Possible translation vectors.
# @param ns         Possible normal vectors.
def decompose_homography_matrix(H, K1, K2):
    assert H.shape == (3, 3) and K1.shape == (3, 3) and K2.shape == (3, 3)
    homography2pose = pyVO.base.Homography2Pose()
    homography2pose.DecomposeHomographyMatrix(H, K1, K2)
    Rs = homography2pose.GetRotationVector()
    ts = homography2pose.GetTranslationVector()
    ns = homography2pose.GetNormalVector()
    return Rs, ts, ns

# Recover the most probable pose from the given homography matrix.
#
# The pose of the first image is assumed to be P = [I | 0].
#
# @param H            3x3 homography matrix.
# @param K1           3x3 calibration matrix of first camera.
# @param K2           3x3 calibration matrix of second camera.
# @param points1      First set of corresponding points.
# @param points2      Second set of corresponding points.
# @param inlier_mask  Only points with `true` in the inlier mask are
#                     considered in the cheirality test. Size of the
#                     inlier mask must match the number of points N.
# @param R            Most probable 3x3 rotation matrix.
# @param t            Most probable 3x1 translation vector.
# @param n            Most probable 3x1 normal vector.
# @param points3D     Triangulated 3D points infront of camera
#                     (only if homography is not pure-rotational).
def pose_from_homography_matrix(H, K1, K2, points1, points2):
    assert H.shape == (3, 3) and K1.shape == (3, 3) and K2.shape == (3, 3)
    assert len(points1) == len(points2)
    homography2pose = pyVO.base.Homography2Pose()
    homography2pose.PoseFromHomographyMatrix(H, K1, K2, points1, points2)
    R = homography2pose.GetRotation()
    t = homography2pose.GetTranslation()
    n = homography2pose.GetNormal()
    points3D = homography2pose.GetPoints3D()
    return R, t, n, points3D

# Compose homography matrix from relative pose.
#
# @param K1      3x3 calibration matrix of first camera.
# @param K2      3x3 calibration matrix of second camera.
# @param R       Most probable 3x3 rotation matrix.
# @param t       Most probable 3x1 translation vector.
# @param n       Most probable 3x1 normal vector.
# @param d       Orthogonal distance from plane.
#
# @return        3x3 homography matrix.
def homography_matrix_from_pose(K1, K2, R, t, n, d):
    assert K1.shape == (3, 3) and K2.shape == (3, 3) and R.shape == (3, 3) and t.shape[0] == 3
    H = pyVO.base.HomographyMatrixFromPose(K1, K2, R, t, n, d)
    return H

if __name__ == "__main__":
    H = np.array([2.649157564634028, 4.583875997496426, 70.694447785121326,
              -1.072756858861583, 3.533262150437228, 1513.656999614321649,
              0.001303887589576, 0.003042206876298, 1], dtype=np.float64).reshape(3, 3) * 3.0

    K = np.array([640, 0, 320, 0, 640, 240, 0, 0, 1], dtype=np.float64).reshape(3, 3)

    Rs, ts, ns = decompose_homography_matrix(H, K, K)

    print(len(Rs) == 4)
    print(len(ts) == 4)
    print(len(ns) == 4)

    R_ref = np.array([0.43307983549125, 0.545749113549648, -0.717356090899523,
        -0.85630229674426, 0.497582023798831, -0.138414255706431,
        0.281404038139784, 0.67421809131173, 0.682818960388909
    ], dtype=np.float64).reshape(3, 3)
    t_ref = np.array([1.826751712278038, 1.264718492450820, 0.195080809998819], dtype=np.float64)
    n_ref = np.array([-0.244875830334816, -0.480857890778889, -0.841909446789566], dtype=np.float64)

    ref_solution_exists = False
    for i in range(4):
        delta_R = Rs[i] - R_ref
        delta_t = ts[i] - t_ref
        delta_n = ns[i] - n_ref
        if np.linalg.norm(delta_R) < 1e-6 and \
            np.linalg.norm(delta_t) < 1e-6 and \
            np.linalg.norm(delta_n) < 1e-6:
            ref_solution_exists = True

    print(ref_solution_exists)