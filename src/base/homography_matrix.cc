#include "src/base/homography_matrix.h"

#include <Eigen/Dense>

#include "src/util/math.h"
#include "src/base/pose.h"
#include "src/util/types.h"
#include "src/util/logging.h"

double ComputeOppositeOfMinor(const Eigen::Matrix3d& matrix, const size_t row,
                              const size_t col) {
  const size_t col1 = col == 0 ? 1 : 0;
  const size_t col2 = col == 2 ? 1 : 2;
  const size_t row1 = row == 0 ? 1 : 0;
  const size_t row2 = row == 2 ? 1 : 2;
  return (matrix(row1, col2) * matrix(row2, col1) -
          matrix(row1, col1) * matrix(row2, col2));
}

Eigen::Matrix3d ComputeHomographyRotation(const Eigen::Matrix3d& H_normalized,
                                          const Eigen::Vector3d& tstar,
                                          const Eigen::Vector3d& n,
                                          const double v) {
  return H_normalized *
         (Eigen::Matrix3d::Identity() - (2.0 / v) * tstar * n.transpose());
}

void Homography2Pose::Reset() {
    Rs.clear();
    ts.clear();
    ns.clear();
    points3D.clear();
    R = Eigen::Matrix3d::Identity();
    t = Eigen::Vector3d::Zero();
    n = Eigen::Vector3d::Zero();
}

void Homography2Pose::DecomposeHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2)
{
    Reset();
    // Remove calibration from homography.
    Eigen::Matrix3d H_normalized = K2.inverse() * H * K1;

    // Remove scale from normalized homography.
    Eigen::JacobiSVD<Eigen::Matrix3d> hmatrix_norm_svd(H_normalized);
    H_normalized.array() /= hmatrix_norm_svd.singularValues()[1];

    // Ensure that we always return rotations, and never reflections.
    //
    // It's enough to take det(H_normalized) > 0.
    //
    // To see this:
    // - In the paper: R := H_normalized * (Id + x y^t)^{-1} (page 32).
    // - Can check that this implies that R is orthogonal: RR^t = Id.
    // - To return a rotation, we also need det(R) > 0.
    // - By Sylvester's idenitity: det(Id + x y^t) = (1 + x^t y), which
    //   is positive by choice of x and y (page 24).
    // - So det(R) and det(H_normalized) have the same sign.
    if (H_normalized.determinant() < 0) {
        H_normalized.array() *= -1.0;
    }

    const Eigen::Matrix3d S =
        H_normalized.transpose() * H_normalized - Eigen::Matrix3d::Identity();

    // Check if H is rotation matrix.
    const double kMinInfinityNorm = 1e-3;
    if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
        Rs = {H_normalized};
        ts = {Eigen::Vector3d::Zero()};
        ns = {Eigen::Vector3d::Zero()};
        return;
    }

    const double M00 = ComputeOppositeOfMinor(S, 0, 0);
    const double M11 = ComputeOppositeOfMinor(S, 1, 1);
    const double M22 = ComputeOppositeOfMinor(S, 2, 2);

    const double rtM00 = std::sqrt(M00);
    const double rtM11 = std::sqrt(M11);
    const double rtM22 = std::sqrt(M22);

    const double M01 = ComputeOppositeOfMinor(S, 0, 1);
    const double M12 = ComputeOppositeOfMinor(S, 1, 2);
    const double M02 = ComputeOppositeOfMinor(S, 0, 2);

    const int e12 = SignOfNumber(M12);
    const int e02 = SignOfNumber(M02);
    const int e01 = SignOfNumber(M01);

    const double nS00 = std::abs(S(0, 0));
    const double nS11 = std::abs(S(1, 1));
    const double nS22 = std::abs(S(2, 2));

    const std::array<double, 3> nS{{nS00, nS11, nS22}};
    const size_t idx =
        std::distance(nS.begin(), std::max_element(nS.begin(), nS.end()));

    Eigen::Vector3d np1;
    Eigen::Vector3d np2;
    if (idx == 0) {
        np1[0] = S(0, 0);
        np2[0] = S(0, 0);
        np1[1] = S(0, 1) + rtM22;
        np2[1] = S(0, 1) - rtM22;
        np1[2] = S(0, 2) + e12 * rtM11;
        np2[2] = S(0, 2) - e12 * rtM11;
    } else if (idx == 1) {
        np1[0] = S(0, 1) + rtM22;
        np2[0] = S(0, 1) - rtM22;
        np1[1] = S(1, 1);
        np2[1] = S(1, 1);
        np1[2] = S(1, 2) - e02 * rtM00;
        np2[2] = S(1, 2) + e02 * rtM00;
    } else if (idx == 2) {
        np1[0] = S(0, 2) + e01 * rtM11;
        np2[0] = S(0, 2) - e01 * rtM11;
        np1[1] = S(1, 2) + rtM00;
        np2[1] = S(1, 2) - rtM00;
        np1[2] = S(2, 2);
        np2[2] = S(2, 2);
    }

    const double traceS = S.trace();
    const double v = 2.0 * std::sqrt(1.0 + traceS - M00 - M11 - M22);

    const double ESii = SignOfNumber(S(idx, idx));
    const double r_2 = 2 + traceS + v;
    const double nt_2 = 2 + traceS - v;

    const double r = std::sqrt(r_2);
    const double n_t = std::sqrt(nt_2);

    const Eigen::Vector3d n1 = np1.normalized();
    const Eigen::Vector3d n2 = np2.normalized();

    const double half_nt = 0.5 * n_t;
    const double esii_t_r = ESii * r;

    const Eigen::Vector3d t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
    const Eigen::Vector3d t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

    const Eigen::Matrix3d R1 =
        ComputeHomographyRotation(H_normalized, t1_star, n1, v);
    const Eigen::Vector3d t1 = R1 * t1_star;

    const Eigen::Matrix3d R2 =
        ComputeHomographyRotation(H_normalized, t2_star, n2, v);
    const Eigen::Vector3d t2 = R2 * t2_star;

    Rs = {R1, R1, R2, R2};
    ts = {t1, -t1, t2, -t2};
    ns = {-n1, n1, -n2, n2};
}

std::vector<Eigen::Matrix3d> Homography2Pose::GetRotationVector() const {
    return Rs;
}

std::vector<Eigen::Vector3d> Homography2Pose::GetTranslationVector() const {
    return ts;
}

std::vector<Eigen::Vector3d> Homography2Pose::GetNormalVector() const {
    return ns;
}

void Homography2Pose::PoseFromHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2,
                              const std::vector<Eigen::Vector2d>& points1,
                              const std::vector<Eigen::Vector2d>& points2)
{
    Reset();
    CHECK_EQ(points1.size(), points2.size());

    DecomposeHomographyMatrix(H, K1, K2);

    for (size_t i = 0; i < Rs.size(); ++i) {
        std::vector<Eigen::Vector3d> points3D_cmb;
        CheckCheirality(Rs[i], ts[i], points1, points2, &points3D_cmb);
        if (points3D_cmb.size() >= points3D.size()) {
            R = Rs[i];
            t = ts[i];
            n = ns[i];
            points3D = points3D_cmb;
        }
    }
}

Eigen::Matrix3d Homography2Pose::GetRotation() const {
    return R;
}

Eigen::Vector3d Homography2Pose::GetTranslation() const {
    return t;
}

Eigen::Vector3d Homography2Pose::GetNormal() const {
    return n;
}

void Homography2Pose::PoseFromHomographyMatrixAfterDecompose(const std::vector<Eigen::Vector2d>& points1,
                              const std::vector<Eigen::Vector2d>& points2)
{
    CHECK_EQ(points1.size(), points2.size());

    points3D.clear();

    for (size_t i = 0; i < Rs.size(); ++i) {
        std::vector<Eigen::Vector3d> points3D_cmb;
        CheckCheirality(Rs[i], ts[i], points1, points2, &points3D_cmb);
        if (points3D_cmb.size() >= points3D.size()) {
            R = Rs[i];
            t = ts[i];
            n = ns[i];
            points3D = points3D_cmb;
        }
    }
}

Eigen::Matrix3d HomographyMatrixFromPose(const Eigen::Matrix3d& K1,
                                         const Eigen::Matrix3d& K2,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& n,
                                         const double d)
{
    CHECK_GT(d, 0);
    return K2 * (R - t * n.normalized().transpose() / d) * K1.inverse();
}