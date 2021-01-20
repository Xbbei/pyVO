//base
#include "pybind/base/homography_matrix.h"
#include "pybind/base/pose.h"

// estimator
#include "pybind/estimator/essential_matrix.h"
#include "pybind/estimator/fundamental_matrix.h"
#include "pybind/estimator/homography_matrix.h"

// optim
#include "pybind/optim/random_sampler.h"
#include "pybind/optim/support_measurement.h"

// feature
#include "pybind/feature/sift.h"
#include "pybind/feature/orb.h"
#include "pybind/feature/types.h"
#include "pybind/feature/matcher.h"

PYBIND11_MODULE(pyVO, m) {
    // base
    pybind_base_homography_matrix(m);
    pybind_base_pose(m);
    // estimator
    pybind_estimator_essential_matrix(m);
    pybind_estimator_fundamental_matrix(m);
    pybind_estimator_homography_matrix(m);
    // optim
    pybind_optim_random_sampler(m);
    pybind_optim_support_measurement(m);
    // feature
    pybind_feature_sift(m);
    pybind_feature_types(m);
    pybind_feature_orb(m);
    pybind_feature_matcher(m);
}