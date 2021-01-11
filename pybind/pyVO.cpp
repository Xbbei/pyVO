#include "pybind/estimator/essential_matrix.h"
#include "pybind/estimator/fundamental_matrix.h"
#include "pybind/estimator/homography_matrix.h"

PYBIND11_MODULE(pyVO, m) {
    pybind_essential_matrix(m);
    pybind_fundamental_matrix(m);
    pybind_homography_matrix(m);
}