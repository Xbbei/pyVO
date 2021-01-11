#include "pybind/estimator/homography_matrix.h"

#include "src/estimator/homography_matrix.h"

void pybind_homography_matrix(py::module &m) {
    py::module m_submodule = m.def_submodule("homography_matrix");
    py::class_<HomographyMatrixEstimator> homography_estimator(m_submodule, "HomographyMatrixEstimator");
    py::detail::bind_default_constructor<HomographyMatrixEstimator>(homography_estimator);
    py::detail::bind_copy_functions<HomographyMatrixEstimator>(homography_estimator);
    homography_estimator
        .def_static("Estimate", &HomographyMatrixEstimator::Estimate)
        .def_static("Residuals", &HomographyMatrixEstimator::Residuals);
}