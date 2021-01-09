#include "pybind/essential_matrix.h"

#include "estimator/essential_matrix.h"

void pybind_essential_matrix(py::module &m) {
    py::module m_submodule = m.def_submodule("essential_matrix");
    py::class_<EssentialMatrixFivePointEstimator> essential_five(m_submodule, "EssentialMatrixFivePointEstimator");
    py::detail::bind_default_constructor<EssentialMatrixFivePointEstimator>(essential_five);
    py::detail::bind_copy_functions<EssentialMatrixFivePointEstimator>(essential_five);
    essential_five
        .def("Estimate", &EssentialMatrixFivePointEstimator::Estimate)
        .def("Residuals", &EssentialMatrixFivePointEstimator::Residuals);
    py::class_<EssentialMatrixEightPointEstimator> essential_eight(m_submodule, "EssentialMatrixEightPointEstimator");
    py::detail::bind_default_constructor<EssentialMatrixEightPointEstimator>(essential_eight);
    py::detail::bind_copy_functions<EssentialMatrixEightPointEstimator>(essential_eight);
    essential_eight
        .def("Estimate", &EssentialMatrixEightPointEstimator::Estimate)
        .def("Residuals", &EssentialMatrixEightPointEstimator::Residuals);
}