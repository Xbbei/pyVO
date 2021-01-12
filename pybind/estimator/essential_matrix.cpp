#include "pybind/estimator/essential_matrix.h"

#include "src/estimator/essential_matrix.h"

void pybind_estimator_essential_matrix(py::module &m) {
    py::module m_submodule = m.def_submodule("estimator");
    py::class_<EssentialMatrixFivePointEstimator> essential_five(m_submodule, "EssentialMatrixFivePointEstimator");
    py::detail::bind_default_constructor<EssentialMatrixFivePointEstimator>(essential_five);
    py::detail::bind_copy_functions<EssentialMatrixFivePointEstimator>(essential_five);
    essential_five
        .def_static("Estimate", &EssentialMatrixFivePointEstimator::Estimate)
        .def_static("Residuals", &EssentialMatrixFivePointEstimator::Residuals)
        .def_readonly("kMinNumSamples", &EssentialMatrixFivePointEstimator::kMinNumSamples);
    py::class_<EssentialMatrixEightPointEstimator> essential_eight(m_submodule, "EssentialMatrixEightPointEstimator");
    py::detail::bind_default_constructor<EssentialMatrixEightPointEstimator>(essential_eight);
    py::detail::bind_copy_functions<EssentialMatrixEightPointEstimator>(essential_eight);
    essential_eight
        .def_static("Estimate", &EssentialMatrixEightPointEstimator::Estimate)
        .def_static("Residuals", &EssentialMatrixEightPointEstimator::Residuals)
        .def_readonly("kMinNumSamples", &EssentialMatrixEightPointEstimator::kMinNumSamples);
}