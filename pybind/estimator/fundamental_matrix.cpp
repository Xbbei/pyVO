#include "pybind/estimator/fundamental_matrix.h"

#include "src/estimator/fundamental_matrix.h"

void pybind_estimator_fundamental_matrix(py::module &m) {
    py::module m_submodule = m.def_submodule("estimator");
    py::class_<FundamentalMatrixSevenPointEstimator> fundalmental_seven(m_submodule, "FundamentalMatrixSevenPointEstimator");
    py::detail::bind_default_constructor<FundamentalMatrixSevenPointEstimator>(fundalmental_seven);
    py::detail::bind_copy_functions<FundamentalMatrixSevenPointEstimator>(fundalmental_seven);
    fundalmental_seven
        .def_static("Estimate", &FundamentalMatrixSevenPointEstimator::Estimate)
        .def_static("Residuals", &FundamentalMatrixSevenPointEstimator::Residuals)
        .def_readonly("kMinNumSamples", &FundamentalMatrixSevenPointEstimator::kMinNumSamples);
    py::class_<FundamentalMatrixEightPointEstimator> fundamental_eight(m_submodule, "FundamentalMatrixEightPointEstimator");
    py::detail::bind_default_constructor<FundamentalMatrixEightPointEstimator>(fundamental_eight);
    py::detail::bind_copy_functions<FundamentalMatrixEightPointEstimator>(fundamental_eight);
    fundamental_eight
        .def_static("Estimate", &FundamentalMatrixEightPointEstimator::Estimate)
        .def_static("Residuals", &FundamentalMatrixEightPointEstimator::Residuals)
        .def_readonly("kMinNumSamples", &FundamentalMatrixEightPointEstimator::kMinNumSamples);
}