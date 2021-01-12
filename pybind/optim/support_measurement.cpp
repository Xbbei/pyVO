#include "pybind/optim/support_measurement.h"

#include "src/optim/support_measurement.h"

void pybind_optim_support_measurement(py::module &m) {
    py::module m_submodule = m.def_submodule("optim");
    py::class_<Support> support(m_submodule, "Support");
    py::detail::bind_default_constructor<Support>(support);
    py::detail::bind_copy_functions<Support>(support);
    support
        .def_readwrite("num_inliers", &Support::num_inliers)
        .def_readwrite("residual_sum", &Support::residual_sum);
    
    py::class_<InlierSupportMeasurer> inlier(m_submodule, "InlierSupportMeasurer");
    py::detail::bind_default_constructor<InlierSupportMeasurer>(inlier);
    py::detail::bind_copy_functions<InlierSupportMeasurer>(inlier);
    inlier
        .def("Evaluate", &InlierSupportMeasurer::Evaluate)
        .def("Compare", &InlierSupportMeasurer::Compare);
    
    py::class_<MEstimatorSupportMeasurer> mestimator(m_submodule, "MEstimatorSupportMeasurer");
    py::detail::bind_default_constructor<MEstimatorSupportMeasurer>(mestimator);
    py::detail::bind_copy_functions<MEstimatorSupportMeasurer>(mestimator);
    mestimator
        .def("Evaluate", &MEstimatorSupportMeasurer::Evaluate)
        .def("Compare", &MEstimatorSupportMeasurer::Compare);
}