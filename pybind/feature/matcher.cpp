#include "pybind/feature/matcher.h"

#include "src/feature/matcher.h"

void pybind_feature_matcher(py::module &m) {
    py::module m_submodule = m.def_submodule("feature");

    m_submodule.def("ORBDescriptorDistance", &ORBDescriptorDistance)
               .def("SiftDescriptorDistance", &SiftDescriptorDistance)
               .def("BFComputeFeatureMatches", &BFComputeFeatureMatches)
               .def("FLANNComputeFeatureMatches", &FLANNComputeFeatureMatches);
}