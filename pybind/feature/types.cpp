#include "pybind/feature/types.h"

#include "src/feature/types.h"

void pybind_feature_types(py::module &m) {
    py::module m_submodule = m.def_submodule("feature");
    py::class_<FeatureKeypoint> featurekp(m_submodule, "FeatureKeypoint");
    featurekp
        .def(py::init<>())
        .def(py::init<const float, const float>())
        .def(py::init<const float, const float, const float, const float>())
        .def(py::init<const float, const float, const float, const float, const float, const float>())
        .def_static("FromParameters", &FeatureKeypoint::FromParameters)
        .def("Rescale", py::overload_cast<const float>(&FeatureKeypoint::Rescale))
        .def("Rescale", py::overload_cast<const float, const float>(&FeatureKeypoint::Rescale))
        .def("ComputeScale", &FeatureKeypoint::ComputeScale)
        .def("ComputeScaleX", &FeatureKeypoint::ComputeScaleX)
        .def("ComputeScaleY", &FeatureKeypoint::ComputeScaleY)
        .def("ComputeOrientation", &FeatureKeypoint::ComputeOrientation)
        .def("ComputeShear", &FeatureKeypoint::ComputeShear)
        .def_readwrite("x", &FeatureKeypoint::x)
        .def_readwrite("y", &FeatureKeypoint::y)
        .def_readwrite("a11", &FeatureKeypoint::a11)
        .def_readwrite("a12", &FeatureKeypoint::a12)
        .def_readwrite("a21", &FeatureKeypoint::a21)
        .def_readwrite("a22", &FeatureKeypoint::a22);
}