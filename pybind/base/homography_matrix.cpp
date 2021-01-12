#include "pybind/base/homography_matrix.h"

#include "src/base/homography_matrix.h"

void pybind_base_homography_matrix(py::module &m) {
    py::module m_submodule = m.def_submodule("base");
    py::class_<Homography2Pose> homo2pose(m_submodule, "Homography2Pose");
    py::detail::bind_default_constructor<Homography2Pose>(homo2pose);
    py::detail::bind_copy_functions<Homography2Pose>(homo2pose);
    homo2pose
        .def("DecomposeHomographyMatrix", &Homography2Pose::DecomposeHomographyMatrix)
        .def("GetRotationVector", &Homography2Pose::GetRotationVector)
        .def("GetTranslationVector", &Homography2Pose::GetTranslationVector)
        .def("GetNormalVector", &Homography2Pose::GetNormalVector)
        .def("PoseFromHomographyMatrix", &Homography2Pose::PoseFromHomographyMatrix)
        .def("PoseFromHomographyMatrixAfterDecompose", &Homography2Pose::PoseFromHomographyMatrixAfterDecompose)
        .def("GetRotation", &Homography2Pose::GetRotation)
        .def("GetTranslation", &Homography2Pose::GetTranslation)
        .def("GetNormal", &Homography2Pose::GetNormal);

    m_submodule.def("HomographyMatrixFromPose", &HomographyMatrixFromPose);
}