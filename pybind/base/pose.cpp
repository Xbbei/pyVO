#include "pybind/base/pose.h"

#include "src/base/pose.h"

std::vector<Eigen::Vector3d> PYCheckCheirality(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                     const std::vector<Eigen::Vector2d>& points1,
                     const std::vector<Eigen::Vector2d>& points2)
{
    std::vector<Eigen::Vector3d> points3D;
    CheckCheirality(R, t, points1, points2, &points3D);
    return points3D;
}

void pybind_base_pose(py::module &m) {
    py::module m_submodule = m.def_submodule("base");
    m_submodule.def("CheckCheirality", &PYCheckCheirality);
}