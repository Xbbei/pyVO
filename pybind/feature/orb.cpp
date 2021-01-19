#include "pybind/feature/orb.h"

#include "src/feature/orb.h"

void pybind_feature_orb(py::module &m) {
    NDArrayConverter::init_numpy();

    py::module m_submodule = m.def_submodule("feature");
    py::class_<ORBExtractionOptions> orb_extract_options(m_submodule, "ORBExtractionOptions");
    py::detail::bind_default_constructor<ORBExtractionOptions>(orb_extract_options);
    py::detail::bind_copy_functions<ORBExtractionOptions>(orb_extract_options);
    orb_extract_options
        .def_readwrite("nFeatures", &ORBExtractionOptions::nFeatures)
        .def_readwrite("scaleFactor", &ORBExtractionOptions::scaleFactor)
        .def_readwrite("nLevels", &ORBExtractionOptions::nLevels)
        .def_readwrite("iniThFAST", &ORBExtractionOptions::iniThFAST)
        .def_readwrite("minThFAST", &ORBExtractionOptions::minThFAST);

    py::class_<ORBExtract> orb_extract(m_submodule, "ORBExtract");
    py::detail::bind_default_constructor<ORBExtract>(orb_extract);
    py::detail::bind_copy_functions<ORBExtract>(orb_extract);
    orb_extract
        .def(py::init<const ORBExtractionOptions&>())
        .def("Reset", &ORBExtract::Reset)
        .def("ExtractORBFeatures", &ORBExtract::ExtractORBFeatures)
        .def("getKeyPoints", &ORBExtract::getKeyPoints)
        .def("getDescriptors", &ORBExtract::getDescriptors);
}