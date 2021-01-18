#include "pybind/feature/sift.h"

#include "src/feature/sift.h"

void pybind_feature_sift(py::module &m) {
    NDArrayConverter::init_numpy();

    py::module m_submodule = m.def_submodule("feature");
    py::class_<SiftExtractionOptions> sift_extraction_options(m_submodule, "SiftExtractionOptions");
    py::detail::bind_default_constructor<SiftExtractionOptions>(sift_extraction_options);
    py::detail::bind_copy_functions<SiftExtractionOptions>(sift_extraction_options);
    sift_extraction_options
        .def_readwrite("num_threads", &SiftExtractionOptions::num_threads)
        .def_readwrite("use_gpu", &SiftExtractionOptions::use_gpu)
        .def_readwrite("gpu_index", &SiftExtractionOptions::gpu_index)
        .def_readwrite("max_image_size", &SiftExtractionOptions::max_image_size)
        .def_readwrite("max_num_features", &SiftExtractionOptions::max_num_features)
        .def_readwrite("first_octave", &SiftExtractionOptions::first_octave)
        .def_readwrite("num_octaves", &SiftExtractionOptions::num_octaves)
        .def_readwrite("octave_resolution", &SiftExtractionOptions::octave_resolution)
        .def_readwrite("peak_threshold", &SiftExtractionOptions::peak_threshold)
        .def_readwrite("edge_threshold", &SiftExtractionOptions::edge_threshold)
        .def_readwrite("estimate_affine_shape", &SiftExtractionOptions::estimate_affine_shape)
        .def_readwrite("max_num_orientations", &SiftExtractionOptions::max_num_orientations)
        .def_readwrite("upright", &SiftExtractionOptions::upright)
        .def_readwrite("darkness_adaptivity", &SiftExtractionOptions::darkness_adaptivity)
        .def_readwrite("domain_size_pooling", &SiftExtractionOptions::domain_size_pooling)
        .def_readwrite("dsp_min_scale", &SiftExtractionOptions::dsp_min_scale)
        .def_readwrite("dsp_max_scale", &SiftExtractionOptions::dsp_max_scale)
        .def_readwrite("dsp_num_scales", &SiftExtractionOptions::dsp_num_scales)
        .def_readwrite("normalization", &SiftExtractionOptions::normalization);
    py::enum_<SiftExtractionOptions::Normalization>(m_submodule, "Normalization")
        .value("L1_ROOT", SiftExtractionOptions::Normalization::L1_ROOT)
        .value("L2", SiftExtractionOptions::Normalization::L2)
        .export_values();

    py::class_<SiftExtract> sift_extract(m_submodule, "SiftExtract");
    py::detail::bind_default_constructor<SiftExtract>(sift_extract);
    py::detail::bind_copy_functions<SiftExtract>(sift_extract);
    sift_extract
        .def(py::init<const SiftExtractionOptions&>())
        .def("Reset", &SiftExtract::Reset)
        .def("getKeyPoints", &SiftExtract::getKeyPoints)
        .def("getDescriptors", &SiftExtract::getDescriptors)
        .def("ExtractSiftFeatures", &SiftExtract::ExtractSiftFeatures);
}