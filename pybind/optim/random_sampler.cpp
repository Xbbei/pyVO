#include "pybind/optim/random_sampler.h"

#include "src/optim/random_sampler.h"

void pybind_optim_random_sampler(py::module &m) {
    py::module m_submodule = m.def_submodule("optim");
    py::class_<RandomSampler> rand_sampler(m_submodule, "RandomSampler");
    rand_sampler
        .def(py::init<const size_t>())
        .def("Initialize", &RandomSampler::Initialize)
        .def("MaxNumSamples", &RandomSampler::MaxNumSamples)
        .def("Sample", &RandomSampler::Sample);
}