#include "pybind/essential_matrix.h"
#include "pybind/fundamental_matrix.h"

PYBIND11_MODULE(pyVO, m) {
    pybind_essential_matrix(m);
    pybind_fundamental_matrix(m);
}