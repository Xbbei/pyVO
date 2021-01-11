#include "pybind/essential_matrix.h"

PYBIND11_MODULE(pyVO, m) {
    pybind_essential_matrix(m);
}