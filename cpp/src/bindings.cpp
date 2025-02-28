#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "patcher.h"

namespace py = pybind11;

PYBIND11_MODULE(dbpm, m) {
    m.def("patch_sequence", &patch_sequence, "Patch the byte sequence based on entropy",
          py::arg("bytes"), py::arg("k"), py::arg("theta"), py::arg("theta_r"));
}
