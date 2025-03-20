#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "patcher.h"
#include "frequency_table.h"

namespace py = pybind11;

PYBIND11_MODULE(dbpm, m) {
    m.doc() = "ELR-LRT sequence patching module";
    
    // First expose the free function since that's what we actually implemented
    m.def("patch_sequence", &patch_sequence,
          "Divide a sequence into patches based on entropy thresholds",
          py::arg("bytes"), py::arg("k"), py::arg("theta"), py::arg("theta_r"));
    
    // Keep these to avoid breaking the binary compatibility, but they're not used
    py::class_<Patcher>(m, "Patcher")
        .def(py::init<>())
        .def("patch_sequence", &Patcher::patch_sequence)
        .def("get_patched_sequence", &Patcher::get_patched_sequence)
        .def("reset", &Patcher::reset);

    py::class_<FrequencyTable>(m, "FrequencyTable")
        .def(py::init<>())
        .def("add_sequence", &FrequencyTable::add_sequence)
        .def("get_frequency", &FrequencyTable::get_frequency)
        .def("clear", &FrequencyTable::clear);
}
