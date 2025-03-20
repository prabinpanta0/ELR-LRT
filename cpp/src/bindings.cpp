#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "patcher.h"
#include "frequency_table.h"

namespace py = pybind11;

PYBIND11_MODULE(dbpm, m) {
    m.doc() = "ELR-LRT sequence patching module";
    
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
