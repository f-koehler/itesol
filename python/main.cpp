#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <itesol/algorithms/power_method.hpp>
#include <itesol/backends/eigen_dense.hpp>

namespace py = pybind11;

PYBIND11_MODULE(itesol_core, m) {
    m.doc() = "itesol core module";

    using Backend = itesol::backends::EigenDense<double>;
    py::class_<Backend>(m, "EigenDenseFloat64")
        .def(py::init<>())
        .def("create_vector", &Backend::create_vector)
        .def("create_zero_vector", &Backend::create_zero_vector)
        .def("create_random_vector", &Backend::create_random_vector)
        .def("create_matrix", &Backend::create_matrix)
        .def("create_zero_matrix", &Backend::create_zero_matrix);

    using Algorithm = itesol::algorithms::PowerMethod<Backend>;
    py::class_<Algorithm>(m, "PowerMethodFloat64")
        .def(py::init<typename Backend::Index, const Backend &,
                      typename Backend::Index, typename Backend::RealScalar>());
}