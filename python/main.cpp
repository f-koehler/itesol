#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <itesol/algorithms/power_method.hpp>
#include <itesol/backends/eigen_dense.hpp>
#include <itesol/observers/single_eigenvalue/verbose.hpp>
#include <itesol/test_matrices/random_symmetric.hpp>

namespace py = pybind11;

template <typename Scalar>
void define_eigen_dense_backend(py::module_ &module, const char *name) {
    using Backend = itesol::backends::EigenDense<Scalar>;
    py::class_<Backend>(module, name)
        .def(py::init<>())
        .def("create_vector", &Backend::create_vector)
        .def("create_zero_vector", &Backend::create_zero_vector)
        .def("create_random_vector", &Backend::create_random_vector)
        .def("create_matrix", &Backend::create_matrix)
        .def("create_zero_matrix", &Backend::create_zero_matrix)
        .def("make_linear_operator", &Backend::make_linear_operator);
}

template <typename Backend>
void define_power_method(py::module_ &module, const char *name) {
    using Algorithm = itesol::algorithms::PowerMethod<Backend>;
    using LinearOperator = typename Backend::LinearOperator;

    py::class_<Algorithm>(module, name)
        .def(py::init<typename Backend::Index, const Backend &,
                      typename Backend::Index, typename Backend::RealScalar>())
        .def("compute", py::overload_cast<const LinearOperator &,
                                          itesol::observers::single_eigenvalue::
                                              BaseObserver<Algorithm> &>(
                            &Algorithm::compute));
}

PYBIND11_MODULE(itesol_core, m) {
    m.doc() = "itesol core module";

    define_eigen_dense_backend<float>(m, "EigenDenseBackendFloat32");
    define_eigen_dense_backend<double>(m, "EigenDenseBackendFloat64");
    define_eigen_dense_backend<std::complex<float>>(
        m, "EigenDenseBackendComplex64");
    define_eigen_dense_backend<std::complex<double>>(
        m, "EigenDenseBackendComplex128");

    define_power_method<itesol::backends::EigenDense<float>>(
        m, "PowerMethodFloat32");
    define_power_method<itesol::backends::EigenDense<double>>(
        m, "PowerMethodFloat64");
    define_power_method<itesol::backends::EigenDense<std::complex<float>>>(
        m, "PowerMethodComplex64");
    define_power_method<itesol::backends::EigenDense<std::complex<double>>>(
        m, "PowerMethodComplex128");

    m.def("initialize_random_symmetric_float_32",
          &itesol::test_matrices::initialize_random_symmetric<
              itesol::backends::EigenDense<float>>);
    m.def("initialize_random_symmetric_float_64",
          &itesol::test_matrices::initialize_random_symmetric<
              itesol::backends::EigenDense<double>>);
    m.def("initialize_random_symmetric_complex_64",
          &itesol::test_matrices::initialize_random_symmetric<
              itesol::backends::EigenDense<std::complex<float>>>);
    m.def("initialize_random_symmetric_complex_128",
          &itesol::test_matrices::initialize_random_symmetric<
              itesol::backends::EigenDense<std::complex<double>>>);

    py::class_<itesol::observers::single_eigenvalue::BaseObserver<
        itesol::algorithms::PowerMethod<itesol::backends::EigenDense<double>>>>(
        m, "BasePowerMethodObserverFloat64")
        .def(py::init<>());
    py::class_<itesol::observers::single_eigenvalue::VerboseObserver<
        itesol::algorithms::PowerMethod<itesol::backends::EigenDense<double>>>>(
        m, "VerbosePowerMethodObserverFloat64")
        .def(py::init<>());
}