#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <itesol/backends/eigen_dense.hpp>
#include <itesol/test_matrices/random_symmetric.hpp>

TEMPLATE_TEST_CASE("Random symmetric matrix EigenDense backend",
                   "[test_matrices][eigen]", float, double, std::complex<float>,
                   std::complex<double>) {
    using Backend = itesol::backends::EigenDense<TestType>;
    Backend backend;

    auto matrix = backend.create_matrix(128, 128);
    itesol::test_matrices::initialize_random_symmetric<Backend>(matrix);

    for (typename Backend::Index i = 0; i < 128; ++i) {
        for (typename Backend::Index j = 0; j <= i; ++j) {
            if constexpr (itesol::IsComplex<TestType>) {
                const auto conj = std::conj(matrix(i, j));
                REQUIRE(matrix(j, i).real() == Catch::Approx(conj.real()));
                REQUIRE(matrix(j, i).imag() == Catch::Approx(conj.imag()));
                REQUIRE(std::abs(conj.real()) <= 1.0);
                REQUIRE(std::abs(conj.imag()) <= 1.0);
            } else {
                REQUIRE(matrix(i, j) == Catch::Approx(matrix(j, i)));
                REQUIRE(std::abs(matrix(i, j)) <= 1.0);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Random symmetric matrix blas backend",
                   "[test_matrices][blas]", float, double, std::complex<float>,
                   std::complex<double>) {
    using Backend = itesol::backends::EigenDense<TestType>;
    Backend backend;

    auto matrix = backend.create_matrix(128, 128);
    itesol::test_matrices::initialize_random_symmetric<Backend>(matrix);

    for (typename Backend::Index i = 0; i < 128; ++i) {
        for (typename Backend::Index j = 0; j <= i; ++j) {
            if constexpr (itesol::IsComplex<TestType>) {
                const auto conj = std::conj(matrix(i, j));
                REQUIRE(matrix(j, i).real() == Catch::Approx(conj.real()));
                REQUIRE(matrix(j, i).imag() == Catch::Approx(conj.imag()));
            } else {
                REQUIRE(matrix(i, j) == Catch::Approx(matrix(j, i)));
            }
        }
    }
}
