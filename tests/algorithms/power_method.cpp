#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <complex>
#include <random>

#include <itesol/algorithms/power_method.hpp>
#include <itesol/backends/blas.hpp>
#include <itesol/backends/eigen_dense.hpp>
#include <itesol/test_matrices/random_symmetric.hpp>

TEMPLATE_TEST_CASE("PowerMethod EigenDense backend", "[algorithms][eigen]",
                   double, std::complex<double>, long double,
                   std::complex<long double>) {
    using Backend = itesol::backends::EigenDense<TestType>;
    using Algorithm = itesol::algorithms::PowerMethod<Backend>;

    const int dimension = 64;
    const typename Backend::RealScalar tolerance = 1e-6;

    Backend backend{};

    auto matrix = backend.create_matrix(dimension, dimension);
    itesol::test_matrices::initialize_random_symmetric<Backend>(matrix);

    Algorithm algorithm(dimension, backend, 10000, tolerance);
    algorithm.compute(backend.make_linear_operator(matrix));

    REQUIRE(algorithm.is_converged());

    Eigen::SelfAdjointEigenSolver<typename Backend::Matrix> solver;
    solver.compute(matrix);

    REQUIRE(Catch::Approx(algorithm.get_eigenvalue()) ==
            solver.eigenvalues().cwiseAbs().maxCoeff());
}

#ifdef ITESOL_TEST_USE_BLAS

TEMPLATE_TEST_CASE("PowerMethod Blas backend", "[algorithms][blas]", double) {
    using Backend = itesol::backends::Blas<TestType>;
    using Algorithm = itesol::algorithms::PowerMethod<Backend>;

    const int dimension = 64;
    const typename Backend::RealScalar tolerance = 1e-6;

    Backend backend{};

    auto matrix = backend.create_matrix(dimension, dimension);
    itesol::test_matrices::initialize_random_symmetric<Backend>(matrix);

    Algorithm algorithm(dimension, backend, 10000, tolerance);
    algorithm.compute(backend.make_linear_operator(matrix));

    REQUIRE(algorithm.is_converged());

    Eigen::Matrix<TestType, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(
        dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            eigen_matrix(i, j) = matrix(i, j);
        }
    }
    Eigen::SelfAdjointEigenSolver<decltype(eigen_matrix)> solver;
    solver.compute(eigen_matrix);

    REQUIRE(Catch::Approx(algorithm.get_eigenvalue()) ==
            solver.eigenvalues().cwiseAbs().maxCoeff());
}

#endif
