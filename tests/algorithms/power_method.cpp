#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <complex>
#include <random>

#include <itesol/algorithms/power_method.hpp>
#include <itesol/backends/eigen_dense.hpp>

TEMPLATE_TEST_CASE("PowerMethod using EigenDenseBackend", "[algorithms][eigen]",
                   float, double) {
    using Backend = itesol::backends::EigenDense<TestType>;
    using Algorithm = itesol::algorithms::PowerMethod<Backend>;

    const int dimension = 128;

    Backend backend{};
    std::uniform_real_distribution<TestType> dist(0.0, 1.0);
    std::mt19937_64 prng(0);

    auto matrix = backend.create_matrix(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = i + 1; j < dimension; ++j) {
            matrix(i, j) = dist(prng);
            matrix(j, i) = matrix(i, j);
        }
        matrix(i, i) = dist(prng);
    }

    Algorithm algorithm(dimension, backend);
    algorithm.compute(backend.make_linear_operator(matrix));

    Eigen::SelfAdjointEigenSolver<typename Backend::Matrix> solver;
    solver.compute(matrix);

    REQUIRE(Catch::Approx(algorithm.get_eigenvalue()) ==
            solver.eigenvalues().cwiseAbs().maxCoeff());
}
