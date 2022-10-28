#include <cstdlib>
#include <random>

#include <Eigen/Dense>

#include <itesol/algorithms/power_method.hpp>
#include <itesol/backends/eigen_dense.hpp>

using Real = double;
using Backend = itesol::backends::EigenDense<Real>;
using Matrix = typename Backend::Matrix;
using Vector = typename Backend::Vector;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    const int size = 128;

    Backend backend;

    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    std::mt19937_64 prng(0);

    Matrix A = Matrix::Zero(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            A(i, j) = dist(prng);
            A(j, i) = A(i, j);
        }
        A(i, i) = dist(prng);
    }

    itesol::algorithms::PowerMethod<Backend> power_method(size, backend);
    auto observer = itesol::algorithms::VerbosePowerMethodObserver<
        decltype(power_method)>();
    power_method.compute(backend.make_linear_operator(A), observer);

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    solver.compute(A);
    spdlog::info("Correct eigenvalue: {}",
                 solver.eigenvalues().cwiseAbs().maxCoeff());

    return EXIT_SUCCESS;
}
