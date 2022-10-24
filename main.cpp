#include <cstdlib>
#include <random>

#include <Eigen/Dense>

#include "itesol/algorithms/power_method.hpp"
#include "itesol/backends/concept.hpp"

using Real = double;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    const int size = 128;

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

    itesol::PowerMethod<itesol::EigenDenseAllocator<double>> power_method(
        size, itesol::EigenDenseAllocator<double>());
    auto observer =
        itesol::VerbosePowerMethodObserver<decltype(power_method)>();
    power_method.compute(itesol::make_linear_operator<Matrix>(A), observer);

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    solver.compute(A);
    spdlog::info("Correct eigenvalue: {}",
                 solver.eigenvalues().cwiseAbs().maxCoeff());

    return EXIT_SUCCESS;
}
