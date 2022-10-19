#include <random>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>

#include "itesol/power_method.hpp"

using Real = double;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

int main() {
    const int size = 128;

    itesol::PowerMethod<itesol::EigenDenseAllocator<double>> power_method(size, itesol::EigenDenseAllocator<double>());
    auto observer = itesol::DebugPowerMethodObserver<decltype(power_method)>();

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

    power_method.compute(A);
    std::cout << power_method.is_converged() << '\n';
    std::cout << power_method.get_residual() << '\n';
    std::cout << power_method.get_eigenvalue() << '\n';

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    solver.compute(A);
    std::cout << solver.eigenvalues().cwiseAbs().maxCoeff() << '\n';
    return 0;
}
