#ifndef ITESOL_BACKENDS_EIGEN_DENSE_HPP
#define ITESOL_BACKENDS_EIGEN_DENSE_HPP

#include "Eigen/Dense"

namespace itesol::backends {
    template <typename ScalarT>
    class EigenDense {
      public:
        using Scalar = ScalarT;
        using Index = int;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        Vector create_vector(Index rows) { return Vector(rows); }

        Vector create_zero_vector(Index rows) { return Vector::Zero(rows); }

        Vector create_random_vector(Index rows) { return Vector::Random(rows); }
    };
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_EIGEN_DENSE_HPP
