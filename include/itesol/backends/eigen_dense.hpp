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
        using MatrixRef = Eigen::Ref<Matrix>;
        using MatrixCRef = const Eigen::Ref<const Matrix> &;

        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorRef = Eigen::Ref<Vector>;
        using VectorCRef = const Eigen::Ref<const Vector> &;

        using LinearOperator = std::function<void(VectorCRef, VectorRef)>;

        Vector create_vector(Index rows) { return Vector(rows); }
        Vector create_zero_vector(Index rows) { return Vector::Zero(rows); }
        Vector create_random_vector(Index rows) { return Vector::Random(rows); }

        LinearOperator make_linear_operator(MatrixCRef matrix) {
            return [&matrix](VectorCRef x, VectorRef y) { y = matrix * x; };
        }
    };
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_EIGEN_DENSE_HPP
