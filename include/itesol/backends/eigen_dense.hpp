#ifndef ITESOL_BACKENDS_EIGEN_DENSE_HPP
#define ITESOL_BACKENDS_EIGEN_DENSE_HPP

#include "Eigen/Dense"

#include "../type_traits.hpp"

namespace itesol::backends {
    template <typename ScalarT>
    class EigenDense {
      public:
        using Scalar = ScalarT;
        using RealScalar = RealType<Scalar>;
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

        Scalar dotc(VectorCRef x, VectorCRef y) { return x.adjoint() * y; }
        RealScalar norm(VectorCRef x) { return x.norm(); }

        void a_x_plus_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            y += alpha * x;
        }

        void x_plus_a_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            y *= alpha;
            y += x;
        }

        void scale(Scalar alpha, VectorRef x) { x *= alpha; }
        void normalize(VectorRef x) { x.normalize(); }
    };
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_EIGEN_DENSE_HPP
