#ifndef ITESOL_BACKENDS_EIGEN_DENSE_HPP
#define ITESOL_BACKENDS_EIGEN_DENSE_HPP

#include "Eigen/Dense"

#include "../concepts.hpp"
#include "concept.hpp"

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

        ///
        /// Create an unitialized vector.
        /// \param rows Number of rows in vector.
        /// \return A new uninitialized vector of specified size.
        ///
        auto create_vector(Index rows) -> Vector { return Vector(rows); }
        auto create_zero_vector(Index rows) -> Vector {
            return Vector::Zero(rows);
        }
        auto create_random_vector(Index rows) -> Vector {
            return Vector::Random(rows);
        }

        auto create_matrix(Index rows, Index cols) -> Matrix {
            return Matrix(rows, cols);
        }
        auto create_zero_matrix(Index rows, Index cols) -> Matrix {
            return Matrix::Zero(rows, cols);
        }

        auto make_linear_operator(MatrixCRef matrix) -> LinearOperator {
            return [&matrix](VectorCRef x, VectorRef y) { y = matrix * x; };
        }

        auto dotc(VectorCRef x, VectorCRef y) -> Scalar {
            return x.adjoint() * y;
        }
        auto norm(VectorCRef x) -> RealScalar { return x.norm(); }

        void a_x_plus_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            y += alpha * x;
        }

        void x_plus_a_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            y *= alpha;
            y += x;
        }

        void scale(Scalar alpha, VectorRef x) { x *= alpha; }
        void normalize(VectorRef x) { x.normalize(); }
        void copy(VectorCRef x, VectorRef y) { y = x; }
    };

    namespace details {
        template <typename Scalar>
        struct SupportsExpressionTemplates<EigenDense<Scalar>>
            : std::true_type {};
    } // namespace details
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_EIGEN_DENSE_HPP
