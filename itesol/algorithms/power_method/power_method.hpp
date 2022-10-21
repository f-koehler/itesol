#ifndef ITESOL_ALGORITHMS_POWER_METHOD_POWER_METHOD_HPP
#define ITESOL_ALGORITHMS_POWER_METHOD_POWER_METHOD_HPP

#include <functional>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "spdlog/spdlog.h"

#include "../../eigen.hpp"

namespace itesol {
    template <typename PowerMethod>
    class PowerMethodObserver;

    template <typename AllocatorT>
    class PowerMethod {
      public:
        using Allocator = AllocatorT;
        using Index = typename Allocator::Index;
        using Scalar = typename Allocator::Scalar;
        using Vector = typename Allocator::Vector;
        using Matrix = typename Allocator::Matrix;
        using LinearOperator = std::function<void(CRef<Vector>, Ref<Vector>)>;
        using Observer = PowerMethodObserver<PowerMethod>;

      private:
        Index m_dimension;
        Allocator m_allocator;
        Index m_max_iterations;
        Scalar m_tolerance;

        bool m_converged;
        Scalar m_rayleigh_quotient;
        Scalar m_residual;
        Vector m_eigenvector;
        Vector m_new_eigenvector;
        Index m_iterations;

      protected:
        virtual void compute_impl(LinearOperator &op, Observer &observer) {
            observer.start(*this);

            m_converged = false;
            m_iterations = 0;

            m_eigenvector.normalize();

            for (Index i = 0; i < m_max_iterations; ++i) {
                op(m_eigenvector, m_new_eigenvector);
                m_rayleigh_quotient =
                    m_eigenvector.adjoint() * m_new_eigenvector;

                m_residual =
                    (m_new_eigenvector - m_rayleigh_quotient * m_eigenvector)
                        .norm();
                m_eigenvector = m_new_eigenvector;
                m_eigenvector.normalize();

                ++m_iterations;

                observer.observe(*this);

                if (m_residual < m_tolerance) {
                    m_converged = true;
                    break;
                }
            }

            observer.finish(*this);
        }

      public:
        explicit PowerMethod(Index dimension, Allocator &&allocator,
                             Index max_iterations = 1000,
                             Scalar tolerance = 1e-10)
            : m_dimension(dimension),
              m_allocator(std::move(allocator)),
              m_max_iterations(max_iterations),
              m_tolerance(tolerance),
              m_converged(false),
              m_rayleigh_quotient(0),
              m_residual(0),
              m_eigenvector(m_allocator.create_random_vector(dimension)),
              m_new_eigenvector(m_allocator.create_vector(dimension)),
              m_iterations(0) {}

        void compute(LinearOperator &op) {
            auto observer = Observer{};
            compute_impl(op, observer);
        }

        void compute(LinearOperator &op, Observer &observer) {
            compute_impl(op, observer);
        }

        void compute(CRef<Matrix> &A) {
            LinearOperator op = [&A](CRef<Vector> x, Ref<Vector> y) {
                y = A * x;
            };
            compute(op);
        }

        void compute(CRef<Matrix> &A, Observer &observer) {
            LinearOperator op = [&A](CRef<Vector> x, Ref<Vector> y) {
                y = A * x;
            };
            compute(op, observer);
        }

        Index get_dimension() const { return m_dimension; }

        [[nodiscard]] bool is_converged() const { return m_converged; }

        const Scalar &get_eigenvalue() const { return m_rayleigh_quotient; }

        CRef<Vector> get_eigenvector() const { return m_eigenvector; }

        const Scalar &get_residual() const { return m_residual; }

        Index get_iterations() const { return m_iterations; }
    };
} // namespace itesol

#endif // ITESOL_ALGORITHMS_POWER_METHOD_POWER_METHOD_HPP
