#ifndef ITESOL_ALGORITHMS_POWER_METHOD_HPP
#define ITESOL_ALGORITHMS_POWER_METHOD_HPP

#include "../backends/concept.hpp"
#include "../concepts.hpp"
#include "../observers/single_eigenvalue/base.hpp"

namespace itesol::algorithms {

    template <backends::IsBackend BackendT>
    class PowerMethod {
      public:
        using Backend = BackendT;
        using Index = typename Backend::Index;
        using Scalar = typename Backend::Scalar;
        using RealScalar = RealType<Scalar>;
        using Vector = typename Backend::Vector;
        using VectorCRef = typename Backend::VectorCRef;
        using Matrix = typename Backend::Matrix;
        using LinearOperator = typename Backend::LinearOperator;
        using Observer =
            observers::single_eigenvalue::BaseObserver<PowerMethod>;

      private:
        Index m_dimension;
        Backend m_backend;
        Index m_max_iterations;
        RealScalar m_tolerance;

        bool m_converged{false};
        Scalar m_rayleigh_quotient;
        RealScalar m_residual;
        Vector m_eigenvector;
        Vector m_new_eigenvector;
        Index m_iterations;

      protected:
        virtual void compute_impl(const LinearOperator &op,
                                  Observer &observer) {
            observer.start(*this);

            m_converged = false;
            m_iterations = 0;

            while (true) {
                m_backend.normalize(m_eigenvector);

                op(m_eigenvector, m_new_eigenvector);

                m_rayleigh_quotient =
                    m_backend.dotc(m_eigenvector, m_new_eigenvector);

                if constexpr (backends::SupportsExpressionTemplates<Backend>) {
                    m_residual =
                        m_backend.norm(m_new_eigenvector -
                                       m_rayleigh_quotient * m_eigenvector);
                } else {
                    m_backend.x_plus_a_y(-m_rayleigh_quotient,
                                         m_new_eigenvector, m_eigenvector);
                    m_residual = m_backend.norm(m_eigenvector);
                }

                m_backend.copy(m_new_eigenvector, m_eigenvector);

                ++m_iterations;

                observer.observe(*this);

                if (m_residual < m_tolerance) {
                    m_converged = true;
                    break;
                }

                if ((m_max_iterations > Index(0)) &&
                    (m_iterations >= m_max_iterations)) {
                    break;
                }
            }

            observer.finish(*this);
        }

      public:
        explicit PowerMethod(Index dimension, const Backend &backend,
                             Index max_iterations = 1000,
                             RealScalar tolerance = 1e-10)
            : m_dimension(dimension),
              m_backend(backend),
              m_max_iterations(max_iterations),
              m_tolerance(tolerance),

              m_rayleigh_quotient(0),
              m_residual(0),
              m_eigenvector(m_backend.create_random_vector(dimension)),
              m_new_eigenvector(m_backend.create_vector(dimension)),
              m_iterations(0) {}

        void compute(const LinearOperator &op) {
            auto observer = Observer{};
            compute_impl(op, observer);
        }

        void compute(const LinearOperator &op, Observer &observer) {
            compute_impl(op, observer);
        }

        auto get_dimension() const -> Index { return m_dimension; }
        auto get_backend() const -> const Backend & { return m_backend; }
        auto get_max_iterations() const -> Index { return m_max_iterations; }
        void set_max_iterations(const Index &max_iterations) {
            m_max_iterations = max_iterations;
        }
        auto get_tolerance() const -> Scalar { return m_tolerance; }
        void set_tolerance(const Scalar &tolerance) { m_tolerance = tolerance; }

        [[nodiscard]] auto is_converged() const -> bool { return m_converged; }

        auto get_eigenvalue() const -> const RealScalar {
            if constexpr (IsComplex<Scalar>) {
                return m_rayleigh_quotient.real();
            } else {
                return m_rayleigh_quotient;
            }
        }

        auto get_eigenvector() const -> VectorCRef { return m_new_eigenvector; }

        auto get_residual() const -> const Scalar & { return m_residual; }

        auto get_iterations() const -> Index { return m_iterations; }

        void set_initial_vector(VectorCRef initial_vector) {
            m_backend.copy(initial_vector, m_eigenvector);
        }
    };
} // namespace itesol::algorithms

#endif
