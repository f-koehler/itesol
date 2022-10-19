#ifndef ITESOL_POWER_METHOD_HPP
#define ITESOL_POWER_METHOD_HPP

#include <utility>
#include <functional>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

namespace itesol {
    template<typename ScalarT>
    class EigenDenseAllocator {
    public:
        using Scalar = ScalarT;
        using Index = int;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using MatrixRef = Eigen::Ref<Matrix>;
        using MatrixCRef = const Eigen::Ref<const Matrix>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorRef = Eigen::Ref<Vector>;
        using VectorCRef = const Eigen::Ref<const Vector>;

        Vector create_vector(Index rows) {
            return Vector(rows);
        }

        Vector create_zero_vector(Index rows) {
            return Vector::Zero(rows);
        }

        Vector create_random_vector(Index rows) {
            return Vector::Random(rows);
        }
    };

    template<typename PowerMethod>
    class DummyPowerMethodObserver;

    template<typename AllocatorT>
    class PowerMethod {
    public:
        using Allocator = AllocatorT;
        using Index = typename Allocator::Index;
        using Scalar = typename Allocator::Scalar;
        using Vector = typename Allocator::Vector;
        using VectorRef = typename Allocator::VectorRef;
        using VectorCRef = typename Allocator::VectorCRef;
        using LinearOperator = std::function<void(VectorCRef, VectorRef)>;

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

    public:
        explicit PowerMethod(Index dimension, Allocator &&allocator, Index max_iterations = 1000,
                             Scalar tolerance = 1e-10)
                : m_dimension(dimension), m_max_iterations(max_iterations), m_tolerance(tolerance),
                  m_allocator(std::move(allocator)),
                  m_eigenvector(m_allocator.create_random_vector(dimension)),
                  m_new_eigenvector(m_allocator.create_vector(dimension)) {}

        template<typename Observer = DummyPowerMethodObserver<PowerMethod>>
        void compute(LinearOperator &op) {
            m_converged = false;
            m_iterations = 0;

            m_eigenvector.normalize();

            for (Index i = 0; i < m_max_iterations; ++i) {
                op(m_eigenvector, m_new_eigenvector);
                m_rayleigh_quotient = m_eigenvector.adjoint() * m_new_eigenvector;

                m_residual = (m_new_eigenvector - m_rayleigh_quotient * m_eigenvector).norm();
                std::cout << m_residual << '\n';
                m_eigenvector = m_new_eigenvector;
                m_eigenvector.normalize();

                ++m_iterations;

                if (m_residual < m_tolerance) {
                    m_converged = true;
                    break;
                }
            }
        }

        template<typename Observer = DummyPowerMethodObserver<PowerMethod>>
        void compute(typename Allocator::MatrixCRef &A) {
            LinearOperator op = [&A](VectorCRef x, VectorRef y) {
                y = A * x;
            };
            compute(op);
        }

        Index get_dimension() const { return m_dimension; }

        bool is_converged() const {
            return m_converged;
        }

        const Scalar &get_eigenvalue() const {
            return m_rayleigh_quotient;
        }

        const Scalar &get_residual() const {
            return m_residual;
        }

        Index get_iterations() const {
            return m_iterations;
        }
    };

    template<typename PowerMethodT>
    class PowerMethodObserver {
    public:
        using PowerMethod = PowerMethodT;

        virtual ~PowerMethodObserver() = default;

        virtual void reset() {}

        virtual void start([[maybe_unused]] const PowerMethod &power_method) {}

        virtual void observe([[maybe_unused]] const PowerMethod &power_method) {}

        virtual void finish([[maybe_unused]] const PowerMethod &power_method) {}
    };

    template<typename PowerMethodT>
    class QuietPowerMethodObserver : public PowerMethodObserver<PowerMethodT> {
    public:
        using PowerMethod = PowerMethodT;
        using ParentClass = PowerMethodObserver<PowerMethod>;
        using Scalar = typename PowerMethod::Scalar;
        using Index = typename PowerMethod::Index;

    protected:
        std::vector<Scalar> m_residuals;
        std::vector<Scalar> m_eigenvalues;
        bool m_converged;
        Index m_iterations;
    };

    template<typename PowerMethodT>
    class DebugPowerMethodObserver : public QuietPowerMethodObserver<PowerMethodT> {
    public:
        using PowerMethod = PowerMethodT;
        using ParentClass = QuietPowerMethodObserver<PowerMethod>;

        void start(const PowerMethod &power_method) override {
            ParentClass::start(power_method);

            spdlog::info("Starting power method with dimension {} …", power_method.get_dimension());
        }

        void observe(const PowerMethod &power_method) override {
            ParentClass::observe(power_method);

            spdlog::info("Iteration: {}", power_method.get_iterations());
            spdlog::info("\tEigenvalue: {}", power_method.get_eigenvalue());
            spdlog::info("\tResidual: {}", power_method.get_residual());
        }

        void finish(const PowerMethod &power_method) override {
            ParentClass::finish(power_method);
            if (power_method.is_converged()) {
                spdlog::info("Power method converged after {} iterations.", power_method.get_iterations());
            } else {
                spdlog::error("Power method did not converge after {} iterations!", power_method.get_iterations());
            }
            spdlog::info("\tEigenvalue: {}", power_method.get_eigenvalue());
        }
    };
}

#endif //ITESOL_POWER_METHOD_HPP
