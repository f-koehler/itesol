#ifndef ITESOL_ALGORITHMS_POWER_METHOD_QUIET_OBSERVER_HPP
#define ITESOL_ALGORITHMS_POWER_METHOD_QUIET_OBSERVER_HPP

#include "observer.hpp"

namespace itesol {
    template <typename PowerMethodT>
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
} // namespace itesol

#endif
