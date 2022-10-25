#ifndef ITESOL_ALGORITHMS_POWER_METHOD_VERBOSE_OBSERVER_HPP
#define ITESOL_ALGORITHMS_POWER_METHOD_VERBOSE_OBSERVER_HPP

#include "quiet_observer.hpp"

namespace itesol {
    template <typename PowerMethodT>
    class VerbosePowerMethodObserver
        : public QuietPowerMethodObserver<PowerMethodT> {
      public:
        using PowerMethod = PowerMethodT;
        using ParentClass = QuietPowerMethodObserver<PowerMethod>;

        void start(const PowerMethod &power_method) override {
            ParentClass::start(power_method);

            spdlog::info("Starting power method with dimension {} …",
                         power_method.get_dimension());
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
                spdlog::info("Power method converged after {} iterations.",
                             power_method.get_iterations());
            } else {
                spdlog::error(
                    "Power method did not converge after {} iterations!",
                    power_method.get_iterations());
            }
        }
    };

} // namespace itesol

#endif
