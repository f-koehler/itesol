#ifndef ITESOL_ALGORITHMS_POWER_METHOD_OBSERVER_HPP
#define ITESOL_ALGORITHMS_POWER_METHOD_OBSERVER_HPP

#include "power_method.hpp"

namespace itesol {
    template <typename PowerMethodT>
    class PowerMethodObserver {
      public:
        using PowerMethod = PowerMethodT;

        virtual ~PowerMethodObserver() = default;

        virtual void reset() {}

        virtual void start([[maybe_unused]] const PowerMethod &power_method) {}

        virtual void observe([[maybe_unused]] const PowerMethod &power_method) {
        }

        virtual void finish([[maybe_unused]] const PowerMethod &power_method) {}
    };
} // namespace itesol

#endif
