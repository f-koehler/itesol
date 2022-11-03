#ifndef INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_VERBOSE
#define INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_VERBOSE

#include <spdlog/spdlog.h>

#include "quiet.hpp"

namespace itesol::observers::single_eigenvalue {
    template <typename AlgorithmT>
    class VerboseObserver : public QuietObserver<AlgorithmT> {
      public:
        using Algorithm = AlgorithmT;
        using ParentClass = QuietObserver<Algorithm>;

        void start(const Algorithm &algorithm) override {
            ParentClass::start(algorithm);

            spdlog::info("Starting algorithm  …");
        }

        void observe(const Algorithm &algorithm) override {
            ParentClass::observe(algorithm);

            spdlog::info("Iteration: {}", algorithm.get_iterations());
            spdlog::info("\tEigenvalue: {}", algorithm.get_eigenvalue());
            spdlog::info("\tResidual: {}", algorithm.get_residual());
        }

        void finish(const Algorithm &algorithm) override {
            ParentClass::finish(algorithm);

            if (algorithm.is_converged()) {
                spdlog::info("Algorithm converged after {} iterations.",
                             algorithm.get_iterations());
            } else {
                spdlog::error("Algorithm did not converge after {} iterations!",
                              algorithm.get_iterations());
            }
        }
    };

} // namespace itesol::observers::single_eigenvalue

#endif /* INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_VERBOSE */
