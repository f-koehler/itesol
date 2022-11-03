#ifndef INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_QUIET
#define INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_QUIET

#include <vector>

#include "base.hpp"

namespace itesol::observers::single_eigenvalue {
    template <typename AlgorithmT>
    class QuietObserver : public BaseObserver<AlgorithmT> {
      public:
        using Algorithm = AlgorithmT;
        using ParentClass = BaseObserver<Algorithm>;
        using Scalar = typename Algorithm::Scalar;
        using Index = typename Algorithm::Index;

      protected:
        std::vector<Scalar> m_residuals;
        std::vector<Scalar> m_eigenvalues;
        bool m_converged;
        Index m_iterations;
    };
} // namespace itesol::observers::single_eigenvalue

#endif /* INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_QUIET */
