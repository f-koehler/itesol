#ifndef INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_BASE
#define INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_BASE

namespace itesol::observers::single_eigenvalue {
    template <typename AlgorithmT>
    class BaseObserver {
      public:
        using Algorithm = AlgorithmT;

        virtual ~BaseObserver() = default;

        virtual void reset() {}

        virtual void start([[maybe_unused]] const Algorithm &algorithm) {}

        virtual void observe([[maybe_unused]] const Algorithm &algorithm) {}

        virtual void finish([[maybe_unused]] const Algorithm &algorithm) {}
    };
} // namespace itesol::observers::single_eigenvalue

#endif /* INCLUDE_ITESOL_OBSERVERS_SINGLE_EIGENVALUE_BASE */
