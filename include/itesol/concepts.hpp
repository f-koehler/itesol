#ifndef INCLUDE_ITESOL_CONCEPTS_HPP
#define INCLUDE_ITESOL_CONCEPTS_HPP

#include <complex>
#include <type_traits>

namespace itesol {

    namespace details {
        template <typename T>
        struct IsComplex : std::false_type {
            using RealType = T;
        };

        template <typename T>
        struct IsComplex<std::complex<T>> : std::true_type {
            using RealType = T;
        };
    } // namespace details

    template <typename T>
    concept IsComplex = details::IsComplex<T>::value;

    template <typename T>
    using RealType = typename details::IsComplex<T>::RealType;
} // namespace itesol

#endif // INCLUDE_ITESOL_CONCEPTS_HPP
