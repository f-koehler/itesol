#ifndef INCLUDE_ITESOL_TYPE_TRAITS_HPP
#define INCLUDE_ITESOL_TYPE_TRAITS_HPP

#include <complex>
#include <type_traits>

namespace itesol {
    template <typename T>
    struct IsComplex : std::false_type {
        using RealType = T;
    };

    template <typename T>
    struct IsComplex<std::complex<T>> : std::true_type {
        using RealType = T;
    };

    template <typename T>
    using RealType = typename IsComplex<T>::RealType;
} // namespace itesol

#endif // INCLUDE_ITESOL_TYPE_TRAITS_HPP
