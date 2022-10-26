#ifndef ITESOL_BACKENDS_CONCEPT_HPP
#define ITESOL_BACKENDS_CONCEPT_HPP

#include <concepts>
#include <functional>
#include <type_traits>

namespace itesol::backends {

    template <typename T>
    concept Backend = requires(T t) {
        typename T::Scalar;
        typename T::Index;

        typename T::Vector;
        typename T::VectorRef;
        typename T::VectorCRef;

        typename T::Matrix;
        typename T::MatrixRef;
        typename T::MatrixCRef;

        typename T::LinearOperator;

        {
            t.create_vector(typename T::Index())
            } -> std::same_as<typename T::Vector>;
        {
            t.create_zero_vector(typename T::Index())
            } -> std::same_as<typename T::Vector>;
        {
            t.create_random_vector(typename T::Index())
            } -> std::same_as<typename T::Vector>;

        {
            t.dotc(typename T::VectorCRef(typename T::Vector()),
                   typename T::VectorCRef(typename T::Vector()))
            } -> std::same_as<typename T::Scalar>;
    };

    //    template <typename T>
    //    concept HasComputeResidual = Backend<T> && requires(T t) {
    //        {
    //            t.compute_residual(typename const T::Scalar&(), typename
    //            T::VectorCRef(typename T::Vector()), typename
    //            T::VectorCRef(typename T::Vector()))
    //        } -> std::same_as<typename T::Scalar>;
    //    };
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_CONCEPT_HPP
