#ifndef ITESOL_BACKENDS_CONCEPT_HPP
#define ITESOL_BACKENDS_CONCEPT_HPP

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace itesol::backends {

    template <typename T>
    concept IsBackend = requires(T t) {
        typename T::Scalar;
        typename T::RealScalar;
        typename T::Index;

        typename T::Vector;
        typename T::VectorRef;
        typename T::VectorCRef;

        typename T::Matrix;
        typename T::MatrixRef;
        typename T::MatrixCRef;

        typename T::LinearOperator;

        {
            t.create_vector(std::declval<typename T::Index>())
            } -> std::same_as<typename T::Vector>;
        {
            t.create_zero_vector(std::declval<typename T::Index>())
            } -> std::same_as<typename T::Vector>;
        {
            t.create_random_vector(std::declval<typename T::Index>())
            } -> std::same_as<typename T::Vector>;

        {
            t.dotc(std::declval<typename T::VectorCRef>(),
                   std::declval<typename T::VectorCRef>())
            } -> std::same_as<typename T::Scalar>;

        {
            t.norm(std::declval<typename T::VectorCRef>())
            } -> std::same_as<typename T::RealScalar>;

        {
            t.scale(std::declval<typename T::Scalar>(),
                    std::declval<typename T::VectorRef>())
            } -> std::same_as<void>;

        {
            t.normalize(std::declval<typename T::VectorRef>())
            } -> std::same_as<void>;

        {
            t.copy(std::declval<typename T::VectorCRef>(),
                   std::declval<typename T::VectorRef>())
            } -> std::same_as<void>;
    };

    namespace details {
        template <typename T>
        struct SupportsExpressionTemplates : std::false_type {};
    } // namespace details

    template <typename Backend>
    concept SupportsExpressionTemplates = IsBackend<Backend>
        &&details::SupportsExpressionTemplates<Backend>::value;
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_CONCEPT_HPP
