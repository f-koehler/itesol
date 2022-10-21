#ifndef ITESOL_EIGEN_HPP
#define ITESOL_EIGEN_HPP

#include <utility>

#include <Eigen/Dense>

namespace itesol {
    template <typename T>
    class IsEigenDense : public std::false_type {};

    template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
              int MaxCols>
    class IsEigenDense<
        Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
        : public std::true_type {};

    template <typename T>
    class IsEigenDenseVector : public std::false_type {};

    template <typename Scalar, int Rows, int Options, int MaxRows>
    class IsEigenDenseVector<
        Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, 1>>
        : public std::true_type {};

    template <typename T>
    using ScalarType = typename std::enable_if<IsEigenDense<T>::value,
                                               typename T::Scalar>::type;

    template <typename T>
    using Ref =
        typename std::enable_if<IsEigenDense<T>::value, Eigen::Ref<T>>::type;

    template <typename T>
    using CRef = typename std::enable_if<IsEigenDense<T>::value,
                                         const Eigen::Ref<const T> &>::type;

    template <typename ScalarT>
    class EigenDenseAllocator {
      public:
        using Scalar = ScalarT;
        using Index = int;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        Vector create_vector(Index rows) { return Vector(rows); }

        Vector create_zero_vector(Index rows) { return Vector::Zero(rows); }

        Vector create_random_vector(Index rows) { return Vector::Random(rows); }
    };

} // namespace itesol

#endif // ITESOL_EIGEN_HPP
