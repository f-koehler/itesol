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
    class IsEigenDenseMatrix : public std::false_type {};

    template <typename Scalar, int Rows, int Columns, int Options, int MaxRows,
              int MaxColumns>
    class IsEigenDenseMatrix<
        Eigen::Matrix<Scalar, Rows, Columns, Options, MaxRows, MaxColumns>>
        : public std::true_type {};

    template <typename T>
    using ScalarType = typename std::enable_if<IsEigenDense<T>::value,
                                               typename T::Scalar>::type;

    template <typename T>
    using VectorType = typename std::enable_if<
        IsEigenDenseMatrix<T>::value,
        Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime, 1, T::Options,
                      T::MaxRowsAtCompileTime, 1>>::type;

    template <typename T>
    using Ref =
        typename std::enable_if<IsEigenDense<T>::value, Eigen::Ref<T>>::type;

    template <typename T>
    using CRef = typename std::enable_if<IsEigenDense<T>::value,
                                         const Eigen::Ref<const T> &>::type;

    template <typename T>
    using LinearOperator =
        typename std::enable_if<IsEigenDenseVector<T>::value,
                                std::function<void(CRef<T>, Ref<T>)>>::type;

    template <typename Matrix>
    typename std::enable_if<IsEigenDenseMatrix<Matrix>::value,
                            LinearOperator<VectorType<Matrix>>>::type
    make_linear_operator(CRef<Matrix> matrix) {
        return [&matrix](CRef<VectorType<Matrix>> x,
                         Ref<VectorType<Matrix>> y) { y = matrix * x; };
    }

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
