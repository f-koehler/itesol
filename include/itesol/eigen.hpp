#ifndef ITESOL_EIGEN_HPP
#define ITESOL_EIGEN_HPP

#include <utility>

#include "Eigen/Dense"

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
} // namespace itesol

#endif // ITESOL_EIGEN_HPP
