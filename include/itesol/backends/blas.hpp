#ifndef ITESOL_BACKENDS_BLAS_HPP
#define ITESOL_BACKENDS_BLAS_HPP

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <spdlog/spdlog.h>

#include "../concepts.hpp"

#include <functional>
#include <memory>
#include <random>

namespace itesol::backends {
    template <typename Scalar>
    concept BlasScalar =
        std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, std::complex<float>> ||
        std::is_same_v<Scalar, std::complex<double>>;

    template <BlasScalar ScalarT>
    class Blas {
      public:
        using Scalar = ScalarT;
        using RealScalar = RealType<Scalar>;
        using Index = int;

        struct Matrix {
            const Index m_rows;
            const Index m_columns;
            std::shared_ptr<Scalar[]> data;

            Index rows() const { return m_rows; }

            Index columns() const { return m_rows; }

            Scalar &operator()(Index row, Index column) {
                return data.get()[row * m_columns + column];
            }

            const Scalar &operator()(Index row, Index column) const {
                return data.get()[row * m_columns + column];
            }
        };

        struct Vector {
            const Index m_rows;
            std::shared_ptr<Scalar[]> data;

            Index rows() const { return m_rows; }
        };

        using MatrixRef = Matrix &;
        using MatrixCRef = Matrix &;
        using VectorRef = Vector &;
        using VectorCRef = Vector &;

        using LinearOperator = std::function<void(VectorCRef, VectorRef)>;

        Vector create_vector(Index rows) {
            return Vector{
                rows, std::shared_ptr<Scalar[]>(
                          new Scalar[rows], std::default_delete<Scalar[]>())};
        }

        Vector create_zero_vector(Index rows) {
            auto v = create_vector(rows);
            std::fill_n(v.data.get(), rows, Scalar(0));
            return v;
        }

        Vector create_random_vector(Index rows) {
            auto v = create_vector(rows);
            std::mt19937_64 prng;
            std::uniform_real_distribution<RealScalar> dist(0, 1);
            std::generate_n(v.data.get(), rows, [&] { return dist(prng); });
            return v;
        }

        Matrix create_matrix(Index rows, Index cols) {
            return Matrix{
                rows, cols,
                std::shared_ptr<Scalar[]>(new Scalar[rows * cols],
                                          std::default_delete<Scalar[]>())};
        }

        Matrix create_zero_matrix(Index rows, Index cols) {
            auto m = create_matrix(rows, cols);
            std::fill_n(m.data.get(), rows * cols, Scalar(0));
            return m;
        }

        LinearOperator make_linear_operator(MatrixCRef matrix) {
            return [matrix](VectorCRef x, VectorRef y) {
                if (matrix.m_rows != y.m_rows) {
                    throw std::runtime_error("matrix.m_rows != y.m_rows");
                }
                if (matrix.m_columns != x.m_rows) {
                    throw std::runtime_error("matrix.m_columns != x.m_rows");
                }
                if constexpr (std::is_same_v<Scalar, float>) {
                    cblas_cgemv(CblasRowMajor, CblasNoTrans, matrix.m_rows,
                                matrix.m_columns, Scalar(1.), matrix.data.get(),
                                matrix.m_columns, x.data.get(), 1, Scalar(0.),
                                y.data.get(), 1);
                } else if constexpr (std::is_same_v<Scalar, double>) {
                    cblas_dgemv(CblasRowMajor, CblasNoTrans, matrix.m_rows,
                                matrix.m_columns, Scalar(1.), matrix.data.get(),
                                matrix.m_columns, x.data.get(), 1, Scalar(0.),
                                y.data.get(), 1);
                } else if constexpr (std::is_same_v<Scalar,
                                                    std::complex<float>>) {
                    const auto one = Scalar(1.);
                    const auto zero = Scalar(0.);
                    cblas_cgemv(CblasRowMajor, CblasNoTrans, matrix.m_rows,
                                matrix.m_columns, &one, matrix.data.get(),
                                matrix.m_columns, x.data.get(), 1, &zero,
                                y.data.get(), 1);
                } else {
                    const auto one = Scalar(1.);
                    const auto zero = Scalar(0.);
                    cblas_zgemv(CblasRowMajor, CblasNoTrans, matrix.m_rows,
                                matrix.m_columns, &one, matrix.data.get(),
                                matrix.m_columns, x.data.get(), 1, &zero,
                                y.data.get(), 1);
                }
            };
        }

        Scalar dotc(VectorCRef x, VectorCRef y) {
            if (x.m_rows != y.m_rows) {
                throw std::length_error("x and y must have the same length");
            }
            if constexpr (std::is_same_v<Scalar, float>) {
                return cblas_cdot(x.m_rows, x.data.get(), 1, y.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, double>) {
                return cblas_ddot(x.m_rows, x.data.get(), 1, y.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
                Scalar result;
                cblas_cdotc(x.m_rows, x.data.get(), 1, y.data.get(), 1,
                            &result);
                return result;
            } else {
                Scalar result;
                cblas_zdotc(x.m_rows, x.data.get(), 1, y.data.get(), 1,
                            &result);
                return result;
            }
        }

        RealScalar norm(VectorCRef x) {
            if constexpr (std::is_same_v<Scalar, float>) {
                return cblas_snrm2(x.m_rows, x.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, double>) {
                return cblas_dnrm2(x.m_rows, x.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
                return cblas_scnrm2(x.m_rows, x.data.get(), 1);
            } else {
                return cblas_dznrm2(x.m_rows, x.data.get(), 1);
            }
        }

        void a_x_plus_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            if (x.m_rows != y.m_rows) {
                throw std::length_error("x and y must have the same length");
            }
            if (!x.data || !y.data) {
                throw std::runtime_error("x and y must not be empty");
            }
            if constexpr (std::is_same_v<Scalar, float>) {
                return cblas_saxpy(x.m_rows, alpha, x.data.get(), 1,
                                   y.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, double>) {
                return cblas_daxpy(x.m_rows, alpha, x.data.get(), 1,
                                   y.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
                return cblas_caxpy(x.m_rows, alpha, x.data.get(), 1,
                                   y.data.get(), 1);
            } else {
                return cblas_zaxpy(x.m_rows, alpha, x.data.get(), 1,
                                   y.data.get(), 1);
            }
        }
        void x_plus_a_y(const Scalar &alpha, VectorCRef x, VectorRef y) {
            scale(alpha, y);
            a_x_plus_y(Scalar(1), x, y);
        }

        void scale(Scalar alpha, VectorRef x) {
            if constexpr (std::is_same_v<Scalar, float>) {
                cblas_csscal(x.m_rows, alpha, x.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, double>) {
                cblas_zdscal(x.m_rows, alpha, x.data.get(), 1);
            } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
                cblas_cscal(x.m_rows, &alpha, x.data.get(), 1);
            } else {
                cblas_zscal(x.m_rows, &alpha, x.data.get(), 1);
            }
        }

        void normalize(VectorRef x) { scale(Scalar(1) / norm(x), x); }

        void copy(VectorCRef x, VectorRef y) {
            if (x.m_rows != y.m_rows) {
                throw std::length_error("x and y must have the same length");
            }
            std::copy_n(x.data.get(), x.m_rows, y.data.get());
        }
    };
} // namespace itesol::backends

#endif // ITESOL_BACKENDS_BLAS_HPP
