#ifndef INCLUDE_ITESOL_TEST_MATRICES_RANDOM_SYMMETRIC_HPP
#define INCLUDE_ITESOL_TEST_MATRICES_RANDOM_SYMMETRIC_HPP

#include <random>

#include "../backends/concept.hpp"
#include "../concepts.hpp"

namespace itesol::test_matrices {
    template <itesol::backends::IsBackend Backend>
    void initialize_random_symmetric(typename Backend::MatrixRef matrix) {
        std::mt19937_64 prng;
        std::uniform_real_distribution<typename Backend::RealScalar> dist(-1,
                                                                          1);

        if constexpr (IsComplex<typename Backend::Scalar>) {
            for (typename Backend::Index i = 0; i < matrix.rows(); ++i) {
                for (typename Backend::Index j = 0; j < i; ++j) {
                    matrix(j, i).real(dist(prng));
                    matrix(j, i).imag(dist(prng));
                    matrix(i, j) = std::conj(matrix(j, i));
                }
                matrix(i, i).real(dist(prng));
                matrix(i, i).imag(typename Backend::RealScalar(0));
            }
        } else {
            for (typename Backend::Index i = 0; i < matrix.rows(); ++i) {
                for (typename Backend::Index j = 0; j < i; ++j) {
                    matrix(i, j) = matrix(j, i) = dist(prng);
                }
                matrix(i, i) = dist(prng);
            }
        }
    }

} // namespace itesol::test_matrices

#endif // INCLUDE_ITESOL_TEST_MATRICES_RANDOM_SYMMETRIC_HPP
