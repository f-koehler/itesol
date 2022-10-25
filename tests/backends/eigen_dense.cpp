#include <catch2/catch_template_test_macros.hpp>

#include <itesol/backends/concept.hpp>
#include <itesol/backends/eigen_dense.hpp>

TEMPLATE_TEST_CASE("EigenDense fulfills Backend concept", "[backends]", float,
                   double, std::complex<float>, std::complex<double>) {
    REQUIRE(itesol::backends::Backend<itesol::backends::EigenDense<TestType>>);
}
