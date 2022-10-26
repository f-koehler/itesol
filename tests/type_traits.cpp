#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

#include <itesol/type_traits.hpp>

TEMPLATE_TEST_CASE("IsComplex type trait", "[type_traits]", float, double) {
    SECTION("Real types") {
        REQUIRE(!itesol::IsComplex<TestType>::value);
        REQUIRE(std::is_same_v<typename itesol::IsComplex<TestType>::RealType,
                               TestType>);
    }
    SECTION("Complex types") {
        REQUIRE(itesol::IsComplex<std::complex<TestType>>::value);
        REQUIRE(std::is_same_v<
                typename itesol::IsComplex<std::complex<TestType>>::RealType,
                TestType>);
    }
}

TEMPLATE_TEST_CASE("RealType alias", "[type_traits]", float, double) {
    SECTION("Real types") {
        REQUIRE(std::is_same_v<itesol::RealType<TestType>, TestType>);
    }
    SECTION("Complex types") {
        REQUIRE(
            std::is_same_v<itesol::RealType<std::complex<TestType>>, TestType>);
    }
}
