#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

#include <itesol/concepts.hpp>

TEMPLATE_TEST_CASE("IsComplex type trait", "[type_traits]", float, double) {
    SECTION("Real types") { REQUIRE(!itesol::IsComplex<TestType>); }
    SECTION("Complex types") {
        REQUIRE(itesol::IsComplex<std::complex<TestType>>);
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
