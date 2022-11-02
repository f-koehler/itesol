
include(CTest)
include(Catch)
enable_testing()

find_package(BLAS)

add_executable(
    itesol_test
    tests.cpp
    concepts.cpp
    backends/blas.cpp
    backends/eigen_dense.cpp
    algorithms/power_method.cpp
    test_matrices/random_symmetric.cpp
)
target_compile_options(itesol_test PRIVATE -Wall -Wextra -Wpedantic)
include_directories(itesol_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(itesol_test PRIVATE Eigen3::Eigen)
target_link_libraries(itesol_test PRIVATE fmt::fmt)
target_link_libraries(itesol_test PRIVATE spdlog::spdlog)
target_link_libraries(itesol_test PRIVATE Catch2::Catch2WithMain)

if(BLAS_FOUND)
    target_link_libraries(itesol_test PRIVATE BLAS::BLAS)
    target_link_libraries(itesol_test PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(itesol_test PRIVATE ITESOL_TEST_USE_BLAS)
endif()

if(itesol_ipo_supported)
    set_property(TARGET itesol_test PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

catch_discover_tests(itesol_test)
