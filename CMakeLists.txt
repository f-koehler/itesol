cmake_minimum_required(VERSION 3.20)
project(include/itesol CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# turn off eigen tests
option(BUILD_TESTING OFF)
add_subdirectory(dependencies/eigen)
add_subdirectory(dependencies/fmt)
add_subdirectory(dependencies/spdlog)

# check if IPO is supported
include(CheckIPOSupported)
check_ipo_supported(
    RESULT itesol_ipo_supported
    OUTPUT itesol_ipo_output
    LANGUAGES CXX
)

if(NOT itesol_ipo_supported)
    message(WARNING "IPO is not supported: ${itesol_ipo_output}")
endif()

find_package(Doxygen)

option(ITESOL_BUILD_TESTS "Build tests" ON)
option(ITESOL_BUILD_PYTHON "Build python module" OFF)
option(ITESOL_BUILD_DOCS "Build documentation" ${DOXYGEN_FOUND})

if(ITESOL_BUILD_TESTS)
    add_subdirectory(dependencies/Catch2)
    add_subdirectory(tests)
endif()

if(ITESOL_BUILD_PYTHON)
    add_subdirectory(dependencies/pybind11)
    add_subdirectory(python)
endif()

if(ITESOL_BUILD_DOCS)
    add_subdirectory(docs)
endif()
