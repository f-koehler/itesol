cmake_minimum_required(VERSION 3.20)
project(itesol CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# turn off eigen tests
option(BUILD_TESTING OFF)

add_subdirectory(dependencies/eigen)
add_subdirectory(dependencies/fmt)
add_subdirectory(dependencies/spdlog)

# check if IPO is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT itesol_ipo_supported OUTPUT itesol_ipo_output LANGUAGES CXX)
if(NOT itesol_ipo_supported)
    message(WARNING "IPO is not supported: ${itesol_ipo_output}")
endif()

add_executable(itesol main.cpp itesol/power_method.hpp)
target_link_libraries(itesol PRIVATE Eigen3::Eigen)
target_link_libraries(itesol PRIVATE fmt::fmt)
target_link_libraries(itesol PRIVATE spdlog::spdlog)
if(itesol_ipo_supported)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
