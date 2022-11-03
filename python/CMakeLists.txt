pybind11_add_module(itesol_core main.cpp)
include_directories(itesol_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(itesol_core PRIVATE Eigen3::Eigen)
target_link_libraries(itesol_core PRIVATE fmt::fmt)
target_link_libraries(itesol_core PRIVATE spdlog::spdlog)

if(itesol_ipo_supported)
    set_property(TARGET itesol_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
