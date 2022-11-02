pybind11_add_module(itesol_core main.cpp)
include_directories(itesol_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(itesol_core PRIVATE Eigen3::Eigen)

if(itesol_ipo_supported)
    set_property(TARGET itesol_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()