pybind11_add_module(itesol_core main.cpp)
target_link_libraries(itesol_core PRIVATE Eigen3::Eigen)