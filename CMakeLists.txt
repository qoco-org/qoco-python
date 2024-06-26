cmake_minimum_required(VERSION 3.12)
project(example)

set(CMAKE_CXX_STANDARD 14)

# find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(example MODULE bindings.cpp example.c)
target_link_libraries(example PRIVATE pybind11::module)
set_target_properties(example PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
