cmake_minimum_required(VERSION 3.12)
project(qcos_ext)

set(CMAKE_CXX_STANDARD 14)

# find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QCOS")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qcos
  GIT_REPOSITORY https://github.com/govindchari/qcos.git
  GIT_TAG d9b4b45d0dee7d7fb9e3159b8d6d80ae3f4fbc98
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qcos)

add_library(qcos_ext MODULE bindings.cpp example.c)
target_link_libraries(qcos_ext PRIVATE pybind11::module)
set_target_properties(qcos_ext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
