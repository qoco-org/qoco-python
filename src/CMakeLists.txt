cmake_minimum_required(VERSION 3.12)
project(extension)

set(CMAKE_CXX_STANDARD 11)

set(QCOS_BUILD_TYPE "Release")

find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QCOS")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qcos
  GIT_REPOSITORY https://github.com/govindchari/qcos.git
  GIT_TAG f971d4334690fc6ee8826b5e886fbdb5edca592a
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qcos)

add_library(qcos_ext MODULE bindings.cpp)
target_link_libraries(qcos_ext PRIVATE pybind11::module qcos)
set_target_properties(qcos_ext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
