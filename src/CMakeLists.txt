cmake_minimum_required(VERSION 3.12)
project(extension)

set(CMAKE_CXX_STANDARD 11)

set(QOCO_BUILD_TYPE "Release")

find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QOCO")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qoco
  GIT_REPOSITORY https://github.com/govindchari/qoco.git
  GIT_TAG f5fb2ee6fb5b51725cd3b79d3bfd440b6bff0d53
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

add_library(qoco_ext MODULE bindings.cpp)
target_link_libraries(qoco_ext PRIVATE pybind11::module qoco)
set_target_properties(qoco_ext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
