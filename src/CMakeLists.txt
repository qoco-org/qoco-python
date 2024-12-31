cmake_minimum_required(VERSION 3.12)
project(extension)

set(CMAKE_CXX_STANDARD 11)

set(QOCO_BUILD_TYPE "Release")

# Detect operating system.
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_compile_definitions(IS_LINUX)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_compile_definitions(IS_MACOS)
endif()

find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QOCO")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qoco
  GIT_REPOSITORY git@github.com:qoco-org/qoco.git
  GIT_TAG 43e81c954a18a02548b398e15bf6b51bc9568bdf
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

add_library(qoco_ext MODULE bindings.cpp)
target_link_libraries(qoco_ext PRIVATE pybind11::module qoco)
set_target_properties(qoco_ext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
