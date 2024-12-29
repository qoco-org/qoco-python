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
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_compile_definitions(IS_WINDOWS)
endif()

find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QOCO")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qoco
  GIT_REPOSITORY https://github.com/govindchari/qoco.git
  GIT_TAG 68f6a7695a911d5adcb35e05ef142376e9f6168c
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

add_library(qoco_ext MODULE bindings.cpp)
target_link_libraries(qoco_ext PRIVATE pybind11::module qoco)
set_target_properties(qoco_ext PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
