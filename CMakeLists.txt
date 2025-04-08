cmake_minimum_required(VERSION 3.15)
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
  GIT_REPOSITORY https://github.com/qoco-org/qoco.git
  GIT_TAG 7526022691e75987aa7b7f1a6e897298011795ee
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

pybind11_add_module(qoco_ext src/bindings.cpp)
target_include_directories(qoco_ext INTERFACE ${qoco_SOURCE_DIR}/include)
target_link_libraries(qoco_ext PUBLIC pybind11::module qocostatic)
install(TARGETS qoco_ext DESTINATION . COMPONENT python)
