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
  GIT_TAG b38e6dd0755508fae98a1fd1ed5cbdb283c5f157
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

pybind11_add_module(qoco_ext src/bindings.cpp)
target_include_directories(qoco_ext INTERFACE ${qoco_SOURCE_DIR}/include)

if(${QOCO_ALGEBRA_BACKEND} STREQUAL "builtin")
target_link_libraries(qoco_ext PUBLIC pybind11::module qocostatic)
elseif(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDA)
    target_link_libraries(qoco_ext PUBLIC pybind11::module qocostatic cudart)
endif()

install(TARGETS qoco_ext DESTINATION . COMPONENT python)
