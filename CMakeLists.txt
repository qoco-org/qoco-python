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
  GIT_TAG d40cb8170b0e967be38ad0e1b134d9c51f5d636e
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
pybind11_add_module(${QOCO_EXT_MODULE_NAME} src/bindings.cpp)
target_include_directories(${QOCO_EXT_MODULE_NAME} INTERFACE ${qoco_SOURCE_DIR}/include)
install(TARGETS ${QOCO_EXT_MODULE_NAME} DESTINATION . COMPONENT python)

if(${QOCO_ALGEBRA_BACKEND} STREQUAL "builtin")
target_link_libraries(qoco_ext PUBLIC pybind11::module qocostatic)
elseif(${QOCO_ALGEBRA_BACKEND} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDA)
    target_link_libraries(qoco_cuda PUBLIC pybind11::module qocostatic)
endif()