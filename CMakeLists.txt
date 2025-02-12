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
endif()

find_package(pybind11 REQUIRED)

message(STATUS "Fetching/configuring QOCO")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
include(FetchContent)
FetchContent_Declare(
  qoco
  GIT_REPOSITORY https://github.com/qoco-org/qoco.git
  GIT_TAG d2bbfc5d2d921242e4e94e4a2254d7adec3632b4
)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(qoco)

pybind11_add_module(qoco_ext src/bindings.cpp)
target_include_directories(qoco_ext INTERFACE ${qoco_SOURCE_DIR}/include)
target_link_libraries(qoco_ext PUBLIC pybind11::module qocostatic)
install(TARGETS qoco_ext DESTINATION . COMPONENT python)
