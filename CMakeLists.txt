cmake_minimum_required(VERSION 3.10)

project(any LANGUAGES CXX)

include(cmake/clangd_compile_info.cmake)

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

add_library(any INTERFACE)
target_compile_features(any INTERFACE cxx_std_23)
target_include_directories(any INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# tests
CPMAddPackage(NAME Catch2 VERSION 3.11.0 GIT_REPOSITORY https://github.com/catchorg/Catch2.git QUIET)
add_subdirectory(test)

# examples
add_subdirectory(example)
