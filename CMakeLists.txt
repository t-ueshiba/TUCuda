cmake_minimum_required(VERSION 3.8)
project(TUCuda)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD		17)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CUDA_PROPAGATE_HOST_FLAGS	ON)
set(CUDA_NVCC_FLAGS		"-std=c++17 --extended-lambda --disable-warnings")

if(APPLE)
  set(MACOSX_RPATH TRUE)
endif()

set(CMAKE_MODULE_PATH    ${CMAKE_MODULE_PATH} "/usr/local/share/cmake/TU")
include(BuildType)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message("-- Cuda version: " ${CUDA_VERSION})
endif()

find_package(TULibs REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${TULIBS_INCLUDE_DIR})

link_directories(
    ${TULIBS_LIBRARY_DIR})

file(GLOB sources *.cc *.cu)

cuda_add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME} TUTools)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(TU)

option(BUILD_TUCUDAPP_TEST_PROGRAMS "Building test programs for TUCuda" OFF)

if(BUILD_TUCUDAPP_TEST_PROGRAMS)
  add_subdirectory(Testprog)
endif()
