cmake_minimum_required(VERSION 3.4.0)

project(pyVO)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(./)

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(lib)

find_package(OpenGL REQUIRED)
find_package(Glew REQUIRED)

file(GLOB volib 
    src/base/*.c* 
    src/base/*.h* 
    src/estimator/*.c* 
    src/estimator/*.h* 
    src/util/*.c* 
    src/util/*.h* 
    src/optim/*.c* 
    src/optim/*.h*)

add_subdirectory(lib)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse -msse2 -msse3 -O3 -std=c++11 -fPIC --shared")
add_library(volib SHARED ${volib})

find_package(pybind11 REQUIRED)
file(GLOB pyvosrc pybind/*.cpp pybind/estimator/*.cpp pybind/base/*.cpp pybind/optim/*.cpp)
pybind11_add_module(pyVO ${pyvosrc})
target_link_libraries(pyVO PRIVATE volib ${GLOG_LIBRARIES})