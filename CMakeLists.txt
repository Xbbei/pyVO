cmake_minimum_required(VERSION 3.4.0)

project(pyVO)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(./)

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

file(GLOB volib base/*.c* base/*.h* estimator/*.c* estimator/*.h*)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse -msse2 -msse3 -O3 -std=c++11")
add_library(volib SHARED ${volib})

find_package(pybind11 REQUIRED)
file(GLOB pyvosrc pybind/*.cpp)
pybind11_add_module(pyVO ${pyvosrc})
target_link_libraries(pyVO PRIVATE volib ${GLOG_LIBRARIES})