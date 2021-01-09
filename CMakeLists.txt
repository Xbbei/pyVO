cmake_minimum_required(VERSION 3.4.0)

project(pyVO)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

file(GLOB volib base/*.c* base/*.h* estimator/*.c* estimator/*.h*)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse -msse2 -msse3 -O3 -std=c++11")
add_library(volib STATIC ${volib})

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJECT_NAME} pybind/*.cpp)
add_executable(${PROJECT_NAME} pybind/*.cpp)
target_link_libraries(${PROJECT_NAME} ${volib} ${GLOG_LIBRARIES})