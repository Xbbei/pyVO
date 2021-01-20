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

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS
             program_options
             filesystem
             graph
             system
             unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PythonLibs 3 REQUIRED)
MESSAGE( " *** PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS} )
MESSAGE( " *** PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES} )

set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /opt/conda/lib/python3.8/site-packages/numpy/core/include)

include_directories(${PYTHON_INCLUDE_DIRS})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB volib 
    src/base/*.c* 
    src/base/*.h* 
    src/estimator/*.c* 
    src/estimator/*.h* 
    src/util/*.c* 
    src/util/*.h* 
    src/optim/*.c* 
    src/optim/*.h*
    src/feature/*.c*
    src/feature/*.h*)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse -msse2 -msse3 -O3 -std=c++11 -fPIC")
add_subdirectory(lib)

add_library(volib SHARED ${volib})

set (CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)
file(GLOB pyvosrc pybind/*.cpp pybind/estimator/*.cpp pybind/base/*.cpp pybind/optim/*.cpp pybind/feature/*.cpp pybind/utils/*.c*)
pybind11_add_module(pyVO ${pyvosrc})
target_link_libraries(pyVO PRIVATE volib ${GLOG_LIBRARIES} ${OpenCV_LIBS} sift_gpu vlfeat flann
    ${SIFT_GPU_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})