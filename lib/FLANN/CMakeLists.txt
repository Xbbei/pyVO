if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

add_library(flann
    flann.hpp flann.cpp
    ext/lz4.h ext/lz4.c
    ext/lz4hc.h ext/lz4hc.c
)
