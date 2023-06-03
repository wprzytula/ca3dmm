cmake_minimum_required(VERSION 3.10)
project(CA3DMM CXX)

SET(CMAKE_CXX_COMPILER mpicxx)   # use CC for okeanos
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

add_executable(
        ca3dmm
        ca3dmm.cc
        densematgen.cpp
)

target_link_libraries(ca3dmm ${MPI_C_LIBRARIES})
