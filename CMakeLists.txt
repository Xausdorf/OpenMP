cmake_minimum_required(VERSION 3.23)
project(practice)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_STANDARD 20)

add_executable(OpenMP main.cpp)
