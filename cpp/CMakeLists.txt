
cmake_minimum_required(VERSION 3.17)
project(mlops_examples)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -march=native -O3 -ffast-math")

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(src)

add_subdirectory(src)
add_subdirectory(googletest)
add_subdirectory(test)
