cmake_minimum_required(VERSION 3.18)
project(mlops_examples)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native -O1")

include_directories(src)
add_subdirectory(eigen)
add_subdirectory(src)
add_subdirectory(test)
