#
#################################
## GTest
#################################
#
##SET(CMAKE_CXX_FLAGS -pthread)
##enable_testing()
#set(GTEST_ROOT googletest)
#set(GTEST_LIBRARY googletest/build_dir/lib)
#set(GTEST_MAIN_LIBRARY googletest/build_dir/lib)
#set(GTEST_INCLUDE_DIR googletest/include)
#find_package(GTest REQUIRED)
##
##include_directories("googletest/include")
#add_executable(UnitTests
#        ../src/domain/data_generation/mandelbrot_generator.cpp
#        domain/data_generation/test_mandelbrot_generator.cxx)
#gtest_discover_tests(UnitTests TEST_LIST myTests_targets
#        )
#target_link_libraries(UnitTests Eigen3::Eigen)

set(BINARY ${CMAKE_PROJECT_NAME}_tests)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

set(SOURCES ${TEST_SOURCES})

add_executable(${BINARY} ${TEST_SOURCES})

add_test(NAME ${BINARY} COMMAND ${BINARY})

target_link_libraries(${BINARY} PUBLIC
        Eigen3::Eigen
        ${CMAKE_PROJECT_NAME}_lib
        gtest)
