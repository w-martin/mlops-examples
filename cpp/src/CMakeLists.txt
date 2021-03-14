

set(BINARY ${CMAKE_PROJECT_NAME})


file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})

add_executable(${BINARY}_run ${SOURCES})
target_link_libraries(${BINARY}_run Eigen3::Eigen OpenMP::OpenMP_CXX)
add_library(${BINARY}_lib STATIC ${SOURCES})
target_link_libraries(${BINARY}_lib Eigen3::Eigen OpenMP::OpenMP_CXX)
#add_executable(buildModel src/modelBuilder.cpp
#        src/domain/data_generation/data_generator.h
#        src/domain/data_generation/mandelbrot_generator.cpp
#        src/domain/data_generation/mandelbrot_generator.h)