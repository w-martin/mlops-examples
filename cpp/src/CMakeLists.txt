
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED)

set(BINARY ${CMAKE_PROJECT_NAME})
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.cpp)
set(SOURCES ${SOURCES})
file(GLOB_RECURSE INCLUDES LIST_DIRECTORIES true *.h)
set(INCLUDES ${INCLUDES})

add_library(${BINARY}_lib SHARED ${SOURCES})
target_link_libraries(${BINARY}_lib Eigen3::Eigen OpenMP::OpenMP_CXX)
add_executable(${BINARY}_run ${SOURCES})
target_link_libraries(${BINARY}_run ${BINARY}_lib)
