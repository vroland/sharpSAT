cmake_minimum_required (VERSION 3.2)
project (SHARPSAT CXX CUDA) 

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -Wall -fdiagnostics-color=always")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3 -DNDEBUG -Wall")

set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -std=c++17 -O3 -pg -g -DNDEBUG -Wall")

find_library(GMP_LIB gmp)
find_library(GMPXX_LIB gmpxx)

add_subdirectory(src)

add_executable (sharpSAT src/main.cpp)
target_link_libraries(sharpSAT libsharpSAT ${GMP_LIB} ${GMPXX_LIB}) 

# FIXME: Encapsulate CUDA in gpusat
target_include_directories(sharpSAT PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
