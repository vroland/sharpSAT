add_subdirectory (component_types)

SET(SOURCES alt_component_analyzer.cpp 
	component_cache.cpp
	component_management.cpp
	instance.cpp
	solver.cpp
	statistics.cpp
        gpusolver.cpp)

set(WITH_CLI Off CACHE BOOL "Enable GPUSAT cli build" FORCE)
add_subdirectory(../GPUSAT GPUSAT)

ADD_LIBRARY(libsharpSAT ${SOURCES} component_types)
target_link_libraries (libsharpSAT component_types gpusat) 
# FIXME: Encapsulate CUDA in gpusat
target_include_directories(libsharpSAT PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
