# Build glue for the AMReX HeatEquation_EX0_C tutorial.
#
# The tutorial source itself is not vendored here: HEAT_SOURCE_DIR points at a
# checkout of AMReX-Codes/amrex-tutorials fetched by
# scripts/build_amrex_cumetal.sh. The tutorials repository's own CMake expects
# the whole ExampleCodes tree and a macro it defines at the top level, so this
# file builds the one example directly instead.
#
# One CMakeLists serves both builds. AMReX_GPU_BACKEND comes from the installed
# AMReXConfig.cmake, so pointing AMReX_ROOT at the CUDA install produces the
# CuMetal binary and pointing it at the CPU install produces the reference --
# from the same source, with the same compiler.
cmake_minimum_required(VERSION 3.20)
project(amrex_heat_equation LANGUAGES CXX)

if (NOT DEFINED HEAT_SOURCE_DIR)
    message(FATAL_ERROR "HEAT_SOURCE_DIR must point at HeatEquation_EX0_C/Source")
endif ()

find_package(AMReX REQUIRED)
include(AMReXTargetHelpers)

if (AMReX_GPU_BACKEND STREQUAL CUDA)
    enable_language(CUDA)
endif ()

add_executable(heat "${HEAT_SOURCE_DIR}/main.cpp")
target_include_directories(heat PRIVATE "${HEAT_SOURCE_DIR}")
target_link_libraries(heat PRIVATE AMReX::amrex_3d)

if (AMReX_GPU_BACKEND STREQUAL CUDA)
    # AMReX compiles .cpp as CUDA when the backend is CUDA; the tutorial's
    # ParallelFor bodies are __device__ lambdas in an ordinary .cpp file.
    setup_target_for_cuda_compilation(heat)
endif ()
