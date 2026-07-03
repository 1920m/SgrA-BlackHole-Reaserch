# Sgr A* Scientific Simulator — root build
# C++20 / CMake 3.25+ / Ninja / vcpkg manifest mode.
cmake_minimum_required(VERSION 3.25)

project(sgr_a_research_sim
    VERSION 0.1.0
    DESCRIPTION "Physically rigorous simulation of Sagittarius A*"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SGR_BUILD_TESTS      "Build Catch2 unit tests"            ON)
option(SGR_BUILD_BENCH      "Build Google Benchmark benchmarks"  ON)
option(SGR_BUILD_SCIENCE    "Build science modules (needs Ceres)" OFF)
option(SGR_BUILD_ENGINE     "Build Vulkan engine"                OFF)

if(MSVC)
    # /fp:precise (default) is mandatory: geodesic integration must not be
    # reordered by fast-math. /EHsc for Ceres/Catch2 exception model.
    add_compile_options(/W4 /permissive- /EHsc /utf-8 /Zc:__cplusplus)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(SGR_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(physics)

if(SGR_BUILD_SCIENCE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/science/CMakeLists.txt)
    add_subdirectory(science)
endif()

if(SGR_BUILD_ENGINE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/engine/CMakeLists.txt)
    add_subdirectory(engine)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/validation/CMakeLists.txt)
    add_subdirectory(validation)
endif()
