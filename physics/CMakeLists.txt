# sgr_physics — CPU physics foundation (Phase 0)
find_package(spdlog CONFIG REQUIRED)

add_library(sgr_physics STATIC
    src/constants.cpp
    src/metric.cpp
    src/null_geodesics.cpp
    src/timelike_geodesics.cpp
    src/hartle_thorne.cpp
)
target_include_directories(sgr_physics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(sgr_physics PUBLIC spdlog::spdlog)

if(SGR_BUILD_TESTS)
    find_package(Catch2 3 CONFIG REQUIRED)
    add_executable(sgr_physics_tests
        tests/test_schwarzschild_limit.cpp
        tests/test_key_radii.cpp
        tests/test_conserved_quantities_null.cpp
        tests/test_timelike_normalisation.cpp
        tests/test_hartle_thorne_limit.cpp
    )
    target_link_libraries(sgr_physics_tests PRIVATE
        sgr_physics Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(sgr_physics_tests)
endif()

if(SGR_BUILD_BENCH)
    find_package(benchmark CONFIG REQUIRED)
    add_executable(geodesic_bench benches/geodesic_bench.cpp)
    target_link_libraries(geodesic_bench PRIVATE sgr_physics benchmark::benchmark_main)
endif()
