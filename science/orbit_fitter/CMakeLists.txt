# S2 orbit fitter (Ceres) — Phases 4 & 5
find_package(Ceres CONFIG REQUIRED)

add_library(sgr_orbit_fitter STATIC
    OrbitFitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../spectral_fitter/SpectralLineFitter.cpp
)
target_include_directories(sgr_orbit_fitter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sgr_orbit_fitter PUBLIC sgr_s2_model sgr_physics Ceres::ceres)

add_executable(fit_s2_tool fit_s2_tool.cpp)
target_link_libraries(fit_s2_tool PRIVATE sgr_orbit_fitter nlohmann_json::nlohmann_json)

# Thesis: joint multi-star cluster fit (paper/THESIS_PLAN.md §2.1)
add_executable(fit_cluster fit_cluster.cpp)
target_link_libraries(fit_cluster PRIVATE sgr_orbit_fitter nlohmann_json::nlohmann_json)

if(SGR_BUILD_TESTS)
    find_package(Catch2 3 CONFIG REQUIRED)
    add_executable(sgr_science_tests
        tests/test_s2_model.cpp
    )
    target_link_libraries(sgr_science_tests PRIVATE
        sgr_orbit_fitter Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(sgr_science_tests)
endif()
