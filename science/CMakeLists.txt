# Science modules
find_package(nlohmann_json CONFIG REQUIRED)

# --- photon ring substructure (Phase 3) — physics lib only ------------------
add_library(sgr_photon_ring STATIC
    photon_ring/SubRingExtractor.cpp
)
target_include_directories(sgr_photon_ring PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sgr_photon_ring PUBLIC sgr_physics nlohmann_json::nlohmann_json)

add_executable(subring_tool photon_ring/subring_tool.cpp)
target_link_libraries(subring_tool PRIVATE sgr_photon_ring)

# --- Ceres-based fitters (Phases 4–5) ---------------------------------------
if(SGR_BUILD_SCIENCE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/orbit_fitter/CMakeLists.txt)
    add_subdirectory(orbit_fitter)
endif()
if(SGR_BUILD_SCIENCE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/spectral_fitter/CMakeLists.txt)
    add_subdirectory(spectral_fitter)
endif()
