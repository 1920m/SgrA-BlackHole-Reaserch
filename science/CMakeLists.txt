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

# --- S-star GR orbit model (Ceres-free) -------------------------------------
# Split out of the fitter so the engine's live 3-D cluster viewer can animate
# the same integrated geodesics the fits use, without pulling Ceres in.
add_library(sgr_s2_model STATIC orbit_fitter/SkyProjection.cpp)
target_include_directories(sgr_s2_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sgr_s2_model PUBLIC sgr_physics)

# --- Ceres-based fitters (Phases 4–5) ---------------------------------------
if(SGR_BUILD_SCIENCE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/orbit_fitter/CMakeLists.txt)
    add_subdirectory(orbit_fitter)
endif()
if(SGR_BUILD_SCIENCE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/spectral_fitter/CMakeLists.txt)
    add_subdirectory(spectral_fitter)
endif()

# --- hot-spot flares (Gate 18, extension paper) ------------------------------
add_library(sgr_flares STATIC flares/Hotspot.cpp)
target_include_directories(sgr_flares PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flares)
target_link_libraries(sgr_flares PUBLIC sgr_physics)

add_executable(flare_tool flares/flare_tool.cpp)
target_link_libraries(flare_tool PRIVATE sgr_flares nlohmann_json::nlohmann_json)

if(SGR_BUILD_TESTS)
    find_package(Catch2 3 CONFIG REQUIRED)
    add_executable(sgr_flares_tests flares/tests/test_hotspot.cpp)
    target_link_libraries(sgr_flares_tests PRIVATE sgr_flares Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(sgr_flares_tests)
endif()

# --- hot accretion flow at 230 GHz (Gate 19, extension paper) ----------------
add_library(sgr_riaf STATIC riaf/Riaf.cpp)
target_include_directories(sgr_riaf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/riaf)
target_link_libraries(sgr_riaf PUBLIC sgr_physics)

add_executable(riaf_tool riaf/riaf_tool.cpp)
target_link_libraries(riaf_tool PRIVATE sgr_riaf nlohmann_json::nlohmann_json)

if(SGR_BUILD_TESTS)
    add_executable(sgr_riaf_tests riaf/tests/test_riaf.cpp riaf/tests/test_polarisation.cpp)
    target_link_libraries(sgr_riaf_tests PRIVATE sgr_riaf Catch2::Catch2WithMain)
    catch_discover_tests(sgr_riaf_tests)
endif()

# --- exotic spacetime: the Dneg wormhole (Gate 21; NOT a model of Sgr A*) ----
add_library(sgr_wormhole STATIC wormhole/Wormhole.cpp)
target_include_directories(sgr_wormhole PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/wormhole)
target_link_libraries(sgr_wormhole PUBLIC sgr_physics)

add_executable(wormhole_tool wormhole/wormhole_tool.cpp)
target_link_libraries(wormhole_tool PRIVATE sgr_wormhole nlohmann_json::nlohmann_json)

if(SGR_BUILD_TESTS)
    add_executable(sgr_wormhole_tests wormhole/tests/test_wormhole.cpp)
    target_link_libraries(sgr_wormhole_tests PRIVATE sgr_wormhole Catch2::Catch2WithMain)
    catch_discover_tests(sgr_wormhole_tests)
endif()
