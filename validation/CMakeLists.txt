# Validation tools: reference renders + CPU/GPU comparison (Phase 7)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb REQUIRED)

add_executable(render_reference render_reference.cpp)
target_link_libraries(render_reference PRIVATE sgr_physics nlohmann_json::nlohmann_json)

add_executable(compare_shadow compare_shadow.cpp)
target_include_directories(compare_shadow PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(compare_shadow PRIVATE sgr_physics nlohmann_json::nlohmann_json)

add_executable(sweep_tool sweep_tool.cpp)
target_link_libraries(sweep_tool PRIVATE sgr_physics nlohmann_json::nlohmann_json)

add_executable(render_hero render_hero.cpp)
target_link_libraries(render_hero PRIVATE sgr_physics)

# Thesis appendix "Image handedness": before/after numbers from code.
add_executable(handedness_tool handedness_tool.cpp)
target_link_libraries(handedness_tool PRIVATE sgr_physics nlohmann_json::nlohmann_json)

# Gate 13: CPU twin of the live PHYSICAL preset for the colour comparison.
add_executable(render_disk_cpu render_disk_cpu.cpp)
target_include_directories(render_disk_cpu PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(render_disk_cpu PRIVATE sgr_physics)

# Thesis data tools (paper/THESIS_PLAN.md §2.1) — additive, read-only w.r.t.
# the validated physics core.
add_executable(physics_tables physics_tables.cpp)
target_link_libraries(physics_tables PRIVATE sgr_physics)

add_executable(convergence_tool convergence_tool.cpp)
target_link_libraries(convergence_tool PRIVATE sgr_physics)

# Gate 17 (extension paper): observer kinematics and the shadow each
# observer sees — the CPU reference for the live renderer's moving cameras.
add_executable(observer_tool observer_tool.cpp)
target_link_libraries(observer_tool PRIVATE sgr_physics nlohmann_json::nlohmann_json)

# Gate 24: falling into Sgr A* (worldline, tides, GPU reference strips)
add_executable(infall_tool infall_tool.cpp)
target_link_libraries(infall_tool PRIVATE sgr_physics nlohmann_json::nlohmann_json)

# Gates 13–15 (extension paper): the hypothetical thin disk at Sgr A*'s
# measured accretion rate, and visual vs bolometric Doppler contrast.
add_executable(spectral_tool spectral_tool.cpp)
target_link_libraries(spectral_tool PRIVATE sgr_physics nlohmann_json::nlohmann_json)
