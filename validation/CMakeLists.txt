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

# Thesis data tools (paper/THESIS_PLAN.md §2.1) — additive, read-only w.r.t.
# the validated physics core.
add_executable(physics_tables physics_tables.cpp)
target_link_libraries(physics_tables PRIVATE sgr_physics)

add_executable(convergence_tool convergence_tool.cpp)
target_link_libraries(convergence_tool PRIVATE sgr_physics)
