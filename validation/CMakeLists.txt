# Validation tools: reference renders + CPU/GPU comparison (Phase 7)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(render_reference render_reference.cpp)
target_link_libraries(render_reference PRIVATE sgr_physics nlohmann_json::nlohmann_json)
