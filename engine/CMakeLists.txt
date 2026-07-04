# Vulkan engine (Phases 6–9).
#
# HARDWARE NOTE (documented deviation from the Vulkan 1.3 design target): this machine's GPUs
# (GTX 1050 @ driver 461-era, Intel UHD 620) expose Vulkan 1.2 only and lack
# VK_KHR_dynamic_rendering on both adapters; synchronization2 is missing on the
# NVIDIA. The engine therefore targets Vulkan 1.2 with sync1 barriers behind
# sync2-shaped helpers and uses a single classic render pass exclusively for
# the ImGui overlay. Bindless descriptors (update-after-bind + partially
# bound) are fully supported and used from day one. All Phase 6-10 gate
# criteria are unaffected.
find_package(volk CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Stb REQUIRED)

# The Vulkan backend is compiled into this target (vendored from the exact
# imgui source vcpkg built, see third_party/) so it shares our VK_NO_PROTOTYPES
# + volk dispatch; the port's prebuilt backend object crashes in Init on this
# setup (its member object is then simply never pulled from imgui.lib).
add_executable(sgr_engine
    src/main.cpp
    src/app.cpp
    src/vk/context.cpp
    src/vk/pipeline.cpp
    src/vk/descriptors.cpp
    src/ui/SimHUD.cpp
    src/modes/MenuMode.cpp
    third_party/imgui_impl_vulkan.cpp
)
target_include_directories(sgr_engine PRIVATE src ${Stb_INCLUDE_DIR})
target_compile_definitions(sgr_engine PRIVATE
    VK_NO_PROTOTYPES
    GLFW_INCLUDE_NONE
    SGR_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    SGR_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)
target_link_libraries(sgr_engine PRIVATE
    sgr_physics
    volk::volk
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glfw
    unofficial::shaderc::shaderc
    imgui::imgui
)
