#pragma once
// Engine application: compute → storage image → blit → present loop.

#include <array>
#include <memory>
#include <string>

#include "vk/context.hpp"
#include "vk/descriptors.hpp"
#include "vk/pipeline.hpp"

namespace sgr {

struct AppOptions {
    int width = 1280;
    int height = 720;
    bool validation = false;
    int max_frames = -1;              // -1: run until window closes
    std::string screenshot_path;      // save last frame if non-empty
    std::string shader = "solid";     // shaders/<name>.comp.glsl
    // Physics parameters forwarded to the shader (Phase 7+):
    float a_star = 0.94f;
    float incl_deg = 80.0f;
    float fov = 0.032f;
    float r_cam = 1000.0f;
    float disk_r_out = 18.0f;
    float exposure_ev = 0.0f;
    uint32_t shade_mode = 0; // 0 beauty, 1 diagnostic (CPU/GPU compare)
};

// Push-constant block shared with the shaders (must match GLSL layout).
struct PushConstants {
    uint32_t image_index;
    uint32_t width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    uint32_t shade_mode;
};
static_assert(sizeof(PushConstants) <= 128);

class App {
public:
    explicit App(const AppOptions& opts);
    ~App();

    // Runs the loop; returns the number of validation errors observed.
    int run();

private:
    void create_render_target();
    void destroy_render_target();
    void record(VkCommandBuffer cmd, uint32_t sc_index, float t);
    void save_screenshot();

    AppOptions opts_;
    std::unique_ptr<vk::Context> ctx_;
    std::unique_ptr<vk::BindlessTable> bindless_;
    std::unique_ptr<vk::ComputePipeline> pipeline_;

    VkImage target_ = VK_NULL_HANDLE;
    VmaAllocation target_alloc_ = nullptr;
    VkImageView target_view_ = VK_NULL_HANDLE;
    uint32_t target_w_ = 0, target_h_ = 0;
    bool target_initialised_ = false;
};

} // namespace sgr
