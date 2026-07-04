#pragma once
// Engine application: compute → storage image → blit → present loop.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "presets/DiskPreset.hpp"
#include "vk/context.hpp"
#include "vk/descriptors.hpp"
#include "vk/pipeline.hpp"

namespace sgr {

class SimHUD;

struct AppOptions {
    int width = 1280;
    int height = 720;
    bool validation = false;
    int max_frames = -1;              // -1: run until window closes
    std::string screenshot_path;      // save last frame if non-empty
    std::string shader = "solid";     // shaders/<name>.comp.glsl
    // Physics parameters forwarded to the shader (Phase 7+). Negative /
    // sentinel values mean "not set on the CLI": the preset (live mode) or
    // the per-shader defaults fill them in.
    float a_star = -1.0f;
    float incl_deg = -1.0f;
    float fov = -1.0f;
    float r_cam = -1.0f;
    float disk_r_out = -1.0f;
    float exposure_ev = -999.0f;
    uint32_t shade_mode = 0; // 0 beauty, 1 diagnostic (CPU/GPU compare)
    bool offscreen = false;  // render target exactly width×height in TDR-safe
                             // row bands, save screenshot, exit (no present)
    bool hud = false;        // ImGui HUD + preset system (Phase 8 live mode)
    bool science = false;    // open the Science Mode window at startup
    int preset = 0;          // initial preset index (kDiskPresets)
    float t_peak_k = -1.0f;
    float render_scale = 1.0f; // live render target scale (perf knob)
};

// Push-constant block shared with the shaders (must match GLSL layout).
struct PushConstants {
    uint32_t image_index;
    uint32_t width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    uint32_t shade_mode;
    uint32_t y_offset;
    float flux_max;   // Page-Thorne max flux (CPU precomputed)
    float t_peak_k;   // peak disk temperature [K]
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
    int run_offscreen();
    PushConstants make_push_constants(float t) const;
    void load_luts();

    AppOptions opts_;
    std::unique_ptr<vk::Context> ctx_;
    std::unique_ptr<vk::BindlessTable> bindless_;
    std::unique_ptr<vk::ComputePipeline> pipeline_;
    std::unique_ptr<SimHUD> hud_;

    VkImage target_ = VK_NULL_HANDLE;
    VmaAllocation target_alloc_ = nullptr;
    VkImageView target_view_ = VK_NULL_HANDLE;
    uint32_t target_w_ = 0, target_h_ = 0;
    bool target_initialised_ = false;
    uint32_t last_sc_index_ = 0;

    struct LutTexture {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::vector<LutTexture> luts_;
    VkSampler lut_sampler_ = VK_NULL_HANDLE;

public:
    // Live-tunable state (edited by the HUD; consumed by make_push_constants).
    struct LiveParams {
        float a_star, incl_deg, fov, r_cam, disk_r_out, exposure_ev, t_peak_k;
        uint32_t lut_slot = 0;
        int preset = 0;
        bool science_mode = false;
    } live_{};
    void apply_preset(int index);
};

} // namespace sgr
