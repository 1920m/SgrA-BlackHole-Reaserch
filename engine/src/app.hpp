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
    bool menu = false;                // boot to the cinematic menu (Phase 9)
    int demo_frames = 0;              // >0: scripted menu→sim→menu transition
                                      // every N frames (Gate 9 evidence)
    std::string panel;                // open this menu panel at boot
                                      // (about|nohair|scope|gallery|settings)
    bool no_mouse = false;            // ignore mouse orbit/zoom (automation)
    float rot_speed = -1.0f;          // disk rotation ×real (-1: default 1.0)
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
    float turb = -1.0f;        // turbulence / cloud density override (0..1)
    bool black_sky = false;    // start with the starfield hidden
    bool show_stars = false;   // draw the S-stars into the live scene
    bool star_paths = true;    // draw their orbit ellipses too
    float star_scale = 1.0f;   // 1 = true distance; N = pulled in by N
    std::string focus_star;    // lock the camera to this star at boot
    bool focus_ride = false;   // ride it instead of orbiting it
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
    float turb;       // artistic turbulence amount 0..1 (0 = pure analytic)
    float cam_phi;    // camera azimuth [rad] (axisymmetric: rotates the disk)
    float time_scale; // filament advection: Ω [1/M] per wall-clock second
    uint32_t sky_off; // 1 = fully black background (starfield hidden)
    // Orthonormal camera basis in the local ZAMO triad, built on the CPU in
    // double precision (see CameraBasis.hpp). Default = look at the hole.
    float fwd_r, fwd_th, fwd_ph;
    float rgt_r, rgt_th, rgt_ph;
    float up_r, up_th, up_ph;
};
static_assert(sizeof(PushConstants) <= 128);

class App {
public:
    explicit App(const AppOptions& opts);
    ~App();

    // Runs the loop; returns the number of validation errors observed.
    int run();

    // UI mode (Phase 9): the menu shows the baked hero plate (no compute);
    // the live renderer only runs in Simulation mode.
    enum class UiMode { Menu, Simulation };
    UiMode mode() const { return mode_; }
    bool menu_enabled() const { return opts_.menu; }
    const std::string& focus_star() const { return opts_.focus_star; }
    const std::string& boot_panel() const { return opts_.panel; }
    void enter_simulation(bool science, bool scope = false);
    void return_to_menu() { mode_ = UiMode::Menu; }
    void request_quit();

    // Recreate the internal render target next frame (render_scale changed
    // via the HUD slider or the auto-resolution controller).
    void request_target_rebuild() { target_rebuild_ = true; }
    uint32_t target_width() const { return target_w_; }
    uint32_t target_height() const { return target_h_; }

    // Generic sampled/blit texture loaded from a PNG (LUTs, hero plate,
    // gallery thumbnails).
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        int w = 0, h = 0;
    };
    // Loads a PNG, uploads via staging, transitions to `final_layout`; the
    // texture is owned by the App (freed in ~App). Returns w=0 on failure.
    Texture create_texture_png(const std::string& path,
                               VkImageUsageFlags usage,
                               VkImageLayout final_layout);
    VkSampler lut_sampler() const { return lut_sampler_; }

private:
    void create_render_target();
    void destroy_render_target();
    void record(VkCommandBuffer cmd, uint32_t sc_index, float t);
    void record_menu(VkCommandBuffer cmd, uint32_t sc_index, float t);
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

    std::vector<Texture> owned_textures_;
    VkSampler lut_sampler_ = VK_NULL_HANDLE;
    size_t n_luts_ = 0;

    UiMode mode_ = UiMode::Simulation;
    Texture hero_{};
    bool esc_prev_ = false;
    bool dragging_ = false;
    double drag_x_ = 0.0, drag_y_ = 0.0;
    bool target_rebuild_ = false;
    bool autores_skip_ = false; // discard the fps sample after a rebuild

public:
    // Live-tunable state (edited by the HUD; consumed by make_push_constants).
    struct LiveParams {
        float a_star, incl_deg, fov, r_cam, disk_r_out, exposure_ev, t_peak_k;
        float turb = 0.0f;    // artistic turbulence layer (post-validation)
        float cam_phi = 0.0f; // orbit azimuth [rad], mouse-driven
        // Disk rotation in multiples of REAL Sgr A* wall-clock time
        // (1.0 = physical: GM/c³ = 21.17 s per geometric time unit, one
        // ISCO orbit ≈ 8.5 min — crank it up to actually see the swirl).
        float rot_speed = 1.0f;
        float render_scale = 1.0f; // internal render resolution multiplier
        bool auto_res = true;      // hold ~13 fps by adjusting render_scale
        uint32_t lut_slot = 0;
        int preset = 0;
        bool black_sky = false; // hide the starfield: fully black backdrop
        // S-stars drawn into the live scene (projected through the same
        // camera model the compute shader uses). The renderer works in
        // true geometric units, so at star_scale_ratio = 1 the stars sit
        // at their real distances (S2's orbit ~2.4e4 M against a ~20 M
        // disk) — fly the camera out to see them. Larger ratios pull the
        // orbits inward by that labelled factor for an overview; the
        // ratio is camera-INDEPENDENT so the cluster never rescales
        // under you while you zoom.
        bool show_stars = false;
        bool show_star_paths = true;
        float star_scale_ratio = 1.0f;
        // Camera basis in the local ZAMO triad, orthonormal, written by
        // SimHUD when a star is focused (CameraBasis.hpp). The default is
        // the historical aim straight at the hole.
        float fwd_r = -1.0f, fwd_th = 0.0f, fwd_ph = 0.0f;
        float rgt_r = 0.0f, rgt_th = 0.0f, rgt_ph = 1.0f;
        float up_r = 0.0f, up_th = -1.0f, up_ph = 0.0f;
        // 0 = Sgr A*, 1..N = cluster_[i-1]; focus_ride: look from the star
        // at the hole instead of orbiting the star.
        int focus_index = 0;
        bool focus_ride = false;
        float focus_dist = 40.0f;   // orbit-mode distance from the star [M]
        float focus_yaw = 0.0f, focus_pitch = 0.35f;
        bool science_mode = false;
        bool scope_mode = false; // photon-ring scope framing active
    } live_{};
    void apply_preset(int index);
};

} // namespace sgr
