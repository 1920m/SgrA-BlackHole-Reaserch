#pragma once
// Engine application: compute → storage image → blit → present loop.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "presets/DiskPreset.hpp"
#include "sgr_physics/infall.hpp"
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
    bool bolometric = false;   // grey (gT)^4 disk display (teaches g^4 beaming)
    bool grid = false;         // 10° celestial grid on the sky
    std::string display;       // linear|agx|aces ("" = agx; science → linear)
    float sky_ev = -999.0f;    // sky brightness override [EV]
    float aim_x = 0.0f, aim_y = 0.0f; // put the hole at this NDC point (composition)
    int observer = -1;          // Observer enum (-1: ZAMO default)
    bool ergo = false;          // ergosphere overlay
    bool procedural_sky = false; // ignore the Milky Way map
    bool uniform_sky = false;  // validation: sky radiance 1 everywhere (masks)
    bool no_disk = false;      // hide the thin disk (lensed sky only)
    int spp = 1;               // offscreen: progressive samples per pixel
    float bloom = -1.0f;       // bloom strength override (0 = off)
    bool show_stars = false;   // draw the S-stars into the live scene
    bool star_paths = true;    // draw their orbit ellipses too
    float star_scale = 1.0f;   // 1 = true distance; N = pulled in by N
    std::string focus_star;    // lock the camera to this star at boot
    bool focus_ride = false;   // ride it instead of orbiting it
    double epoch_yr = -1.0;    // freeze the S-star clock at this epoch
    // Gate 18: orbiting hot spot (GRAVITY 2018/2020 flare model) and movies
    bool flare = false;        // hot spot on
    float flare_r = -1.0f;     // orbit radius [M] (default 7: GRAVITY 2018)
    float flare_t = 0.0f;      // observing time [M of camera KS time]
    bool flare_gray = false;   // validation: spot drawn as grey intensity
    float flare_amp = -1.0f;   // display brightness of the spot
    int movie_frames = 0;      // offscreen: render this many frames ...
    float movie_dt = 2.0f;     // ... this many M of observing time apart
    int max_steps = 6000;      // live ray step budget (validation of G17)
    std::string band;          // visible | 230 | kband (Gate 19 multi-band views)
    bool wormhole = false;     // Gate 21 exotic demo (NOT Sgr A*)
    float wh_l = -1000.0f;     // camera position ℓ [throat radii] (unset)
    // Gate 24: fall into Sgr A*
    bool flight = false;       // start in the in-fall mode
    int flight_frames = 0;     // >0: record the fall as N frames (offscreen)
    float flight_r = -1.0f;    // >0: one frame where the faller passes r [M]
    float flight_r0 = 30.0f;   // start radius [M]
    float flight_theta = 60.0f;   // start polar angle [deg]
    bool flight_rest = false;  // start at rest (ZAMO) instead of rain (E = 1)
    bool no_sky_shift = false; // validation masks: sky radiance not shifted by g
    std::string look;          // ahead | back | side: the camera's look basis
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
    // bit0: bolometric display — the disk shows grey (gT/6500 K)^4 instead
    // of its visual (Planck x CIE) colour, isolating the g^4 beaming.
    // bit1: 10° celestial coordinate grid on the sky.
    // bit2: Milky Way map loaded and in use. bit3: uniform sky (validation).
    uint32_t display_flags;
    uint32_t accum_index;  // progressive sample index (0 = restart)
    float accum_weight;    // weight of this sample in the running mean
    // Linear gain on the background sky radiance (display: the Milky Way
    // map has no physical flux scale relative to the disk). Includes the
    // camera meter so the sky looks the same at any disk temperature.
    float sky_gain;
};
static_assert(sizeof(PushConstants) <= 128);

// Post-processing push constants (Gate 14; must match the GLSL blocks).
struct BloomDownPC {
    uint32_t src_slot, dst_slot;
    uint32_t src_w, src_h, dst_w, dst_h;
    uint32_t src_is_hdr;
};
struct BloomUpPC {
    uint32_t src_slot, dst_slot;
    uint32_t src_w, src_h, dst_w, dst_h;
    float radius;
};
struct TonemapPC {
    uint32_t hdr_slot, bloom_slot, ldr_slot;
    uint32_t w, h, bloom_w, bloom_h;
    uint32_t op;           // 0 linear/science, 1 AgX, 2 ACES
    float exposure;        // 2^EV
    float bloom_strength;
    float bloom_norm;      // 1 / bloom levels summed
    float saturation;
    float wb_r, wb_g, wb_b;
    uint32_t y_offset;
};

// Per-frame camera/observer parameters (bindless binding 2, std430; must
// match the GLSL block in disk.comp.glsl). Gate 17: the observer's tetrad is
// built on the CPU in double precision — any 4-velocity (static ZAMO,
// orbiting, free-falling, hovering; later the G24 in-fall) — and handed to
// the shader in Kerr-Schild components, so the shader only lowers the
// received photon's momentum with the KS metric.
struct FrameParams {
    float cam_x[4];  // camera event, KS coordinates (t, x, y, z)
    float e0[4];     // observer 4-velocity, KS contravariant
    float e_fwd[4];  // spatial leg: look direction
    float e_rgt[4];  // spatial leg: image right
    float e_up[4];   // spatial leg: image up
    float ks_chart;  // −1 outgoing (exterior cameras), +1 ingoing
    uint32_t flags;  // bit0: Doppler/gravitational shift of the sky
                     // bit1: ergoregion overlay, bit2: hot spot
    uint32_t max_steps; // ray step budget (6000; G17 validation can lower it)
    float pad1;
    // Gate 18 hot spot (science/flares/Hotspot.hpp, the fp64 CPU twin):
    float spot0[4];  // R, s, phase0, display amplitude
    float spot1[4];  // Omega, u^t, t_obs (camera KS-time offset), alpha
    float spot_h0[4]; // comoving metric at the KS-azimuth-0 centre:
    float spot_h1[4]; //   (h00, h01, h02, h11), (h12, h22, x0, y0)
    // Gate 19 hot accretion flow (flags bit3; science/riaf/Riaf.hpp twin):
    float riaf0[4];  // n0 [cm^-3], T0 [K], beta, nu_obs [GHz]
    float riaf1[4];  // E_isco, L_isco, r_isco, r_max
    float riaf2[4];  // r_g [cm], unused ×3
    // Gate 21 EXOTIC (flags bit4) — the Dneg wormhole, NOT Sgr A*:
    float wh0[4];    // throat ρ, half-length a, lensing M, camera ℓ
};
static_assert(sizeof(FrameParams) == 224);

// Observer whose eyes the camera uses (Gate 17).
enum class Observer : int {
    Zamo = 0,     // zero angular momentum ("locally non-rotating")
    Orbiting = 1, // circulating at the prograde Keplerian Ω(r)
    Infall = 2,   // free fall from rest at infinity (E = 1, L = Q = 0)
    Hovering = 3, // static w.r.t. the distant stars (outside the ergosphere)
    RideStar = 4, // moving with the focused S-star (its full-GR orbit velocity)
};

// Kinematics of the current observer, for the HUD.
struct ObserverState {
    bool valid = true;      // false: the chosen observer cannot exist here
    double alpha = 1.0;     // lapse of the ZAMO at the camera
    double omega = 0.0;     // frame-dragging angular velocity ω [1/M]
    double v = 0.0;         // speed relative to the local ZAMO [c]
    double gamma = 1.0;     // Lorentz factor relative to the ZAMO
    double dtau_dt = 1.0;   // observer proper time per far-away time, 1/u^t
    double r_ergo = 2.0;    // ergosphere radius at the camera's θ [M]
};

// Storage-image slot map (bindless binding 0).
inline constexpr uint32_t kSlotLdr = 0;        // rgba8 display image (blit/screenshot)
inline constexpr uint32_t kSlotHdr = 1;        // rgba32f linear radiance accumulator
inline constexpr uint32_t kSlotBloom0 = 2;     // rgba16f bloom pyramid, slots 2..
inline constexpr uint32_t kMaxBloomLevels = 6;

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
    double boot_epoch() const { return opts_.epoch_yr; }
    const std::string& boot_panel() const { return opts_.panel; }
    void enter_simulation(bool science, bool scope = false);
    void enter_exotic();   // Gate 21 wormhole demo (labelled: NOT Sgr A*)
    void enter_flight();   // Gate 24: fall into Sgr A*
    void start_flight();   // (re)compute the worldline from live_ settings
    const InfallTrajectory* flight_trajectory() const { return flight_.get(); }
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
    // Same, from w×h RGBA float texels (R32G32B32A32_SFLOAT).
    Texture create_texture_rgba32f(const std::vector<float>& rgba, int w, int h,
                                   VkImageUsageFlags usage,
                                   VkImageLayout final_layout);
    VkSampler lut_sampler() const { return lut_sampler_; }
    // Disk peak temperature actually rendered: computed from the accretion
    // rate (live_.mdot_msun_yr) unless overridden (live_.t_peak_from_mdot).
    float effective_t_peak() const;
    // Samples currently averaged in the radiance accumulator (HUD readout).
    uint32_t accumulated_samples() const { return accum_n_; }
    // Kinematics of the current observer at the camera (HUD readout).
    ObserverState observer_state() const {
        ObserverState s;
        make_frame_params(&s);
        return s;
    }

private:
    Texture upload_texture(const void* texels, int w, int h, VkFormat format,
                           size_t bytes_per_texel, VkImageUsageFlags usage,
                           VkImageLayout final_layout);
    // Gate 14 post chain (live disk shader only).
    struct StorageImage {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0;
    };
    StorageImage create_storage_image(uint32_t w, uint32_t h, VkFormat format,
                                      uint32_t slot);
    void destroy_storage_image(StorageImage& img);
    bool post_enabled() const { return static_cast<bool>(tonemap_); }
    // Ray-trace dispatch over rows [rows_from, rows_from + rows).
    void record_trace(VkCommandBuffer cmd, PushConstants pc,
                      uint32_t rows_from, uint32_t rows);
    // First use: accumulator + bloom pyramid UNDEFINED → GENERAL; afterwards
    // a memory barrier against the previous frame's accesses.
    void prepare_post_images(VkCommandBuffer cmd);
    // Bloom pyramid (skipped when off / science) + tone map into target_.
    void record_post(VkCommandBuffer cmd);
    TonemapPC make_tonemap_pc() const;
    // Camera meter: luminance mapped to 1.0 at EV 0 (1 in linear/science).
    double display_meter() const;
    // Gate 15: NASA SVS Deep Star Maps galactic sky (scripts/prepare_sky.py).
    void load_sky();
    // Progressive-sampling bookkeeping: restart when the scene changes.
    void update_accumulation(PushConstants& pc, const FrameParams& fp);
    // Gate 17: build the observer tetrad and upload it (vkCmdUpdateBuffer).
    FrameParams make_frame_params(ObserverState* state = nullptr) const;
    void record_frame_params(VkCommandBuffer cmd, const FrameParams& fp);
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
    std::unique_ptr<InfallTrajectory> flight_;   // Gate 24 worldline (fp64)

    VkImage target_ = VK_NULL_HANDLE;
    VmaAllocation target_alloc_ = nullptr;
    VkImageView target_view_ = VK_NULL_HANDLE;
    uint32_t target_w_ = 0, target_h_ = 0;
    bool target_initialised_ = false;

    std::unique_ptr<vk::ComputePipeline> bloom_down_, bloom_up_, tonemap_;
    VkBuffer frame_buf_ = VK_NULL_HANDLE;   // FrameParams SSBO (binding 2)
    VmaAllocation frame_buf_alloc_ = nullptr;
    StorageImage hdr_;
    std::vector<StorageImage> bloom_;
    bool post_initialised_ = false;     // post images still in UNDEFINED
    uint32_t accum_n_ = 0;              // samples in the accumulator
    std::vector<unsigned char> accum_sig_; // scene signature of those samples
    uint32_t last_sc_index_ = 0;

    std::vector<Texture> owned_textures_;
    VkSampler lut_sampler_ = VK_NULL_HANDLE;
    VkSampler nearest_sampler_ = VK_NULL_HANDLE; // texelFetch-only tables
    VkSampler sky_sampler_ = VK_NULL_HANDLE;     // trilinear, wraps in longitude
    bool sky_loaded_ = false;
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
        uint32_t style = 0; // display style bits (DiskPreset::style)
        int preset = 0;
        bool black_sky = false; // hide the starfield: fully black backdrop
        bool bolometric = false; // grey (gT)^4 disk instead of visual colour
        bool show_grid = false;  // 10° celestial grid (lensing teaching aid)
        float sky_ev = 0.0f;     // Milky Way brightness [EV] (display)
        bool use_sky_map = true; // false: procedural point stars
        bool uniform_sky = false; // validation masks: sky = 1 everywhere
        bool show_disk = true;    // the thin disk (off: lensing of the sky only)
        // Gate 17 — whose eyes: the observer's motion sets aberration and
        // the Doppler/gravitational shift of everything it sees.
        int observer = 0;        // Observer enum
        // Observer::RideStar: the focused star's coordinate velocity dx/dt
        // [c] in the camera's local spherical directions (r, theta, phi),
        // written by SimHUD each frame while a star is ridden.
        float ride_v[3] = {0.0f, 0.0f, 0.0f};
        bool ride_v_valid = false;
        bool sky_shift = true;   // shift the sky's colour/brightness by g
        bool show_ergo = false;  // translucent ergosphere surface overlay
        // Gate 18 — a hot spot on a circular geodesic, the model GRAVITY
        // fitted to the 2018 near-infrared flares: Gaussian emissivity of
        // size s (0.5 M, GRAVITY 2020), observed intensity ∫ g^{2+α} ε dλ
        // with slow light (each ray carries its own travel time).
        bool flare = false;
        float flare_R = 7.0f;      // orbit radius [M] (GRAVITY 2018 best fit)
        float flare_sigma = 0.5f;  // Gaussian size [M]
        float flare_alpha = 0.0f;  // spectral index, j_nu ~ nu^-alpha
        float flare_amp = 1.0f;    // display brightness (false colour)
        float flare_t = 0.0f;      // observing time [M]
        float flare_speed = 60.0f; // observing-time M per wall-clock second
        bool flare_play = true;
        bool flare_gray = false;   // grey intensity (validation)
        // Gate 19 — which band the camera sees: 0 visible (hypothetical thin
        // disk, Milky Way), 1 = 230 GHz (the hot accretion flow, EHT band),
        // 2 = 2.2 µm (hot-spot flares; no quiescent model, no visible sky).
        int band = 0;
        double riaf_n0 = 1.0e7;   // cm^-3; from validation/extension/riaf.json
        bool riaf_fitted = false; // n0 normalised to the 2.4 Jy benchmark?
        float riaf_T0 = 1.5e11f;  // K (Broderick & Loeb 2006, a* = 0.998 model)
        float riaf_beta = 10.0f;  // Broderick et al. 2011
        float riaf_nu_ghz = 230.0f;
        // Gate 21 — EXOTIC SPACETIME, NOT SGR A*: the Interstellar (Dneg)
        // wormhole of James et al. 2015, in units of the throat radius ρ.
        bool wormhole = false;
        float wh_l = 6.0f;      // camera position ℓ (negative: the other side)
        float wh_a = 0.005f;    // half-length of the cylinder (2a = 0.01ρ)
        float wh_W = 0.05f;     // lensing width W = 1.42953 M
        // Gate 24 — the fall into Sgr A* (ingoing Kerr–Schild worldline)
        bool flight = false;
        double flight_tau = 0.0;     // proper time since the start [M]
        bool flight_play = false;
        float flight_warp = 2.0f;    // proper time [M] per wall-clock second
        float flight_r0 = 30.0f;     // start radius [M]
        float flight_theta = 60.0f;  // start polar angle [deg]
        bool flight_rest = false;    // rain (E = 1) unless set
        // Display transform (Gate 14) — never touches the radiance image.
        int tonemap = 1;            // 0 linear/science, 1 AgX, 2 ACES
        float bloom = 0.04f;        // glare strength (0 = off)
        float wb_kelvin = 6500.0f;  // camera white balance (blackbody white)
        float saturation = 1.0f;
        bool accumulate = true;     // progressive AA while the view is static
        // Disk temperature from physics: T_peak = max T_eff of a Novikov-
        // Thorne disk fed at mdot (nt_peak_temperature_k, Sgr A* mass).
        float mdot_msun_yr = 2.0e-8f;
        bool t_peak_from_mdot = true; // false: t_peak_k is a free override
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
