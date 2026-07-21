#pragma once
// ImGui HUD (Phase 8): render pass + framebuffers for the overlay (classic
// render pass — dynamic rendering unavailable on this hardware, see
// engine/CMakeLists.txt), parameter sliders, preset switching, and the
// Science Mode window (S2 orbit view + fit numbers loaded from the Phase 4
// artefacts when present).

#include <volk.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CameraBasis.hpp"
#include "orbit_fitter/SkyProjection.hpp" // S2Model (Ceres-free GR orbit model)

struct ImDrawList;

namespace sgr {

class App;
class MenuMode;
namespace vk { class Context; }

class SimHUD {
public:
    SimHUD(vk::Context& ctx, App& app);
    ~SimHUD();
    SimHUD(const SimHUD&) = delete;
    SimHUD& operator=(const SimHUD&) = delete;

    // Rebuild framebuffers after swapchain recreation.
    void on_swapchain_recreated();

    // Build this frame's UI (call between glfwPollEvents and record).
    void new_frame(double fps);

    // True when ImGui owns the mouse (hovering/dragging a window) — the
    // camera orbit/zoom controls stand down then.
    bool wants_mouse() const;

    // Record the overlay pass over the given swapchain image
    // (expects TRANSFER_DST_OPTIMAL; leaves PRESENT_SRC_KHR).
    void record(VkCommandBuffer cmd, uint32_t sc_index);

private:
    void create_framebuffers();
    void destroy_framebuffers();
    void load_science_artifacts();
    void load_cluster_orbits();
    void load_stellar_parameters();
    void rescale_cluster(float ratio);   // re-integrate at 1:N (real shrink)
    void update_cluster_time();
    void draw_controls(double fps);
    void draw_science();
    void draw_cluster3d();
    void draw_stars_overlay();
    void draw_focus_list();
    void update_focus_camera();
    void spawn_debris(size_t star_index);
    void draw_debris(ImDrawList* dl, double k, float W, float H);
    // BH-frame position of a focus target: 0 = Sgr A*, i>0 = cluster_[i-1],
    // already multiplied by the display scale k.
    Vec3 focus_target(int index, double k) const;
    void draw_scope_overlay();

    vk::Context* ctx_;
    App* app_;
    std::unique_ptr<MenuMode> menu_; // Phase 9 (only when --menu)
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkImageView> sc_views_;

    // Science Mode data (loaded once if the Phase 4 outputs exist)
    struct Track { std::vector<float> t, ra, dec, vr; };
    Track model_;
    struct Points { std::vector<float> t, ra, dec, sra, sdec, vr, svr; };
    Points ast_;
    std::string fit_summary_;
    bool science_available_ = false;

    // 3-D S-star cluster viewer: the Phase-13 joint-fit elements
    // re-integrated through the SAME full-GR geodesic model the fitter
    // uses (sgr_s2_model); the viewer animates those cached trajectories.
    struct ClusterStar {
        std::string name;
        S2Model model{0.0, 0.0};
        S2Parameters base{};             // fitted elements at scale 1
        double period_yr = 0.0;          // at the CURRENT scale
        double base_period_yr = 0.0;     // fitted, at scale 1
        double t0 = 1992.0, t1 = 2152.0; // integrated epoch span [yr]
        double mas_per_M = 1.0;          // BH-frame [M] -> sky [mas]
        double a_M = 0.0;                // semi-major axis in geometric M
        double peri_M = 0.0;             // pericentre at the CURRENT scale
        // Habibi+2017 spectroscopy (measured) and what follows from it.
        double mass_msun = 0.0, teff_k = 0.0, logg = 0.0;
        bool assumed = false;            // not in the paper's sample
        double radius_M = 0.0;           // derived: sqrt(G M*/g), in units of M
        double r_tidal_M = 0.0;          // eta * R* * (M_BH/M*)^(1/3)
        bool has_params = false;
        // disruption state machine (display-layer what-if)
        enum class Phase { Intact, Stretching, Disrupted };
        Phase phase = Phase::Intact;
        bool valid = true;   // false = orbit could not be integrated at this
                             // scale (NOT a disruption claim)
        double stretch = 0.0;            // 0..1 tidal elongation cue
        double disrupt_epoch = 0.0;
        uint32_t line_col = 0xffffffff;
        // Trajectory samples in BH-frame geometric units [M] — the same
        // units the renderer's camera works in, so the in-scene overlay
        // and the 3-D window share one source.
        std::vector<std::array<float, 3>> track;  // one dense orbit
        std::vector<std::array<float, 3>> trail;  // recent positions
        double trail_last_t = -1.0;
    };
    // Tidal debris: bound fragments after a disruption, each an orbit in the
    // same validated S2Model with the frozen-in energy spread applied
    // (Rees 1988). Display-layer what-if, like everything scale-related.
    struct DebrisParticle {
        S2Model model{0.0, 0.0};
        double t0 = 0.0, t1 = 0.0;
        uint32_t col = 0xffffffff;
        bool ok = false;
    };
    std::vector<DebrisParticle> debris_;

    CameraFrame cam_frame_{};   // this frame's camera, shared with the shader
    std::vector<ClusterStar> cluster_;
    bool cluster_available_ = false;
    double clu_rmax_M_ = 0.0;   // largest apoapsis in the set [true M]
    double clu_scale_applied_ = 1.0;  // ratio the cached orbits were built at
    float clu_eta_ = 1.0f;      // 1 = tidal radius, 2.44 = fluid Roche limit
    bool clu_disruption_ = true;      // allow stars to break up
    double clu_rg_m_ = 0.0;     // metres per M for the fitted mass
    double clu_M_ = 0.0;          // joint-fit mass [Msun] (caption)
    double clu_epoch_ = 2026.5;   // displayed epoch [Julian yr]
    double clu_end_ = 2150.0;     // common end of the integrated spans
    float clu_speed_ = 3.0e6f;    // wall-clock multiplier (~0.1 yr/s)
    bool clu_play_ = true;
    float clu_yaw_ = 0.0f, clu_pitch_ = 0.0f, clu_zoom_ = 1.0f;
};

} // namespace sgr
