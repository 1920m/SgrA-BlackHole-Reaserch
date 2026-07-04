#pragma once
// ImGui HUD (Phase 8): render pass + framebuffers for the overlay (classic
// render pass — dynamic rendering unavailable on this hardware, see
// engine/CMakeLists.txt), parameter sliders, preset switching, and the
// Science Mode window (S2 orbit view + fit numbers loaded from the Phase 4
// artefacts when present).

#include <volk.h>

#include <memory>
#include <string>
#include <vector>

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
    void draw_controls(double fps);
    void draw_science();
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
};

} // namespace sgr
