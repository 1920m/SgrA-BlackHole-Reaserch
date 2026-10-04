#pragma once
// Cinematic main menu (Phase 9): engraved serif title over the baked hero
// plate (Sgr A* at 230 GHz, rendered by this engine), square-bullet vertical
// menu, corner lore text, and the info panels (Settings / No-hair /
// Photon-ring scope / Gallery / About).
// Drawn with ImGui inside SimHUD's frame; App::record_menu blits the plate.

#include <string>
#include <vector>

struct ImFont;

namespace sgr {

class App;

class MenuMode {
public:
    // Fonts must be added to the ImGui atlas before the Vulkan backend builds
    // it — SimHUD calls this between CreateContext and ImGui_ImplVulkan_Init.
    void load_fonts();

    // Loads panel data (phase-3/5 JSON artefacts) and gallery thumbnails.
    // Called once after the ImGui Vulkan backend is initialised.
    void load_artifacts(App& app);

    // Build this frame's menu UI. Returns true if the SETTINGS panel is open
    // (SimHUD then draws the shared controls window on top).
    bool draw(App& app);

private:
    void draw_title(float x, float y);
    void draw_corners();
    void draw_items(App& app, float x, float y);
    void draw_about(App& app);
    void draw_nohair();
    void draw_scope_panel(App& app);
    void draw_gallery();

    ImFont* font_title_ = nullptr;
    ImFont* font_item_ = nullptr;
    ImFont* font_small_ = nullptr;

    int selected_ = 0;
    float sel_anim_ = 0.0f;   // animated highlight position (item index units)
    bool boot_panel_applied_ = false;
    bool show_settings_ = false;
    bool show_about_ = false;
    bool show_nohair_ = false;
    bool show_scope_ = false;
    bool show_gallery_ = false;

    // Phase 5 artefacts (no-hair panel)
    bool nohair_loaded_ = false;
    double eps_q_ci95_ = 0.0, eps_q_sigma_ = 0.0;
    float eps_q_slider_ = 0.0f;
    // Extension-paper forecasts (validation/extension/forecasts.json)
    bool forecast_loaded_ = false;
    double fc_ten_ = 0.0, fc_ten_naive_ = 0.0, fc_grav32_ = 0.0, fc_bias32_ = 0.0;

    // Phase 3 artefacts (photon-ring panel)
    bool subring_loaded_ = false;
    double ratio_n0_n1_ = 0.0, ratio_n1_n2_ = 0.0, beaming_ = 0.0;

    // Gallery thumbnails (ImGui texture handles; images owned by App)
    struct GalleryItem {
        std::string name;
        unsigned long long tex_id = 0;
        int w = 0, h = 0;
    };
    std::vector<GalleryItem> gallery_;
};

} // namespace sgr
