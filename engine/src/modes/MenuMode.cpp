#include "modes/MenuMode.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "app.hpp"

namespace sgr {

namespace {

// Scan a numeric value out of a JSON file after `"key":` — same minimal
// text-scan the HUD uses (no JSON dependency in the engine). Starts the digit
// search after the colon so digits inside key names are never picked up.
double grab_json(const std::string& j, const char* key, double fallback) {
    const auto k = j.find(std::string("\"") + key + "\":");
    if (k == std::string::npos) return fallback;
    const auto colon = j.find(':', k);
    const auto s = j.find_first_of("0123456789.-", colon);
    if (s == std::string::npos) return fallback;
    const auto e = j.find_first_not_of("0123456789.-e+", s);
    return std::stod(j.substr(s, e - s));
}

std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Wide-tracked engraved text: per-glyph advance with extra tracking, dark
// drop shadow under an ivory face (the "engraved serif" look of the design
// reference).
void tracked_text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos,
                  ImU32 col, const char* s, float tracking,
                  bool shadow = true) {
    float x = pos.x;
    char buf[2] = {0, 0};
    for (const char* p = s; *p; ++p) {
        buf[0] = *p;
        if (shadow)
            dl->AddText(font, size, {x + 1.5f, pos.y + 2.0f},
                        IM_COL32(0, 0, 0, 170), buf, buf + 1);
        dl->AddText(font, size, {x, pos.y}, col, buf, buf + 1);
        const ImVec2 adv =
            font ? font->CalcTextSizeA(size, FLT_MAX, 0.0f, buf, buf + 1)
                 : ImGui::CalcTextSize(buf);
        x += adv.x + tracking;
    }
}

float tracked_width(ImFont* font, float size, const char* s, float tracking) {
    float w = 0.0f;
    char buf[2] = {0, 0};
    for (const char* p = s; *p; ++p) {
        buf[0] = *p;
        w += font->CalcTextSizeA(size, FLT_MAX, 0.0f, buf, buf + 1).x + tracking;
    }
    return w;
}

constexpr ImU32 kIvory = IM_COL32(232, 229, 220, 255);
constexpr ImU32 kIvoryDim = IM_COL32(200, 197, 188, 165);
constexpr ImU32 kLore = IM_COL32(190, 188, 182, 110);

struct MenuItem {
    const char* label;
};
constexpr MenuItem kItems[] = {
    {"ENTER SIMULATION"}, {"SCIENCE MODE"},   {"NO-HAIR TEST"},
    {"PHOTON RING SCOPE"}, {"FALL INTO SGR A*"}, {"SETTINGS"},
    {"GALLERY"},           {"EXOTIC (NOT SGR A*)"}, {"ABOUT"},
    {"EXIT"},
};
constexpr int kItemCount = static_cast<int>(sizeof(kItems) / sizeof(kItems[0]));

} // namespace

void MenuMode::load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    // The first font in the atlas becomes ImGui's default — keep that the
    // built-in one so the HUD windows are unaffected by the menu serifs.
    io.FontDefault = io.Fonts->AddFontDefault();
    const char* serif = "C:\\Windows\\Fonts\\georgia.ttf";
    font_title_ = io.Fonts->AddFontFromFileTTF(serif, 64.0f);
    font_item_ = io.Fonts->AddFontFromFileTTF(serif, 22.0f);
    font_small_ = io.Fonts->AddFontFromFileTTF(serif, 13.0f);
    if (!font_title_)
        std::fprintf(stderr, "[menu] georgia.ttf unavailable — default font\n");
}

void MenuMode::load_artifacts(App& app) {
    namespace fs = std::filesystem;
    const std::string j5 = slurp("validation/phase5/s2_fit.json");
    if (!j5.empty()) {
        eps_q_ci95_ = grab_json(j5, "eps_q_ci95_halfwidth", 0.0);
        eps_q_sigma_ = grab_json(j5, "eps_q_sigma", 0.0);
        forecast_sigma_ = grab_json(j5, "forecast_sigma_10_more_orbits", 0.0);
        nohair_loaded_ = eps_q_ci95_ > 0.0;
    }
    const std::string j3 = slurp("validation/phase3/subring_report.json");
    if (!j3.empty()) {
        ratio_n0_n1_ = grab_json(j3, "flux_ratio_n0_n1", 0.0);
        ratio_n1_n2_ = grab_json(j3, "flux_ratio_n1_n2", 0.0);
        beaming_ = grab_json(j3, "beaming_flux_ratio", 0.0);
        subring_loaded_ = ratio_n0_n1_ > 0.0;
    }

    // Gallery thumbnails: milestone renders from the validation pipeline.
    const std::pair<const char*, const char*> shots[] = {
        {"Schwarzschild shadow (Gate 1)",
         "validation/reference_images/schwarzschild_cpu.png"},
        {"Kerr a*=0.94 shadow (Gate 2)",
         "validation/reference_images/kerr_cpu_a94.png"},
        {"Novikov-Thorne disk (Gate 3)", "validation/phase3/disk_nt.png"},
        {"Sub-ring n=1 (Gate 3)", "validation/phase3/subring_n1.png"},
        {"FIRESTORM preset (Gate 8)",
         "writeup/screenshots/phase_8/preset_firestorm.png"},
        {"GARGANTUA preset (Gate 8)",
         "writeup/screenshots/phase_8/preset_gargantua.png"},
        {"ARCTIC preset (Gate 8)",
         "writeup/screenshots/phase_8/preset_arctic.png"},
        {"NEBULA preset (Gate 8)",
         "writeup/screenshots/phase_8/preset_nebula.png"},
    };
    for (const auto& [name, path] : shots) {
        if (!fs::exists(path)) continue;
        const App::Texture tex = app.create_texture_png(
            path, VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (tex.w == 0) continue;
        const VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(
            app.lut_sampler(), tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        gallery_.push_back({name, reinterpret_cast<unsigned long long>(ds),
                            tex.w, tex.h});
    }
}

void MenuMode::draw_title(float x, float y) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    // Thin rule above the title, like the reference.
    dl->AddLine({x + 2, y - 14}, {x + 46, y - 14}, kIvoryDim, 1.0f);
    tracked_text(dl, font_title_, 64.0f, {x, y}, kIvory, "SGR A*", 14.0f);
    tracked_text(dl, font_small_, 13.0f, {x + 2, y + 78}, kIvoryDim,
                 "A PHYSICALLY RIGOROUS SIMULATION OF THE GALACTIC CENTRE",
                 2.2f, false);
}

void MenuMode::draw_corners() {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    tracked_text(dl, font_small_, 12.0f, {24, 14}, kLore,
                 "+  THE CENTRE OF THE GALAXY NEVER CAN REST", 1.6f, false);
    const char* tr = "MMXXVI";
    tracked_text(dl, font_small_, 12.0f,
                 {ds.x - 24 - tracked_width(font_small_, 12.0f, tr, 1.6f), 14},
                 kLore, tr, 1.6f, false);
    tracked_text(dl, font_small_, 12.0f, {24, ds.y - 26}, kLore,
                 "+  KERR SPACETIME  .  CARTER GEODESICS  .  GILLESSEN+2017",
                 1.6f, false);
    const char* br = "EVERY NUMBER REPRODUCIBLE FROM CODE";
    tracked_text(dl, font_small_, 12.0f,
                 {ds.x - 24 - tracked_width(font_small_, 12.0f, br, 1.6f),
                  ds.y - 26},
                 kLore, br, 1.6f, false);
}

void MenuMode::draw_items(App& app, float x, float y) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    const float row_h = 46.0f;
    const float item_size = 20.0f;

    // Keyboard navigation (skipped while a panel widget owns the keyboard).
    bool activate = false;
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
            selected_ = (selected_ + 1) % kItemCount;
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
            selected_ = (selected_ + kItemCount - 1) % kItemCount;
        activate = ImGui::IsKeyPressed(ImGuiKey_Enter) ||
                   ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);
    }

    // Mouse hover / click (only when no ImGui window wants the mouse).
    if (!io.WantCaptureMouse) {
        for (int i = 0; i < kItemCount; ++i) {
            const float ry = y + i * row_h;
            if (io.MousePos.x >= x - 8 && io.MousePos.x <= x + 380 &&
                io.MousePos.y >= ry - 6 && io.MousePos.y <= ry + item_size + 10) {
                selected_ = i;
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) activate = true;
            }
        }
    }

    // Animated highlight position.
    sel_anim_ += (selected_ - sel_anim_) *
                 std::min(1.0f, io.DeltaTime * 12.0f);

    for (int i = 0; i < kItemCount; ++i) {
        const float ry = y + i * row_h;
        const float focus = std::max(0.0f, 1.0f - std::fabs(sel_anim_ - i));
        const float bx = x + 6.0f * focus; // slight slide on selection
        // Square bullet: outline; fills as the highlight arrives.
        const ImVec2 b0{bx, ry + 3}, b1{bx + 13, ry + 16};
        dl->AddRect(b0, b1, kIvoryDim, 0.0f, 0, 1.4f);
        if (focus > 0.05f) {
            const float inset = 2.0f + 3.0f * (1.0f - focus);
            dl->AddRectFilled({b0.x + inset, b0.y + inset},
                              {b1.x - inset, b1.y - inset},
                              IM_COL32(232, 229, 220,
                                       static_cast<int>(230 * focus)));
        }
        const ImU32 col = focus > 0.5f ? kIvory : kIvoryDim;
        tracked_text(dl, font_item_, item_size, {bx + 30, ry - 2}, col,
                     kItems[i].label, 5.5f);
    }

    if (!activate) return;
    switch (selected_) {
        case 0: app.enter_simulation(false); break;
        case 1: app.enter_simulation(true); break;
        case 2: show_nohair_ = !show_nohair_; break;
        case 3: show_scope_ = !show_scope_; break;
        case 4: app.enter_flight(); break;   // Gate 24
        case 5: show_settings_ = !show_settings_; break;
        case 6: show_gallery_ = !show_gallery_; break;
        case 7: app.enter_exotic(); break;   // Gate 21 wormhole demo
        case 8: show_about_ = !show_about_; break;
        case 9: app.request_quit(); break;
    }
}

void MenuMode::draw_about(App& app) {
    (void)app;
    ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x - 470, 90},
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({440, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("ABOUT - Sagittarius A*", &show_about_);
    ImGui::TextWrapped(
        "Sagittarius A* is the supermassive black hole at the centre of the "
        "Milky Way, 8,277 pc from Earth. The rendered spacetime beside this "
        "panel is this simulator's own Kerr solution.");
    ImGui::Separator();
    ImGui::Text("Mass          4.297 +/- 0.012 x 10^6 Msun   (GRAVITY 2022)");
    ImGui::Text("Distance      8,277 +/- 9 pc                (GRAVITY 2022)");
    ImGui::Text("Spin (best)   a* ~ 0.94 prograde            (EHT + models)");
    ImGui::Text("Shadow        48.7 +/- 7 uas                (EHT 2022)");
    ImGui::Text("Photon ring   51.8 +/- 2.3 uas              (EHT 2022)");
    ImGui::Text("S2 period     16.0518 +/- 0.0022 yr         (Gillessen 2017)");
    ImGui::Separator();
    ImGui::TextDisabled("This build:");
    ImGui::BulletText("S2 fit: M = 4.270e6 Msun, chi2/dof = 0.96 (Phase 4)");
    ImGui::BulletText("Photon ring at Sgr A*: 49.7 uas (Phase 10)");
    ImGui::BulletText("Research paper: paper/main.pdf");
    ImGui::End();
}

void MenuMode::draw_nohair() {
    ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x - 500, 120},
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({470, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("NO-HAIR TEST - free quadrupole", &show_nohair_);
    ImGui::TextWrapped(
        "Kerr fixes the quadrupole at Q = -M a^2. The Hartle-Thorne module "
        "frees it: Q = -M a^2 + eps_q M^3. A non-zero eps_q would violate the "
        "no-hair theorem.");
    ImGui::Separator();
    ImGui::SliderFloat("eps_q (forecast)", &eps_q_slider_, -2.0e5f, 2.0e5f,
                       "%.0f");
    // Quadrupole pericentre-shift forecast for S2 (Will 2008 scaling):
    // d(omega)_Q = -3 pi eps_q (M/p)^2 per orbit. From the Phase-4 fit
    // (a = 1032 AU, e = 0.8828, R_g = 0.0424 AU): p = a(1-e^2) = 228 AU,
    // (M/p)^2 = 3.47e-8.
    const double dw_q = -3.0 * 3.14159265358979 * eps_q_slider_ * 3.47e-8;
    ImGui::Text("S2 quadrupole precession: %+.4f arcmin/orbit",
                dw_q * 180.0 / 3.14159265358979 * 60.0);
    ImGui::Text("(Schwarzschild term: +11.98 arcmin/orbit)");
    ImGui::Separator();
    if (nohair_loaded_) {
        ImGui::TextDisabled("Fit to Gillessen+2017 S2 data (Phase 5):");
        ImGui::Text("eps_q = 0 +/- %.3g   (95%% CI half-width)", eps_q_ci95_);
        ImGui::Text("sigma(eps_q) = %.3g", eps_q_sigma_);
        ImGui::Text("forecast, +10 S2 orbits: sigma -> %.3g", forecast_sigma_);
    } else {
        ImGui::TextDisabled("Phase 5 artefacts not found.");
    }
    ImGui::End();
}

void MenuMode::draw_scope_panel(App& app) {
    ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x - 460, 150},
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({430, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("PHOTON RING SCOPE", &show_scope_);
    ImGui::TextWrapped(
        "Photons that orbit the hole n extra half-turns before escaping form "
        "nested sub-rings converging on the critical curve (Johnson+2020).");
    ImGui::Separator();
    if (subring_loaded_) {
        ImGui::Text("flux(n=0)/flux(n=1) = %.1f", ratio_n0_n1_);
        ImGui::Text("flux(n=1)/flux(n=2) = %.1f", ratio_n1_n2_);
        ImGui::Text("Doppler beaming asymmetry = %.1fx", beaming_);
        ImGui::TextDisabled("(a*=0.94, incl 80 deg - Phase 3 CPU metrology)");
    } else {
        ImGui::TextDisabled("Phase 3 artefacts not found.");
    }
    ImGui::Separator();
    if (ImGui::Button("OPEN LIVE SCOPE (fov 0.22, r=60M)"))
        app.enter_simulation(false, true);
    ImGui::End();
}

void MenuMode::draw_gallery() {
    ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x - 560, 80},
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({530, 520}, ImGuiCond_FirstUseEver);
    ImGui::Begin("GALLERY - milestone renders", &show_gallery_);
    if (gallery_.empty()) {
        ImGui::TextDisabled("No renders found.");
    } else {
        const float cell_w = ImGui::GetContentRegionAvail().x / 2 - 8;
        int col = 0;
        for (const auto& g : gallery_) {
            ImGui::BeginGroup();
            const float ar = static_cast<float>(g.h) / g.w;
            ImGui::Image(static_cast<ImTextureID>(g.tex_id),
                         {cell_w, cell_w * ar});
            ImGui::TextWrapped("%s", g.name.c_str());
            ImGui::EndGroup();
            col = (col + 1) % 2;
            if (col != 0) ImGui::SameLine();
        }
    }
    ImGui::End();
}

bool MenuMode::draw(App& app) {
    if (!boot_panel_applied_) {
        boot_panel_applied_ = true;
        const std::string& p = app.boot_panel();
        if (p == "about") show_about_ = true;
        else if (p == "nohair") show_nohair_ = true;
        else if (p == "scope") show_scope_ = true;
        else if (p == "gallery") show_gallery_ = true;
        else if (p == "settings") show_settings_ = true;
    }
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float mx = std::max(64.0f, ds.x * 0.055f);
    draw_title(mx, ds.y * 0.14f);
    draw_items(app, mx, ds.y * 0.34f);
    draw_corners();

    if (show_about_) draw_about(app);
    if (show_nohair_) draw_nohair();
    if (show_scope_) draw_scope_panel(app);
    if (show_gallery_) draw_gallery();
    return show_settings_;
}

} // namespace sgr
