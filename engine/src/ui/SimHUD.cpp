#include "ui/SimHUD.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/observers.hpp"
#include "sgr_physics/infall.hpp"
#include "Wormhole.hpp"

#include "app.hpp"
#include "modes/MenuMode.hpp"
#include "vk/context.hpp"

namespace sgr {

using vk::check;

namespace {
// S-star ball colour from the EXACT line-of-sight redshift carried by the
// trajectory sample (1+z: full Doppler + gravitational, as fitted).
// White when approaching, warm gold at rest, bronze when receding; the
// signed square root keeps the slow outer stars visibly tinted.
struct StarShade {
    float r, g, b, u;   // u = signed, normalised LOS velocity
};
StarShade star_shade(double one_plus_z) {
    const double v_kms = (one_plus_z - 1.0) * 299792.458;
    const float u = std::clamp(
        static_cast<float>((v_kms < 0 ? -1.0 : 1.0) *
                           std::sqrt(std::abs(v_kms) / 4000.0)),
        -1.0f, 1.0f);
    StarShade s{1.0f, 0.82f, 0.55f, u};
    if (u <= 0.0f) {                 // gold -> white (approaching)
        s.g = 0.82f - 0.18f * u;
        s.b = 0.55f - 0.45f * u;
    } else {                         // gold -> bronze (receding)
        s.r = 1.00f - 0.42f * u;
        s.g = 0.82f - 0.55f * u;
        s.b = 0.55f - 0.48f * u;
    }
    return s;
}
ImU32 star_col(const StarShade& s, float alpha) {
    return IM_COL32(static_cast<int>(s.r * 255), static_cast<int>(s.g * 255),
                    static_cast<int>(s.b * 255), static_cast<int>(alpha));
}
} // namespace

SimHUD::SimHUD(vk::Context& ctx, App& app) : ctx_(&ctx), app_(&app) {
    // Render pass: single colour attachment (swapchain image), LOAD the
    // blitted compute output, end in PRESENT_SRC.
    VkAttachmentDescription att{};
    att.format = ctx_->swapchain_format();
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    check(vkCreateRenderPass(ctx_->device(), &rpci, nullptr, &render_pass_),
          "vkCreateRenderPass(hud)");
    create_framebuffers();

    // ImGui descriptor pool
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 32;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(ctx_->device(), &dpci, nullptr, &imgui_pool_),
          "vkCreateDescriptorPool(imgui)");
    // Release builds compile IM_ASSERT out - evaluate the ABI/layout check
    // explicitly so a header/lib mismatch fails loudly instead of corrupting.
    const bool layout_ok = ImGui::DebugCheckVersionAndDataLayout(
        IMGUI_VERSION, sizeof(ImGuiIO), sizeof(ImGuiStyle), sizeof(ImVec2),
        sizeof(ImVec4), sizeof(ImDrawVert), sizeof(ImDrawIdx));
    std::fprintf(stderr, "[hud] imgui %s layout check: %s\n", IMGUI_VERSION,
                 layout_ok ? "OK" : "MISMATCH");
    std::fflush(stderr);
    if (!layout_ok)
        throw std::runtime_error("ImGui header/library layout mismatch");
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.0f;
    if (app_->menu_enabled()) {
        // Menu fonts must be in the atlas before the backend builds it.
        menu_ = std::make_unique<MenuMode>();
        menu_->load_fonts();
    }
    ImGui_ImplGlfw_InitForVulkan(ctx_->window(), true);

    // Vendored backend built with VK_NO_PROTOTYPES: route its dispatch
    // through the volk-loaded instance.
    ImGui_ImplVulkan_LoadFunctions(
        VK_API_VERSION_1_2,
        [](const char* name, void*) {
            return vkGetInstanceProcAddr(volkGetLoadedInstance(), name);
        });

    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_2;
    ii.Instance = ctx_->instance();
    ii.PhysicalDevice = ctx_->physical();
    ii.Device = ctx_->device();
    ii.QueueFamily = ctx_->queue_family();
    ii.Queue = ctx_->queue();
    ii.DescriptorPool = imgui_pool_;
    ii.MinImageCount = 2;
    ii.ImageCount = static_cast<uint32_t>(ctx_->swapchain_images().size());
    ii.PipelineInfoMain.RenderPass = render_pass_;
    ii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&ii);

    load_science_artifacts();
    load_cluster_orbits();
    if (menu_) menu_->load_artifacts(*app_);
}

void SimHUD::create_framebuffers() {
    const auto& images = ctx_->swapchain_images();
    sc_views_.resize(images.size());
    framebuffers_.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ctx_->swapchain_format();
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(ctx_->device(), &vci, nullptr, &sc_views_[i]),
              "vkCreateImageView(sc)");
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = render_pass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &sc_views_[i];
        fci.width = ctx_->swapchain_extent().width;
        fci.height = ctx_->swapchain_extent().height;
        fci.layers = 1;
        check(vkCreateFramebuffer(ctx_->device(), &fci, nullptr, &framebuffers_[i]),
              "vkCreateFramebuffer(hud)");
    }
}

void SimHUD::destroy_framebuffers() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(ctx_->device(), fb, nullptr);
    for (auto v : sc_views_) vkDestroyImageView(ctx_->device(), v, nullptr);
    framebuffers_.clear();
    sc_views_.clear();
}

void SimHUD::on_swapchain_recreated() {
    destroy_framebuffers();
    create_framebuffers();
    ImGui_ImplVulkan_SetMinImageCount(2);
}

void SimHUD::load_science_artifacts() {
    namespace fs = std::filesystem;
    const fs::path base = "validation/phase4";
    const fs::path data = "data/gillessen_2017";
    if (!fs::exists(base / "s2_model_track.csv")) return;

    auto load_csv = [](const fs::path& p, auto cb) {
        std::ifstream f(p);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::vector<std::string> tok;
            std::string t;
            while (std::getline(ss, t, ',')) tok.push_back(t);
            cb(tok);
        }
    };
    load_csv(base / "s2_model_track.csv", [&](const std::vector<std::string>& v) {
        model_.t.push_back(std::stof(v[0]));
        model_.ra.push_back(std::stof(v[1]));
        model_.dec.push_back(std::stof(v[2]));
        model_.vr.push_back(std::stof(v[3]));
    });
    if (fs::exists(data / "s2_astrometry.csv"))
        load_csv(data / "s2_astrometry.csv", [&](const std::vector<std::string>& v) {
            ast_.t.push_back(std::stof(v[0]));
            ast_.ra.push_back(std::stof(v[1]));
            ast_.dec.push_back(std::stof(v[2]));
        });
    // fit summary from the JSON (plain text scan - no JSON dep in the engine)
    std::ifstream fj(base / "s2_fit.json");
    if (fj) {
        std::stringstream ss;
        ss << fj.rdbuf();
        const std::string j = ss.str();
        auto grab = [&](const char* key) -> double {
            const auto k = j.find(std::string("\"") + key + "\":");
            if (k == std::string::npos) return 0.0;
            // Scan for the value AFTER the colon: key names such as "R0_pc"
            // and "chi2_red" contain digits, so starting the search at the
            // key position would grab the '0'/'2' inside the name.
            const auto colon = j.find(':', k);
            auto s = j.find_first_of("0123456789.-", colon);
            auto e = j.find_first_not_of("0123456789.-e+", s);
            return std::stod(j.substr(s, e - s));
        };
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "M = %.3f e6 Msun   R0 = %.0f pc   chi2/dof = %.3f",
                      grab("M_msun") / 1e6, grab("R0_pc"), grab("chi2_red"));
        fit_summary_ = buf;
    }
    science_available_ = true;
}

void SimHUD::load_cluster_orbits() {
    namespace fs = std::filesystem;
    const fs::path p = "validation/phase13/cluster_fit.json";
    if (!fs::exists(p)) return;
    nlohmann::json j;
    try {
        std::ifstream(p) >> j;
    } catch (...) {
        return;
    }
    if (!j.contains("joint_elements")) return;
    const double M = j["M_msun"], R0 = j["R0_pc"];
    // per-star orbit-line colours (the balls are coloured by redshift)
    static const uint32_t kCols[] = {
        IM_COL32(130, 185, 255, 255), // S2  ice blue
        IM_COL32(95, 220, 190, 255),  // S1  teal
        IM_COL32(235, 125, 120, 255), // S8  soft red
        IM_COL32(195, 145, 255, 255), // S12 violet
        IM_COL32(255, 185, 95, 255),  // S13 amber
    };
    int ci = 0;
    for (const auto& e : j["joint_elements"]) {
        ClusterStar st;
        st.name = e["star"].get<std::string>();
        S2Parameters prm;
        prm.M_msun = M;
        prm.R0_pc = R0;
        prm.a_mas = e["a_mas"];
        prm.e = e["e"];
        prm.incl_deg = e["incl_deg"];
        prm.Omega_deg = e["Omega_deg"];
        prm.omega_deg = e["omega_deg"];
        prm.t_peri_yr = e["t_peri_yr"];
        // BH-centred view: the instrumental frame offsets/drift are zeroed
        prm.x0_mas = prm.y0_mas = 0.0;
        prm.vx_masyr = prm.vy_masyr = 0.0;
        prm.vz_kms = 0.0;
        st.period_yr = e["period_yr"];
        st.base_period_yr = st.period_yr;
        st.base = prm;
        // Display limits: the instrument integrates hypothetical orbits
        // shrunk by up to ~2000x. They are ordinary bound GR orbits, so the
        // fitter's floors (5 mas / 100 M / 50 M plunge) are lowered HERE
        // only — the fitting path keeps the strict defaults untouched.
        st.model.set_limits(1e-4, 20.0, 3.0);
        st.t0 = 1992.0;
        st.t1 = 1992.0 + std::max(160.0, st.period_yr * 1.05);
        if (!st.model.integrate(prm, st.t0, st.t1)) continue;
        st.mas_per_M = st.model.mas_per_M(prm);
        // semi-major axis in geometric units (mas_per_M is the exact
        // conversion at this (M, R0) — nothing is assumed here)
        st.a_M = (st.mas_per_M > 0.0) ? prm.a_mas / st.mas_per_M : 0.0;
        st.line_col = kCols[ci++ % 5];
        // dense display track in BH-frame M (index subsampling keeps the
        // adaptive pericentre resolution of the integrator)
        const auto& tr = st.model.trajectory();
        const size_t stride = std::max<size_t>(1, tr.size() / 1400);
        for (size_t i = 0; i < tr.size(); i += stride)
            st.track.push_back({static_cast<float>(tr[i].pos_M[0]),
                                static_cast<float>(tr[i].pos_M[1]),
                                static_cast<float>(tr[i].pos_M[2])});
        cluster_.push_back(std::move(st));
    }
    if (cluster_.size() < 2) {
        cluster_.clear();
        return;
    }
    clu_end_ = 1e9;
    for (const auto& st : cluster_) clu_end_ = std::min(clu_end_, st.t1);
    clu_end_ -= 1.0;
    clu_M_ = M;
    // metres per geometric M for THIS fitted mass (no hardcoded constant)
    clu_rg_m_ = M * constants::GM_sun_si /
                (constants::c_si * constants::c_si);
    // largest apoapsis actually reached, straight off the integrated
    // tracks (no assumed ellipse) — drives the in-scene "fly out"
    for (const auto& st : cluster_)
        for (const auto& q : st.track)
            clu_rmax_M_ = std::max(
                clu_rmax_M_,
                static_cast<double>(std::sqrt(q[0] * q[0] + q[1] * q[1] +
                                              q[2] * q[2])));
    for (auto& st : cluster_) st.peri_M = st.a_M * (1.0 - st.base.e);
    cluster_available_ = true;
    load_stellar_parameters();

    // boot-time focus (--focus S2 [--ride]) for repeatable screenshots
    const std::string& want = app_->focus_star();
    if (!want.empty()) {
        for (size_t i = 0; i < cluster_.size(); ++i)
            if (cluster_[i].name == want) {
                app_->live_.focus_index = static_cast<int>(i) + 1;
                app_->live_.focus_dist = static_cast<float>(
                    std::clamp(cluster_[i].radius_M * 40.0, 0.5, 400.0));
            }
    }
    // --epoch Y: freeze the S-star clock there (repeatable screenshots)
    if (app_->boot_epoch() > 0.0) {
        clu_epoch_ = app_->boot_epoch();
        clu_play_ = false;
    }
}

// Habibi et al. (2017), Table 2 — measured Teff, log g and mass. The stellar
// RADIUS is derived here from the paper's own quantities, R = sqrt(G M*/g),
// rather than taken from anywhere: the paper does not tabulate it. The tidal
// radius then follows as r_t = eta * R* * (M_BH/M*)^(1/3), with M_BH from our
// own cluster fit, so no constant is smuggled in.
void SimHUD::load_stellar_parameters() {
    namespace fs = std::filesystem;
    const fs::path p = "data/habibi_2017/s_star_parameters.csv";
    if (!fs::exists(p) || clu_rg_m_ <= 0.0) return;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("star,", 0) == 0)
            continue;
        std::stringstream ss(line);
        std::vector<std::string> v;
        std::string tok;
        while (std::getline(ss, tok, ',')) v.push_back(tok);
        if (v.size() < 8) continue;
        for (auto& st : cluster_) {
            if (st.name != v[0]) continue;
            st.teff_k = std::stod(v[1]);
            st.logg = std::stod(v[3]);
            st.mass_msun = std::stod(v[5]);
            st.assumed = (std::stoi(v[7]) != 0);
            // g: log10(cm/s^2) -> m/s^2 ; R = sqrt(G M*/g) [m] -> units of M
            const double g_si = std::pow(10.0, st.logg) * 0.01;
            const double m_kg = st.mass_msun * constants::M_sun_kg;
            const double R_m = std::sqrt(constants::G_si * m_kg / g_si);
            st.radius_M = R_m / clu_rg_m_;
            st.r_tidal_M = st.radius_M *
                           std::cbrt(clu_M_ / std::max(st.mass_msun, 1e-6));
            st.has_params = true;
        }
    }
}

// The 1:N control is a REAL orbit shrink, not a drawing trick: the orbit is
// re-integrated with a -> a/N through the same validated S2Model, so the
// period follows Kepler III (P/N^1.5) and relativistic precession grows on
// its own. N = 1 restores the fitted orbit exactly.
void SimHUD::rescale_cluster(float ratio) {
    const double N = std::max(1.0, static_cast<double>(ratio));
    if (std::abs(N - clu_scale_applied_) < 1e-9) return;
    clu_rmax_M_ = 0.0;
    for (auto& st : cluster_) {
        S2Parameters prm = st.base;
        prm.a_mas = st.base.a_mas / N;
        // Anchor pericentre at the start of the display span. The fitted
        // t_peri (2002) is meaningless for a hypothetical orbit whose period
        // is now hours, and integrating from 2002 back to 1992 would cost
        // hundreds of orbits per star for nothing. At N = 1 the fitted epoch
        // is kept exactly.
        if (N > 1.0) prm.t_peri_yr = 1992.0;
        st.phase = ClusterStar::Phase::Intact;
        st.stretch = 0.0;
        st.trail.clear();
        st.track.clear();
        // Kepler III: P scales as a^{3/2}, so the shrunken star genuinely
        // races round instead of keeping its 16-year period.
        st.period_yr = st.base_period_yr / std::pow(N, 1.5);
        st.t0 = 1992.0;
        // integrate a few periods (and never a silly-long span for the
        // fast shrunken orbits)
        st.t1 = st.t0 + std::clamp(st.period_yr * 6.0, 0.05, 160.0);
        st.valid = st.model.integrate(prm, st.t0, st.t1);
        if (!st.valid) {
            // The orbit could not be integrated at this scale — report that
            // honestly rather than dressing it up as a disruption.
            st.a_M = st.base.a_mas / N / st.mas_per_M;
            st.peri_M = st.a_M * (1.0 - st.base.e);
            continue;
        }
        st.a_M = prm.a_mas / st.mas_per_M;
        st.peri_M = st.a_M * (1.0 - prm.e);
        const auto& tr = st.model.trajectory();
        const size_t stride = std::max<size_t>(1, tr.size() / 1400);
        for (size_t i = 0; i < tr.size(); i += stride) {
            st.track.push_back({static_cast<float>(tr[i].pos_M[0]),
                                static_cast<float>(tr[i].pos_M[1]),
                                static_cast<float>(tr[i].pos_M[2])});
            clu_rmax_M_ = std::max(
                clu_rmax_M_,
                std::sqrt(tr[i].pos_M[0] * tr[i].pos_M[0] +
                          tr[i].pos_M[1] * tr[i].pos_M[1] +
                          tr[i].pos_M[2] * tr[i].pos_M[2]));
        }
    }
    clu_scale_applied_ = N;
    // The integrated span collapses with the period, so re-derive the wrap
    // point and put the clock back at the start of it.
    clu_end_ = 1e9;
    double p_fast = 1e9;
    for (const auto& st : cluster_) {
        if (st.track.empty()) continue;
        clu_end_ = std::min(clu_end_, st.t1);
        p_fast = std::min(p_fast, st.period_yr);
    }
    if (clu_end_ > 1e8) clu_end_ = 2152.0;
    clu_epoch_ = 1992.0;
    // Keep it watchable: one orbit of the fastest star in ~8 s of wall
    // clock. Without this a 1:1000 orbit (period ~4 h) would flicker past
    // hundreds of times a second at the default multiplier.
    if (p_fast < 1e8 && p_fast > 0.0)
        clu_speed_ = static_cast<float>(
            std::clamp(p_fast * constants::yr_s / 8.0, 1.0e4, 1.0e8));
}

bool SimHUD::wants_mouse() const {
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
}

void SimHUD::new_frame(double fps) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (menu_ && app_->mode() == App::UiMode::Menu) {
        if (menu_->draw(*app_)) draw_controls(fps);
    } else if (app_->live_.wormhole) {
        draw_exotic(fps);   // a different spacetime: none of the Sgr A* panels
    } else if (app_->live_.flight) {
        draw_flight(fps);   // Gate 24: the in-faller's own instruments
    } else {
        if (app_->live_.show_stars || app_->live_.science_mode) {
            rescale_cluster(app_->live_.star_scale_ratio);
            update_cluster_time();
        }
        // camera first: the overlays project through the same frame the
        // ray tracer is about to use this frame
        update_focus_camera();
        // background draw list: over the rendered scene, under the panels
        if (app_->live_.show_stars) {
            draw_stars_overlay();
            draw_focus_list();
        }
        draw_controls(fps);
        if (app_->live_.science_mode) {
            draw_science();
            draw_cluster3d();
        }
        if (app_->live_.scope_mode) draw_scope_overlay();
    }
    ImGui::Render();
}

void SimHUD::update_cluster_time() {
    if (!cluster_available_ || !clu_play_) return;
    clu_epoch_ += ImGui::GetIO().DeltaTime * clu_speed_ / 3.1557e7;
    if (clu_epoch_ > clu_end_) clu_epoch_ = 1992.0; // wrap the span
}

// Target list: Sgr A* plus every star, with the live state that decides
// whether it survives the current (hypothetical) orbit shrink.
void SimHUD::draw_focus_list() {
    if (!cluster_available_) return;
    auto& lp = app_->live_;
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos({ds.x - 330, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({315, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Focus");
    const double k = 1.0 / std::max(lp.star_scale_ratio, 1.0f);
    const double N = std::max(1.0f, lp.star_scale_ratio);

    if (ImGui::RadioButton("Sgr A* (free camera)", lp.focus_index == 0))
        lp.focus_index = 0;
    ImGui::Separator();
    for (size_t i = 0; i < cluster_.size(); ++i) {
        ClusterStar& st = cluster_[i];
        const int idx = static_cast<int>(i) + 1;
        ImGui::PushID(idx);
        const bool gone = st.phase == ClusterStar::Phase::Disrupted ||
                          !st.valid;
        if (gone) ImGui::BeginDisabled();
        if (ImGui::RadioButton(st.name.c_str(), lp.focus_index == idx)) {
            lp.focus_index = idx;
            // frame the star at a few stellar radii, or a sane minimum
            lp.focus_dist = static_cast<float>(
                std::clamp(st.radius_M * k * 40.0, 0.5, 400.0));
        }
        if (gone) ImGui::EndDisabled();
        ImGui::SameLine();
        if (st.has_params) {
            const double rt = clu_eta_ * st.r_tidal_M;
            const bool doomed = st.peri_M <= rt;
            const char* verdict =
                !st.valid ? "orbit n/a"
                          : (st.phase == ClusterStar::Phase::Disrupted
                                 ? "disrupted"
                                 : (doomed ? "DISRUPTS" : "survives"));
            ImGui::TextColored(
                gone ? ImVec4(0.85f, 0.35f, 0.30f, 1.0f)
                     : (doomed ? ImVec4(0.95f, 0.65f, 0.25f, 1.0f)
                               : ImVec4(0.55f, 0.75f, 0.55f, 1.0f)),
                "%s", verdict);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "M* = %.1f Msun%s, R* = %.2f Rsun (derived from log g)\n"
                    "r_t = %.1f M   pericentre now = %.1f M\n"
                    "Habibi+2017 spectroscopy; r_t = eta R* (M_BH/M*)^(1/3)",
                    st.mass_msun, st.assumed ? " (ASSUMED)" : "",
                    st.radius_M * clu_rg_m_ / 6.957e8, rt, st.peri_M);
        } else {
            ImGui::TextDisabled("no spectroscopy");
        }
        ImGui::PopID();
    }
    ImGui::Separator();
    ImGui::Checkbox("ride the star", &lp.focus_ride);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Sit on the star and look back at Sgr A*.\n"
                          "Unticked: orbit the star (drag/scroll).");
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat("eta", &clu_eta_, 1.0f, 2.44f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("1.00 = classic tidal radius R*(M_BH/M*)^(1/3)\n"
                          "2.44 = fluid Roche limit (Roche's coefficient)");
    ImGui::SameLine();
    ImGui::Checkbox("disruption", &clu_disruption_);
    if (ImGui::Button("reset stars")) {
        for (auto& st : cluster_) {
            st.phase = ClusterStar::Phase::Intact;
            st.stretch = 0.0;
        }
        debris_.clear();
        clu_epoch_ = 1992.0;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu debris", debris_.size());
    if (N > 1.0)
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f),
                           "HYPOTHETICAL: orbits shrunk %.0fx (P/N^1.5)",
                           N);
    else
        ImGui::TextDisabled("true fitted orbits");
    ImGui::End();
}

Vec3 SimHUD::focus_target(int index, double k) const {
    if (index <= 0 || index > static_cast<int>(cluster_.size()))
        return {0.0, 0.0, 0.0};
    const ClusterStar& st = cluster_[index - 1];
    if (st.track.empty()) return {0.0, 0.0, 0.0};
    const double span = std::max(st.t1 - st.t0, 1e-9);
    const double t = std::clamp(clu_epoch_, st.t0 + 1e-4 * span,
                                st.t1 - 1e-4 * span);
    const TrajectorySample s = st.model.sample_at(t);
    return {s.pos_M[0] * k, s.pos_M[1] * k, s.pos_M[2] * k};
}

// Places the camera for this frame and publishes the basis the ray tracer
// will use. With no focus this is exactly the historical configuration
// (camera at (r_cam, incl, cam_phi), aimed at the hole); with a star focused
// the camera either orbits it or rides it, and the aim follows the star.
void SimHUD::update_focus_camera() {
    auto& lp = app_->live_;
    const double k = 1.0 / std::max(lp.star_scale_ratio, 1.0f);
    int idx = lp.focus_index;
    if (idx > static_cast<int>(cluster_.size())) idx = 0;
    // A disrupted (or unintegrable) star is no longer there to look at: drop
    // the lock for good and leave the camera where it was, rather than
    // snapping it somewhere arbitrary.
    if (idx > 0 && (cluster_[idx - 1].phase == ClusterStar::Phase::Disrupted ||
                    !cluster_[idx - 1].valid)) {
        idx = 0;
        lp.focus_index = 0;
    }

    bool ride_valid = false;
    if (idx <= 0 || !cluster_available_) {
        cam_frame_ = camera_frame(lp.r_cam, lp.incl_deg, lp.cam_phi);
    } else {
        const Vec3 tgt = focus_target(idx, k);
        Vec3 cam{};
        if (lp.focus_ride) {
            // sit at the star (offset by a hair so the aim is well defined)
            // and look back at the hole
            const double L = std::max(v_len(tgt), 1e-6);
            const double off = std::max(0.002 * L, 0.05);
            cam = {tgt[0] * (1.0 + off / L), tgt[1] * (1.0 + off / L),
                   tgt[2] * (1.0 + off / L)};
        } else {
            const double cy = std::cos(lp.focus_yaw), sy = std::sin(lp.focus_yaw);
            const double cp = std::cos(lp.focus_pitch),
                         sp = std::sin(lp.focus_pitch);
            const double d = std::max(lp.focus_dist, 0.05f);
            cam = {tgt[0] + d * cp * cy, tgt[1] + d * cp * sy,
                   tgt[2] + d * sp};
        }
        // keep the camera outside the hole and inside the supported range
        double r = v_len(cam);
        if (r < 15.0) {
            const double s = 15.0 / std::max(r, 1e-9);
            cam = {cam[0] * s, cam[1] * s, cam[2] * s};
            r = 15.0;
        }
        double rc = lp.r_cam, inc = lp.incl_deg, phi = lp.cam_phi;
        cartesian_to_camera(cam, rc, inc, phi);
        lp.r_cam = static_cast<float>(std::min(rc, 1.0e6));
        lp.incl_deg = static_cast<float>(std::clamp(inc, 0.05, 179.95));
        lp.cam_phi = static_cast<float>(phi);
        cam_frame_ = camera_frame(lp.r_cam, lp.incl_deg, lp.cam_phi,
                                  lp.focus_ride ? Vec3{0.0, 0.0, 0.0} : tgt);
        if (lp.focus_ride) {
            // Observer::RideStar: the star's own velocity (dx/dt [c] from
            // its full-GR orbit) in the camera's local (r, theta, phi)
            const ClusterStar& st = cluster_[idx - 1];
            const double span = std::max(st.t1 - st.t0, 1e-9);
            const double t = std::clamp(clu_epoch_, st.t0 + 1e-4 * span,
                                        st.t1 - 1e-4 * span);
            const TrajectorySample s = st.model.sample_at(t);
            const Vec3 vel{s.vel[0], s.vel[1], s.vel[2]};
            lp.ride_v[0] = static_cast<float>(v_dot(vel, cam_frame_.er));
            lp.ride_v[1] = static_cast<float>(v_dot(vel, cam_frame_.eth));
            lp.ride_v[2] = static_cast<float>(v_dot(vel, cam_frame_.eph));
            ride_valid = true;
        }
    }
    lp.ride_v_valid = ride_valid;
    lp.fwd_r = static_cast<float>(cam_frame_.fwd[0]);
    lp.fwd_th = static_cast<float>(cam_frame_.fwd[1]);
    lp.fwd_ph = static_cast<float>(cam_frame_.fwd[2]);
    lp.rgt_r = static_cast<float>(cam_frame_.rgt[0]);
    lp.rgt_th = static_cast<float>(cam_frame_.rgt[1]);
    lp.rgt_ph = static_cast<float>(cam_frame_.rgt[2]);
    lp.up_r = static_cast<float>(cam_frame_.up[0]);
    lp.up_th = static_cast<float>(cam_frame_.up[1]);
    lp.up_ph = static_cast<float>(cam_frame_.up[2]);
}

// Tidal debris, frozen-in approximation (Rees 1988): at disruption every
// fragment keeps the star's orbital plane and pericentre direction but
// carries a spread in specific binding energy
//     dE = G M_BH R* / r_t^2   ->   1/a_i = 1/a -+ 2 dE      (G = M_BH = 1)
// so the fragments fan out into orbits of different period — the classic
// TDE "fallback" stream. Only the BOUND half is drawn; the other half is
// unbound and leaves the scene (stated in the panel). Each fragment is an
// ordinary orbit in the SAME validated S2Model, so nothing here re-derives
// physics.
void SimHUD::spawn_debris(size_t star_index) {
    if (star_index >= cluster_.size()) return;
    ClusterStar& st = cluster_[star_index];
    if (!st.has_params || st.a_M <= 0.0) return;
    if (debris_.size() > 400) return;   // several stars may disrupt in turn
    const double rt = std::max(clu_eta_ * st.r_tidal_M, 1e-9);
    const double dE = st.radius_M / (rt * rt);       // geometric units
    const double inv_a = 1.0 / st.a_M;
    const double r_p = std::max(st.peri_M, 1e-6);
    const int N = 48;
    for (int i = 0; i < N; ++i) {
        const double u = -1.0 + 2.0 * (i + 0.5) / N;   // -1..1
        const double inv_ai = inv_a + 2.0 * dE * u;
        if (inv_ai <= 1e-9) continue;                  // unbound half
        const double a_i = 1.0 / inv_ai;
        if (a_i <= r_p) continue;
        DebrisParticle d;
        d.model.set_limits(1e-4, 20.0, 3.0);
        S2Parameters prm = st.base;
        prm.a_mas = a_i * st.mas_per_M;
        prm.e = std::clamp(1.0 - r_p / a_i, 0.0, 0.995);
        // the fragment leaves pericentre at the moment of disruption
        prm.t_peri_yr = st.disrupt_epoch;
        const double P_i = st.base_period_yr *
                           std::pow(a_i / (st.base.a_mas / st.mas_per_M), 1.5);
        d.t0 = st.disrupt_epoch;
        d.t1 = d.t0 + std::clamp(P_i * 3.0, 1e-3, 200.0);
        if (!d.model.integrate(prm, d.t0, d.t1)) continue;
        d.ok = true;
        d.col = st.line_col;
        debris_.push_back(std::move(d));
    }
}

void SimHUD::draw_debris(ImDrawList* dl, double k, float W, float H) {
    if (debris_.empty()) return;
    const auto& lp = app_->live_;
    for (const auto& d : debris_) {
        if (!d.ok) continue;
        const double span = std::max(d.t1 - d.t0, 1e-9);
        double t = clu_epoch_;
        if (t < d.t0) continue;
        t = std::clamp(t, d.t0 + 1e-4 * span, d.t1 - 1e-4 * span);
        const TrajectorySample s = d.model.sample_at(t);
        const Projected q = project_point(
            cam_frame_, Vec3{s.pos_M[0] * k, s.pos_M[1] * k, s.pos_M[2] * k},
            lp.fov, W, H);
        if (!q.ok || q.vis <= 0.02) continue;
        const StarShade sh = star_shade(s.redshift);
        const ImVec2 p(static_cast<float>(q.x), static_cast<float>(q.y));
        dl->AddCircleFilled(p, 2.6f, star_col(sh, 40.0f * float(q.vis)));
        dl->AddCircleFilled(p, 1.3f, star_col(sh, 220.0f * float(q.vis)));
    }
}

// The S-stars drawn INTO the live scene, projected through the exact
// inverse of the compute shader's camera model (same ZAMO tetrad, same
// fov and aspect convention), so a star sits precisely where that camera
// would see it and tracks the mouse orbit/zoom for free.
//
// Scale caveat, stated in the UI: the fitted orbits are ~2e4 M across
// against a ~20 M disk, so the radius is compressed by the labelled
// factor. Orbit SHAPE, phase, period and the redshift colour are the
// unmodified fitted geodesics; only the overall radial scale is a
// display choice (as in the published NCSA/UCLA cluster renderings).
// Starlight bending is not applied to the sprites — with a compressed
// radius it would not correspond to any real geometry either.
void SimHUD::draw_stars_overlay() {
    if (!cluster_available_) return;
    const auto& lp = app_->live_;
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float W = ds.x, H = ds.y;
    if (W < 16.0f || H < 16.0f) return;

    // Camera-independent scale: 1 = true geometric distance (the renderer
    // already works in real M, so this is the physical configuration);
    // larger ratios pull the orbits inward by that factor only.
    const double k = 1.0 / std::max(lp.star_scale_ratio, 1.0f);

    // One camera frame, shared with the ray tracer: cam_frame_ is what
    // update_focus_camera() pushed into the shader this frame, so a sprite
    // and its ray-traced surroundings can never disagree (CameraBasis.hpp).
    const CameraFrame& cf = cam_frame_;
    struct Proj { ImVec2 p; float vis; bool ok; double dist; };
    auto project = [&](double x, double y, double z) {
        const Projected q = project_point(cf, Vec3{x, y, z}, lp.fov, W, H);
        return Proj{ImVec2(static_cast<float>(q.x), static_cast<float>(q.y)),
                    static_cast<float>(q.vis), q.ok, q.dist};
    };

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    for (auto& st : cluster_) {
        // orbit line (the full integrated GR track — S2's pericentre
        // advance draws its own rosette); optional, so the stars can be
        // watched as plain points of light
        Proj prev{};
        if (lp.show_star_paths)
            for (const auto& q : st.track) {
                const Proj cur = project(q[0] * k, q[1] * k, q[2] * k);
                if (cur.ok && prev.ok) {
                    const float a = 60.0f * std::min(cur.vis, prev.vis);
                    if (a > 3.0f)
                        dl->AddLine(prev.p, cur.p,
                                    (st.line_col & 0x00ffffffu) |
                                        (static_cast<uint32_t>(a) << 24),
                                    1.1f);
                }
                prev = cur;
            }

        if (st.phase == ClusterStar::Phase::Disrupted || !st.valid ||
            st.track.empty())
            continue;

        const double span = std::max(st.t1 - st.t0, 1e-9);
        const double t = std::clamp(clu_epoch_, st.t0 + 1e-4 * span,
                                    st.t1 - 1e-4 * span);
        const TrajectorySample smp = st.model.sample_at(t);
        const double r_now = std::sqrt(smp.pos_M[0] * smp.pos_M[0] +
                                       smp.pos_M[1] * smp.pos_M[1] +
                                       smp.pos_M[2] * smp.pos_M[2]);

        // ---- tidal state (display-layer what-if; see the panel caption) ---
        // r_t = eta * R* * (M_BH/M*)^(1/3) from the Habibi+2017 spectroscopy.
        // The strain that stretches the star goes as (r_t/r)^3, so the visual
        // elongation follows the same law rather than an arbitrary ramp.
        if (clu_disruption_ && st.has_params && st.r_tidal_M > 0.0) {
            const double rt = clu_eta_ * st.r_tidal_M;
            if (r_now <= rt) {
                st.phase = ClusterStar::Phase::Disrupted;
                st.disrupt_epoch = t;
                spawn_debris(static_cast<size_t>(&st - cluster_.data()));
                continue;
            }
            const double x = rt / std::max(r_now, 1e-9);
            st.stretch = std::clamp(x * x * x, 0.0, 1.0);
            st.phase = (st.stretch > 0.02) ? ClusterStar::Phase::Stretching
                                           : ClusterStar::Phase::Intact;
        }

        // riding this star: you are ON it, so it must not block the view
        const int my_idx = static_cast<int>(&st - cluster_.data()) + 1;
        if (lp.focus_ride && lp.focus_index == my_idx) continue;

        const Proj cur = project(smp.pos_M[0] * k, smp.pos_M[1] * k,
                                 smp.pos_M[2] * k);
        if (!cur.ok || cur.vis <= 0.02f) continue;
        const StarShade sh = star_shade(smp.redshift);
        // angular size from the DERIVED stellar radius, so flying close in
        // focus mode grows the star honestly instead of holding a fixed dot
        float rad = 3.2f + 2.2f * std::min(1.0f, std::abs(sh.u));
        if (st.radius_M > 0.0 && cur.dist > 1e-9) {
            const double ang = st.radius_M * k / cur.dist;   // radians
            const double px = ang / std::max<double>(lp.fov, 1e-3) * W;
            rad = static_cast<float>(std::max<double>(rad, px));
        }
        const float a_glow = 255.0f * cur.vis;
        if (st.stretch > 0.02) {
            // spaghettification cue: an ellipse elongated along the radial
            // direction, drawn as a short streak toward/away from the hole
            const Proj inward = project(smp.pos_M[0] * k * 0.94,
                                        smp.pos_M[1] * k * 0.94,
                                        smp.pos_M[2] * k * 0.94);
            if (inward.ok) {
                const float ex = inward.p.x - cur.p.x, ey = inward.p.y - cur.p.y;
                const float L = std::sqrt(ex * ex + ey * ey) + 1e-3f;
                const float s = static_cast<float>(st.stretch) * 6.0f * rad;
                const ImVec2 p0{cur.p.x - ex / L * s, cur.p.y - ey / L * s};
                const ImVec2 p1{cur.p.x + ex / L * s, cur.p.y + ey / L * s};
                dl->AddLine(p0, p1, star_col(sh, a_glow * 0.75f),
                            rad * 1.4f);
            }
        }
        // Unresolved stars get a wide point-source glow; once the disc is
        // actually resolved (flying up to it in focus mode) the halo tightens
        // so it reads as a sphere instead of concentric rings.
        rad = std::min(rad, 0.45f * std::min(W, H));
        const bool resolved = rad > 6.0f;
        const float g1 = resolved ? 1.9f : 4.0f;
        const float g2 = resolved ? 1.32f : 2.1f;
        dl->AddCircleFilled(cur.p, rad * g1, star_col(sh, 22.0f * cur.vis));
        dl->AddCircleFilled(cur.p, rad * g2, star_col(sh, 60.0f * cur.vis));
        dl->AddCircleFilled(cur.p, rad, star_col(sh, a_glow));
        dl->AddCircleFilled(cur.p, rad * (resolved ? 0.72f : 0.45f),
                            IM_COL32(255, 255, 255,
                                     static_cast<int>((resolved ? 90 : 220) *
                                                      cur.vis)));
        dl->AddText({cur.p.x + rad + 5.0f, cur.p.y - 7.0f},
                    (st.line_col & 0x00ffffffu) |
                        (static_cast<uint32_t>(200 * cur.vis) << 24),
                    st.name.c_str());
    }
    draw_debris(dl, k, W, H);
}

void SimHUD::draw_cluster3d() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos({ds.x - 560, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({545, 640}, ImGuiCond_FirstUseEver);
    ImGui::Begin("S-star cluster - 3D orbits");
    if (!cluster_available_) {
        ImGui::TextWrapped(
            "Phase-13 artefacts not found (or missing joint elements). Run "
            "fit_cluster to generate validation/phase13/cluster_fit.json.");
        ImGui::End();
        return;
    }

    // ---- time controls ------------------------------------------------------
    if (ImGui::Button(clu_play_ ? "pause" : "play")) clu_play_ = !clu_play_;
    ImGui::SameLine();
    if (ImGui::SmallButton("1995")) clu_epoch_ = 1995.0;
    ImGui::SameLine();
    if (ImGui::SmallButton("today")) clu_epoch_ = 2026.55;
    ImGui::SameLine();
    ImGui::Text("epoch %.2f", clu_epoch_);
    ImGui::SetNextItemWidth(220.0f);
    ImGui::SliderFloat("time [x real]", &clu_speed_, 1.0e4f, 1.0e8f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    ImGui::TextDisabled("= %.2f yr/s", clu_speed_ / 3.1557e7);

    // ---- 3-D canvas ---------------------------------------------------------
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float W = ImGui::GetContentRegionAvail().x;
    const float H = std::max(ImGui::GetContentRegionAvail().y - 42.0f, 120.0f);
    ImGui::InvisibleButton("cluster_canvas", {W, H});
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
        clu_yaw_ += ImGui::GetIO().MouseDelta.x * 0.008f;
        clu_pitch_ += ImGui::GetIO().MouseDelta.y * 0.008f;
        clu_pitch_ = std::clamp(clu_pitch_, -1.55f, 1.55f);
    }
    if (ImGui::IsItemHovered()) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f)
            clu_zoom_ = std::clamp(clu_zoom_ * std::pow(1.18f, wheel),
                                   0.15f, 40.0f);
        if (ImGui::IsMouseDoubleClicked(0)) { // reset to the Earth view
            clu_yaw_ = clu_pitch_ = 0.0f;
            clu_zoom_ = 1.0f;
        }
    }
    dl->PushClipRect(origin, {origin.x + W, origin.y + H}, true);
    dl->AddRectFilled(origin, {origin.x + W, origin.y + H},
                      IM_COL32(6, 7, 12, 255));

    const float cx = origin.x + W * 0.5f, cy = origin.y + H * 0.5f;
    float maxext = 1.0f;
    for (const auto& st : cluster_)
        for (const auto& q : st.track)
            maxext = std::max({maxext, std::abs(q[0]), std::abs(q[1])});
    const float scale = 0.46f * std::min(W, H) / maxext * clu_zoom_;

    // Sky frame: x = dRA (East, LEFT on screen), y = dDec (North, up),
    // z = away from the observer. yaw rotates about North, pitch about
    // East; yaw = pitch = 0 is exactly the view from Earth.
    const float cyw = std::cos(clu_yaw_), syw = std::sin(clu_yaw_);
    const float cpt = std::cos(clu_pitch_), spt = std::sin(clu_pitch_);
    auto project = [&](float x, float y, float z) {
        const float X = x * cyw + z * syw;
        const float Zr = -x * syw + z * cyw;
        const float Y = y * cpt - Zr * spt;
        return ImVec2(cx - X * scale, cy - Y * scale);
    };

    // orientation gizmo (bottom left): E / N / line of sight
    {
        const ImVec2 gz{origin.x + 34.0f, origin.y + H - 30.0f};
        auto tip = [&](float x, float y, float z) {
            const float X = x * cyw + z * syw;
            const float Zr = -x * syw + z * cyw;
            const float Y = y * cpt - Zr * spt;
            return ImVec2(gz.x - X * 22.0f, gz.y - Y * 22.0f);
        };
        dl->AddLine(gz, tip(1, 0, 0), IM_COL32(230, 140, 140, 200), 1.5f);
        dl->AddText(tip(1.25f, 0, 0), IM_COL32(230, 140, 140, 220), "E");
        dl->AddLine(gz, tip(0, 1, 0), IM_COL32(140, 230, 150, 200), 1.5f);
        dl->AddText(tip(0, 1.25f, 0), IM_COL32(140, 230, 150, 220), "N");
        dl->AddLine(gz, tip(0, 0, 1), IM_COL32(150, 170, 255, 200), 1.5f);
        dl->AddText(tip(0, 0, 1.35f), IM_COL32(150, 170, 255, 220), "los");
    }

    // Sgr A* at the origin
    {
        const ImVec2 c = project(0, 0, 0);
        dl->AddCircleFilled(c, 3.0f, IM_COL32(255, 255, 255, 230));
        dl->AddCircle(c, 6.0f, IM_COL32(255, 255, 255, 90));
        dl->AddText({c.x + 8, c.y - 6}, IM_COL32(255, 255, 255, 170),
                    "Sgr A*");
    }

    for (auto& st : cluster_) {
        // orbit line (full integrated GR track — S2's pericentre advance
        // draws its rosette by itself)
        const uint32_t lc = (st.line_col & 0x00ffffffu) | (110u << 24);
        ImVec2 prev{};
        bool first = true;
        for (const auto& q : st.track) {
            const ImVec2 s = project(q[0], q[1], q[2]);
            if (!first) dl->AddLine(prev, s, lc, 1.2f);
            prev = s;
            first = false;
        }

        // current state from the SAME integrated geodesic
        const double t = std::clamp(clu_epoch_, st.t0 + 1e-3, st.t1 - 1e-3);
        const TrajectorySample smp = st.model.sample_at(t);
        const float px = static_cast<float>(smp.pos_M[0]);
        const float py = static_cast<float>(smp.pos_M[1]);
        const float pz = static_cast<float>(smp.pos_M[2]);

        // trail (recent world positions; cleared on scrubs/wraps)
        if (std::abs(t - st.trail_last_t) > st.period_yr * 0.05)
            st.trail.clear();
        if (clu_play_ && t != st.trail_last_t) {
            st.trail.push_back({px, py, pz});
            if (st.trail.size() > 220)
                st.trail.erase(st.trail.begin());
            st.trail_last_t = t;
        }
        for (size_t i = 1; i < st.trail.size(); ++i) {
            const float a = 150.0f * static_cast<float>(i) /
                            static_cast<float>(st.trail.size());
            const uint32_t tc = (st.line_col & 0x00ffffffu) |
                                (static_cast<uint32_t>(a) << 24);
            dl->AddLine(project(st.trail[i - 1][0], st.trail[i - 1][1],
                                st.trail[i - 1][2]),
                        project(st.trail[i][0], st.trail[i][1],
                                st.trail[i][2]),
                        tc, 2.0f);
        }

        // ---- the shining ball: colour = EXACT line-of-sight redshift ----
        const StarShade sh = star_shade(smp.redshift);
        const ImVec2 s = project(px, py, pz);
        const float rad = 4.0f + 2.0f * std::min(1.0f, std::abs(sh.u));
        auto col = [&](float a) { return star_col(sh, a); };
        dl->AddCircleFilled(s, rad * 3.1f, col(28.0f));
        dl->AddCircleFilled(s, rad * 1.8f, col(76.0f));
        dl->AddCircleFilled(s, rad, col(255.0f));
        dl->AddCircleFilled(s, rad * 0.45f,
                            IM_COL32(255, 255, 255, 210));
        dl->AddText({s.x + rad + 4, s.y - 7},
                    (st.line_col & 0x00ffffffu) | (220u << 24),
                    st.name.c_str());
    }
    dl->PopClipRect();

    ImGui::TextDisabled(
        "tracks: full-GR geodesics at the joint cluster fit (M = %.2fe6 "
        "Msun)",
        clu_M_ / 1e6);
    ImGui::TextDisabled(
        "ball colour = LOS redshift: white (approaching) - gold - bronze "
        "(receding)  |  drag: rotate, wheel: zoom, double-click: Earth view");
    ImGui::End();
}

void SimHUD::draw_scope_overlay() {
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos({ds.x - 340, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({320, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("PHOTON RING SCOPE - live");
    ImGui::TextWrapped("Narrow-field view of the shadow edge: the bright thin "
                       "arc hugging the silhouette is the n=1 sub-ring.");
    ImGui::TextDisabled("fov %.2f rad, r_cam %.0f M  |  Esc: menu",
                        app_->live_.fov, app_->live_.r_cam);
    ImGui::End();
}

void SimHUD::draw_controls(double fps) {
    auto& lp = app_->live_;
    ImGui::SetNextWindowPos({12, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({330, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("SGR A* - controls");
    ImGui::Text("%.1f fps  |  %ux%u internal  |  fp32 live path", fps,
                app_->target_width(), app_->target_height());
    ImGui::Separator();

    int preset = lp.preset;
    if (ImGui::Combo("preset", &preset,
                     [](void*, int i) { return kDiskPresets[i].name; }, nullptr,
                     static_cast<int>(kDiskPresets.size()))) {
        app_->apply_preset(preset);
    }
    ImGui::SliderFloat("a* (spin)", &lp.a_star, 0.0f, 0.998f, "%.3f");
    ImGui::SliderFloat("inclination [deg]", &lp.incl_deg, 1.0f, 179.0f, "%.1f");
    // log range out to 1e6 M: true-scale S-star viewing needs the camera
    // beyond their orbits (S8/S1 reach ~1.5e5 M at apoapsis)
    ImGui::SliderFloat("camera r [M]", &lp.r_cam, 15.0f, 1000000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("fov [rad]", &lp.fov, 0.2f, 1.4f, "%.2f");
    ImGui::SliderFloat("disk r_out [M]", &lp.disk_r_out, 8.0f, 30.0f, "%.1f");
    ImGui::SliderFloat("exposure [EV]", &lp.exposure_ev, -6.0f, 6.0f, "%.1f");
    if (lp.t_peak_from_mdot) {
        ImGui::SliderFloat("accretion [Msun/yr]", &lp.mdot_msun_yr,
                           static_cast<float>(constants::sgrA_mdot_min_msun_yr),
                           static_cast<float>(constants::sgrA_mdot_max_msun_yr),
                           "%.1e", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Sgr A*'s accretion rate is bounded by the Faraday\n"
                "rotation of its 230 GHz light: 2e-9 .. 2e-7 Msun/yr\n"
                "(Marrone et al. 2007). The disk temperature is\n"
                "COMPUTED from it (Novikov-Thorne flux, sigma T^4 = F)\n"
                "- a hypothetical thin disk at the real feeding rate;\n"
                "the real flow is hot and geometrically thick.");
        ImGui::Text("T_peak = %.0f K (computed)", app_->effective_t_peak());
    } else {
        ImGui::SliderFloat("T_peak [K] (override)", &lp.t_peak_k, 2000.0f,
                           30000.0f, "%.0f");
    }
    ImGui::Checkbox("T_peak from accretion rate", &lp.t_peak_from_mdot);
    ImGui::SliderFloat("turbulence", &lp.turb, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Artistic filament layer sheared by the real\n"
                          "Keplerian rotation. 0 = pure Page-Thorne\n"
                          "(the physics is identical either way).");
    ImGui::SliderFloat("rotation [x real]", &lp.rot_speed, 0.0f, 1000.0f,
                       "%.1f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Filament advection speed in multiples of real\n"
                          "Sgr A* time. At 1.0 the innermost orbit takes\n"
                          "~8.5 minutes of wall clock - a real black hole\n"
                          "this heavy looks almost frozen. Crank it up to\n"
                          "watch the shear.");
    ImGui::Separator();
    if (ImGui::Checkbox("auto resolution (target 13 fps)", &lp.auto_res) &&
        !lp.auto_res)
        app_->request_target_rebuild();
    if (ImGui::SliderFloat("render scale", &lp.render_scale, 0.25f, 2.0f,
                           "%.2f"))
        app_->request_target_rebuild();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Internal render resolution vs window size.\n"
                          "Above 1.0 = supersampled anti-aliasing.\n"
                          "Auto mode adjusts this to hold ~13 fps.");
    ImGui::Checkbox("black background", &lp.black_sky);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Hide the starfield backdrop: fully black sky.\n"
                          "Display only - the disk and lensing are\n"
                          "unchanged.");
    ImGui::SameLine();
    ImGui::Checkbox("thin disk", &lp.show_disk);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Hide the (hypothetical) thin disk to see the\n"
                          "lensing of the background sky alone.");
    if (ImGui::CollapsingHeader("Display (camera, not physics)",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* ops[] = {"linear (science)", "AgX filmic", "ACES filmic"};
        ImGui::Combo("tone curve", &lp.tonemap, ops, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How radiance becomes screen brightness. Filmic\n"
                              "curves roll highlights off to white instead of\n"
                              "clipping; 'linear' is the exact transform the\n"
                              "CPU reference uses (Science Mode forces it).");
        ImGui::SliderFloat("bloom", &lp.bloom, 0.0f, 0.30f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lens glare around bright regions (a camera\n"
                              "effect, energy-preserving). Off in Science Mode.");
        ImGui::SliderFloat("white balance [K]", &lp.wb_kelvin, 4000.0f,
                           12000.0f, "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Camera white point: a blackbody at this\n"
                              "temperature renders neutral grey. Higher\n"
                              "values make the scene warmer.");
        ImGui::SliderFloat("saturation", &lp.saturation, 0.0f, 1.5f, "%.2f");
        ImGui::Checkbox("progressive AA", &lp.accumulate);
        ImGui::SameLine();
        ImGui::TextDisabled("%u spp", app_->accumulated_samples());
        ImGui::Checkbox("celestial grid", &lp.show_grid);
        ImGui::SliderFloat("sky [EV]", &lp.sky_ev, -6.0f, 4.0f, "%.1f");
        ImGui::Checkbox("real Milky Way", &lp.use_sky_map);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "NASA/GSFC SVS Deep Star Maps 2020 (1.7 billion stars;\n"
                "Gaia DR2: ESA/Gaia/DPAC; Hipparcos-2, Tycho-2), lensed\n"
                "by Sgr A*. It is the sky as seen from EARTH used as the\n"
                "distant background - from the Galactic Centre nearby\n"
                "stars would lie elsewhere. Spin axis: i = 30 deg to our\n"
                "line of sight, position angle 0 (an assumption).");
        if (lp.science_mode)
            ImGui::TextDisabled("Science Mode: linear transform, no bloom");
    }
    if (ImGui::CollapsingHeader("Observer - whose eyes",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* obs[] = {"static (ZAMO, non-rotating frame)",
                             "orbiting at Keplerian Omega(r)",
                             "falling in from rest at infinity",
                             "hovering (at rest vs. the stars)",
                             "riding the focused S-star"};
        ImGui::Combo("observer", &lp.observer, obs, 5);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Who is holding the camera. Motion changes what you see:\n"
                "light ahead is aberrated forward, blueshifted and\n"
                "brightened (relativistic beaming); light behind is\n"
                "redshifted. ZAMO = the observer that co-rotates with\n"
                "the frame dragging and has zero angular momentum.\n"
                "Hovering needs a rocket and is impossible inside the\n"
                "ergosphere; orbiting at Omega(r) is a geodesic only in\n"
                "the equatorial plane. Riding a star uses that star's\n"
                "velocity from its full-GR orbit (focus it and tick\n"
                "'ride the star').");
        if (lp.observer == static_cast<int>(Observer::RideStar) && !lp.ride_v_valid)
            ImGui::TextColored({1.0f, 0.75f, 0.35f, 1.0f},
                               "focus a star and tick 'ride the star' -"
                               " showing the ZAMO view");
        const ObserverState os = app_->observer_state();
        if (!os.valid)
            ImGui::TextColored({1.0f, 0.45f, 0.35f, 1.0f},
                               "this observer would need v >= c here -"
                               " showing the ZAMO view");
        const double tg = constants::sgrA_tg_s;        // GM/c^3 [s]
        const double rg_km = constants::sgrA_rg_m * 1e-3;
        ImGui::Text("speed vs local ZAMO  %.4f c   (gamma %.4f)", os.v, os.gamma);
        ImGui::Text("clock rate dtau/dt   %.6f", os.dtau_dt);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Seconds of the observer's own time per second\n"
                              "of far-away (Earth-frame) time: gravitational\n"
                              "and velocity time dilation together, 1/u^t.");
        ImGui::Text("lapse alpha          %.6f", os.alpha);
        ImGui::Text("frame dragging omega %.3e /M = %.3e rad/s", os.omega,
                    os.omega / tg);
        if (os.omega > 0.0)
            ImGui::Text("  (one ZAMO revolution every %.3g h)",
                        2.0 * 3.14159265358979 / (os.omega / tg) / 3600.0);
        const double a = lp.a_star, r = lp.r_cam;
        const double rp = 1.0 + std::sqrt(std::max(0.0, 1.0 - a * a));
        const KerrMetric km(1.0, std::clamp(static_cast<double>(a), -0.9999, 0.9999));
        ImGui::Text("r_cam %.1f M = %.4g km   (r+ %.3f, r_E here %.3f M)", r,
                    r * rg_km, rp, os.r_ergo);
        ImGui::Text("photon orbits %.3f / %.3f M, ISCO %.3f M",
                    km.r_photon(true), km.r_photon(false), km.r_isco(true));
        // circular equatorial prograde orbit at this radius (BPT72)
        const double sr = std::sqrt(r);
        const double den = r * sr - 3.0 * sr + 2.0 * a;
        if (den > 0.0) {
            const double ut = (r * sr + a) / (std::sqrt(sr) * std::sqrt(r) * std::sqrt(den));
            ImGui::Text("orbit at this r: dt/dtau = %.5f (1 h there = %.4g h far away)",
                        ut, ut);
        } else {
            ImGui::TextDisabled("no circular orbit at this radius (inside the photon orbit)");
        }
        ImGui::Checkbox("shift the sky by g (Doppler + gravity)", &lp.sky_shift);
        ImGui::Checkbox("light that visited the ergoregion", &lp.show_ergo);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Tint the light that passed INSIDE the ergosphere\n"
                              "(r < r_E, where nothing can stay at rest\n"
                              "relative to the stars) and still escaped: the\n"
                              "frame-dragged crescent on the prograde side.");
    }
    if (ImGui::CollapsingHeader("Band - what kind of camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* bands[] = {"visible light (hypothetical thin disk)",
                               "230 GHz radio (the real hot flow, EHT band)",
                               "2.2 um near-infrared (flares only)"};
        ImGui::Combo("band", &lp.band, bands, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Sgr A* does not have the bright thin disk of the visible view:\n"
                "at its measured accretion rate such a disk would outshine it\n"
                "by 12-1200x. Its real flow is a hot, thin plasma seen best at\n"
                "230 GHz - what the Event Horizon Telescope imaged. That band\n"
                "traces thermal synchrotron emission through the plasma with\n"
                "absorption (shown as brightness temperature, false colour).\n"
                "At 2.2 um only the flares show (switch the hot spot on).");
        if (lp.band == 1) {
            ImGui::Text("n0 = %.3g cm^-3 %s", lp.riaf_n0,
                        lp.riaf_fitted ? "(fitted: 2.4 Jy at 230 GHz)" : "(NOT normalised)");
            ImGui::TextDisabled("T0 = %.2g K, beta = %.0f; colour: Tb, 1 = 5e10 K", lp.riaf_T0,
                                lp.riaf_beta);
        }
    }
    if (ImGui::CollapsingHeader("Flare - orbiting hot spot (GRAVITY 2018)")) {
        ImGui::Checkbox("hot spot", &lp.flare);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The model GRAVITY fitted to Sgr A*'s 2018 near-infrared\n"
                "flares: a compact Gaussian blob (size s) on a circular\n"
                "orbit. Shown in FALSE COLOUR (2.2 um synchrotron light\n"
                "has no visible colour). Slow light: every ray carries its\n"
                "own travel time, so the lensed image behind the hole is\n"
                "an echo of where the spot was minutes earlier.");
        const double tg = constants::sgrA_tg_s;
        const KerrMetric km(1.0, std::clamp(static_cast<double>(lp.a_star), -0.9999, 0.9999));
        const float r_isco = static_cast<float>(km.r_isco(true));
        ImGui::SliderFloat("orbit radius R [M]", &lp.flare_R, std::max(r_isco, 1.0f) + 0.05f,
                           20.0f, "%.2f");
        if (ImGui::SmallButton("GRAVITY 2018 fit (R = 7)")) lp.flare_R = 7.0f;
        ImGui::SameLine();
        if (ImGui::SmallButton("GRAVITY 2020 fit (R = 9)")) lp.flare_R = 9.0f;
        ImGui::SameLine();
        if (ImGui::SmallButton("period 40 min")) {
            // R whose prograde period 2pi(R^1.5 + a) t_g is 40 min
            const double P = 40.0 * 60.0 / tg;
            lp.flare_R = static_cast<float>(
                std::pow(P / (2.0 * 3.14159265358979) - lp.a_star, 2.0 / 3.0));
        }
        ImGui::SliderFloat("spot size s [M]", &lp.flare_sigma, 0.2f, 2.0f, "%.2f");
        ImGui::SliderFloat("spectral index alpha", &lp.flare_alpha, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("brightness", &lp.flare_amp, 0.05f, 20.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("time warp [M per s]", &lp.flare_speed, 0.0f, 300.0f, "%.0f");
        ImGui::Checkbox("play", &lp.flare_play);
        ImGui::SameLine();
        if (ImGui::SmallButton("restart")) lp.flare_t = 0.0f;
        const double Om = km.omega_circular(lp.flare_R);
        const double P = 2.0 * 3.14159265358979 / Om;
        ImGui::Text("period %.1f min (%.0f M); observing time %.1f min", P * tg / 60.0, P,
                    lp.flare_t * tg / 60.0);
        const auto v = observer_velocity(km, ObserverKind::Orbiting, lp.flare_R,
                                         0.5 * 3.14159265358979);
        ImGui::Text("orbital speed %.3f c (vs the local ZAMO)", std::abs(v[2]));
        if (lp.flare_R <= r_isco)
            ImGui::TextDisabled("inside the ISCO: no stable circular orbit");
    }
    ImGui::Checkbox("bolometric view (gT)^4", &lp.bolometric);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Show the disk as grey total (all-wavelength)\n"
                          "power (gT/6500 K)^4 instead of its visible\n"
                          "colour. This isolates the g^4 Doppler beaming;\n"
                          "the default view is what a camera would see\n"
                          "(blackbody at g*T through the CIE eye curves).");
    ImGui::Checkbox("S-stars in view", &lp.show_stars);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Draw S2/S1/S8/S12/S13 orbiting in the scene,\n"
                          "on their fitted full-GR geodesics, coloured by\n"
                          "line-of-sight redshift. The real orbits are\n"
                          "~24000 M across (the disk is ~20 M), so the\n"
                          "radius is compressed by the factor shown -\n"
                          "shape, phase, period and colour are unmodified.");
    if (lp.show_stars && cluster_available_) {
        double a_ref = 0.0;
        for (const auto& st : cluster_)
            if (st.name == "S2") a_ref = st.a_M;
        if (a_ref <= 0.0) a_ref = cluster_.front().a_M;
        ImGui::Checkbox("orbit paths", &lp.show_star_paths);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(96.0f);
        ImGui::SliderFloat("scale 1:N", &lp.star_scale_ratio, 1.0f, 2000.0f,
                           "%.0f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("1 = TRUE distance (the renderer is already\n"
                              "in real geometric units). Higher values pull\n"
                              "the orbits inward by that factor for an\n"
                              "overview - never rescales with the camera.");
        if (ImGui::SmallButton("1:1 true")) lp.star_scale_ratio = 1.0f;
        ImGui::SameLine();
        if (ImGui::SmallButton("1:100")) lp.star_scale_ratio = 100.0f;
        ImGui::SameLine();
        if (ImGui::SmallButton("1:1000")) lp.star_scale_ratio = 1000.0f;
        ImGui::SameLine();
        // Frame the cluster by moving the CAMERA, not by shrinking the
        // orbits: the honest way to see a 2.4e4 M orbit is to fly out
        // past it (the hole then really is a lensed dot).
        if (ImGui::SmallButton("fly out")) {
            const double rmax = clu_rmax_M_ / std::max(lp.star_scale_ratio,
                                                       1.0f);
            const double half = std::tan(0.5 * lp.fov);
            lp.r_cam = static_cast<float>(
                std::clamp(rmax / (0.62 * half), 15.0, 1000000.0));
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Move the camera out until the whole cluster\n"
                              "fits - orbits keep their true proportions.");
        if (a_ref > 0.0) {
            // 1 M = GM/c^2; a_AU printed straight from the fit geometry
            const double a_disp = a_ref / std::max(lp.star_scale_ratio, 1.0f);
            ImGui::TextDisabled("S2: a = %.0f M true", a_ref);
            ImGui::TextDisabled("  -> drawn at %.0f M  (disk r_out %.0f M)",
                                a_disp, lp.disk_r_out);
        }
        ImGui::SetNextItemWidth(126.0f);
        ImGui::SliderFloat("star time [x real]", &clu_speed_, 1.0e4f, 1.0e8f,
                           "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine();
        if (ImGui::SmallButton(clu_play_ ? "||" : ">")) clu_play_ = !clu_play_;
        ImGui::TextDisabled("epoch %.2f", clu_epoch_);
    }
    ImGui::Separator();
    ImGui::Checkbox("Science Mode (S2 orbit + fit)", &lp.science_mode);
    ImGui::TextDisabled("drag: orbit  |  wheel: zoom  |  keys 1-5: presets");
    ImGui::TextDisabled(app_->menu_enabled() ? "Esc: menu" : "Esc: quit");
    ImGui::End();
}

// Gate 21: the exotic-spacetime demo. Everything Sgr A* is switched off;
// the panel title, the banner and every tooltip say so.
void SimHUD::draw_exotic(double fps) {
    auto& lp = app_->live_;
    ImDrawList* bg = ImGui::GetBackgroundDrawList();
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const char* banner = "EXOTIC SPACETIME - NOT SGR A*  |  hypothetical Dneg wormhole";
    const ImVec2 tsz = ImGui::CalcTextSize(banner);
    bg->AddRectFilled({0.5f * (ds.x - tsz.x) - 10, ds.y - 40},
                      {0.5f * (ds.x + tsz.x) + 10, ds.y - 40 + tsz.y + 10},
                      IM_COL32(0, 0, 0, 150), 4.0f);
    bg->AddText({0.5f * (ds.x - tsz.x), ds.y - 35}, IM_COL32(255, 196, 120, 255), banner);

    ImGui::SetNextWindowPos({12, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({420, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("EXOTIC SPACETIME - NOT SGR A*");
    ImGui::Text("%.1f fps  |  %ux%u internal", fps, app_->target_width(),
                app_->target_height());
    ImGui::Separator();
    ImGui::TextWrapped(
        "The Double Negative wormhole of Interstellar (James, von Tunzelmann, "
        "Franklin & Thorne 2015): a hypothetical traversable tunnel to another "
        "universe. Nothing here describes Sgr A* - it is a test that the "
        "renderer handles a spacetime other than Kerr.");
    ImGui::Separator();
    ImGui::SliderFloat("camera l [rho]", &lp.wh_l, -60.0f, 60.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Proper distance from the throat (radius rho).\n"
                          "Negative: you are through, in the other universe.\n"
                          "Mouse wheel: fly along l.");
    float len = 2.0f * lp.wh_a;
    if (ImGui::SliderFloat("length 2a [rho]", &len, 0.001f, 4.0f, "%.3f",
                           ImGuiSliderFlags_Logarithmic))
        lp.wh_a = 0.5f * len;
    ImGui::SliderFloat("lens width W [rho]", &lp.wh_W, 0.01f, 2.0f, "%.3f",
                       ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Lensing width: how gradually the mouth flares out\n"
                          "(W = 1.42953 M). The film's wormhole: 2a = 0.01,\n"
                          "W = 0.05 (James et al. 2015).");
    if (ImGui::SmallButton("Interstellar (2a = 0.01, W = 0.05)")) {
        lp.wh_a = 0.005f;
        lp.wh_W = 0.05f;
    }
    ImGui::SliderFloat("fov [rad]", &lp.fov, 0.2f, 2.4f, "%.2f");
    ImGui::SliderFloat("exposure [EV]", &lp.exposure_ev, -6.0f, 6.0f, "%.1f");
    // the other universe's disk, computed: sin(alpha) = rho / r(l_c)
    sgr::DnegParams p;
    p.a = lp.wh_a;
    p.M = lp.wh_W / sgr::DnegWormhole::lensing_width_closed_form(1.0);
    const sgr::DnegWormhole w(p);
    const double rc = w.r(lp.wh_l);
    ImGui::TextWrapped("r(l) = %.3f rho;  other universe: %.2f deg radius", rc,
                std::asin(std::min(1.0, p.rho / rc)) * 180.0 / 3.14159265358979323846);
    ImGui::TextDisabled("(the edge is exact: sin(alpha) = rho / r(l))");
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("Other universe = our Milky Way map, turned around and "
                       "recoloured (illustrative: no one knows its sky).");
    ImGui::PopStyleColor();
    ImGui::Separator();
    if (ImGui::Button("back to Sgr A*")) app_->enter_simulation(false);
    ImGui::TextDisabled("drag: orbit  |  wheel: fly along l");
    ImGui::TextDisabled(app_->menu_enabled() ? "Esc: menu" : "Esc: quit");
    ImGui::End();
}

// Gate 24: falling into Sgr A*. Every readout is computed from the fp64
// worldline (sgr_physics/infall.hpp) at the current proper time.
void SimHUD::draw_flight(double fps) {
    auto& lp = app_->live_;
    const InfallTrajectory* tr = app_->flight_trajectory();
    ImGui::SetNextWindowPos({12, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({430, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("FALL INTO SGR A* - your own worldline");
    ImGui::Text("%.1f fps  |  %ux%u internal", fps, app_->target_width(),
                app_->target_height());
    if (!tr) {
        ImGui::TextDisabled("no trajectory");
        ImGui::End();
        return;
    }
    const KerrSchild& ks = tr->chart();
    const InfallSample s = tr->at(lp.flight_tau);
    const double rp = ks.r_plus(), rm = ks.r_minus();
    const double tg = constants::sgrA_tg_s;              // seconds per M of time
    const double rg_km = constants::sgrA_rg_m / 1000.0;  // km per M of length
    constexpr double kG0 = 9.80665;                      // standard gravity [m/s²]

    // ---- controls ------------------------------------------------------------
    if (ImGui::Button(lp.flight_play ? "pause" : "fall")) lp.flight_play = !lp.flight_play;
    ImGui::SameLine();
    if (ImGui::Button("restart")) {
        lp.flight_tau = 0.0;
        lp.flight_play = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170.0f);
    ImGui::SliderFloat("time warp [M/s]", &lp.flight_warp, 0.02f, 50.0f, "%.2f",
                       ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Your proper time per second of wall clock, in units of\n"
                          "M = GM/c^3 = %.1f s for Sgr A*. At this setting the\n"
                          "whole fall takes %.0f s of wall clock.", tg,
                          tr->tau_end() / lp.flight_warp);
    float tau = static_cast<float>(lp.flight_tau);
    if (ImGui::SliderFloat("proper time [M]", &tau, 0.0f, static_cast<float>(tr->tau_end()),
                           "%.3f"))
        lp.flight_tau = tau;
    if (ImGui::Button("look ahead (in)")) {
        lp.fwd_r = -1; lp.fwd_th = 0; lp.fwd_ph = 0;
        lp.rgt_r = 0;  lp.rgt_th = 0; lp.rgt_ph = 1;
        lp.up_r = 0;   lp.up_th = -1; lp.up_ph = 0;
    }
    ImGui::SameLine();
    if (ImGui::Button("look back (out)")) {
        lp.fwd_r = 1; lp.fwd_th = 0; lp.fwd_ph = 0;
        lp.rgt_r = 0; lp.rgt_th = 0; lp.rgt_ph = -1;
        lp.up_r = 0;  lp.up_th = -1; lp.up_ph = 0;
    }
    ImGui::SameLine();
    if (ImGui::Button("look along orbit")) {
        lp.fwd_r = 0; lp.fwd_th = 0; lp.fwd_ph = 1;
        lp.rgt_r = 1; lp.rgt_th = 0; lp.rgt_ph = 0;
        lp.up_r = 0;  lp.up_th = -1; lp.up_ph = 0;
    }
    ImGui::Checkbox("hypothetical thin disk", &lp.show_disk);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Sgr A* has no bright thin disk (its real flow is a hot,\n"
                          "thin plasma). This one is hypothetical, and up close its\n"
                          "Doppler-boosted light swamps the view.");
    if (ImGui::Checkbox("auto resolution (target 13 fps)", &lp.auto_res) && !lp.auto_res)
        app_->request_target_rebuild();
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::SliderFloat("render scale", &lp.render_scale, 0.25f, 2.0f, "%.2f"))
        app_->request_target_rebuild();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Internal render resolution vs window size. Pause the\n"
                          "fall and the image also refines itself (progressive\n"
                          "anti-aliasing) while the camera is still.");

    // ---- where and when ------------------------------------------------------
    ImGui::SeparatorText("where you are");
    ImGui::Text("r = %.3f M  =  %.3g million km", s.r, s.r * rg_km / 1e6);
    const double ct = std::cos(s.theta);
    const double r_ergo = 1.0 + std::sqrt(std::max(0.0, 1.0 - ks.a() * ks.a() * ct * ct));
    if (s.r > r_ergo)
        ImGui::TextDisabled("outside the ergosphere (r_E = %.2f M here)", r_ergo);
    else if (s.r > rp)
        ImGui::TextColored({1.0f, 0.75f, 0.35f, 1.0f},
                           "inside the ergosphere: nothing can stay at rest");
    else
        ImGui::TextColored({1.0f, 0.45f, 0.35f, 1.0f},
                           "inside the horizon: every future leads inward");
    ImGui::SeparatorText("time");
    ImGui::Text("your proper time   %.3f M  =  %.1f s", s.tau, s.tau * tg);
    if (s.r > rp && std::isfinite(s.t_bl))
        ImGui::Text("far-away time      %.1f M  =  %.2f h", s.t_bl, s.t_bl * tg / 3600.0);
    else
        ImGui::TextWrapped("far-away time: undefined - no signal you send can ever "
                           "reach Earth again.");
    if (s.tau < tr->tau_horizon())
        ImGui::Text("horizon in          %.3f M  =  %.1f s", tr->tau_horizon() - s.tau,
                    (tr->tau_horizon() - s.tau) * tg);
    else
        ImGui::Text("horizon crossed %.3f M (%.1f s) ago", s.tau - tr->tau_horizon(),
                    (s.tau - tr->tau_horizon()) * tg);
    ImGui::Text("end of the model in %.3f M  =  %.1f s", tr->tau_end() - s.tau,
                (tr->tau_end() - s.tau) * tg);

    // ---- light and tides --------------------------------------------------------
    const auto e = infall_tetrad(ks, s.X, s.U);
    {
        // light arriving from straight behind (outward): k = e0 − e1
        const Metric4<double> g = ks.metric(s.X);
        Vec4 k{}, kc{};
        for (int m = 0; m < 4; ++m) k[m] = e[0][m] - e[1][m];
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n) kc[m] += g(m, n) * k[n];
        const double ratio = 1.0 / (-kc[0]);   // ν_seen / ν_at_infinity
        ImGui::SeparatorText("light");
        ImGui::Text("sky straight behind you: frequency x %.3f (%s)", ratio,
                    ratio > 1.0 ? "blueshifted" : "redshifted");
    }
    {
        const TidalField t = tidal_field(ks, s.X, e);
        const double L = 2.0;   // a 2 m body
        const double stretch = -t.eig[0] * L / (tg * tg), squeeze = t.eig[2] * L / (tg * tg);
        ImGui::SeparatorText("tides across a 2 m body");
        ImGui::Text("stretch %.3g m/s^2 (%.2g g)   squeeze %.3g m/s^2", stretch,
                    stretch / kG0, squeeze);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Eigenvalues of the tidal tensor in your frame (exact,\n"
                              "from the Riemann tensor). They grow like M/r^3, so a\n"
                              "black hole as heavy as Sgr A* has a gentle horizon.");
    }
    ImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("The trip ends just above the inner horizon r- = %.2f M: inside it "
                       "the classical Kerr interior is unstable (mass inflation), so "
                       "nothing beyond is modelled. The sky is the Milky Way as seen from "
                       "Earth, used as the distant background.", rm);
    ImGui::PopStyleColor();

    // ---- launch -------------------------------------------------------------------
    if (ImGui::CollapsingHeader("launch")) {
        ImGui::SliderFloat("start r [M]", &lp.flight_r0, 3.0f, 300.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("start polar angle [deg]", &lp.flight_theta, 15.0f, 90.0f, "%.0f");
        ImGui::Checkbox("start at rest (else: falling from infinity)", &lp.flight_rest);
        if (ImGui::Button("launch")) {
            app_->start_flight();
            lp.flight_play = true;
        }
    }
    if (ImGui::Button("back to the exterior views")) app_->enter_simulation(false);
    ImGui::TextDisabled(app_->menu_enabled() ? "Esc: menu" : "Esc: quit");
    ImGui::End();
}

void SimHUD::draw_science() {
    ImGui::SetNextWindowPos({360, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({430, 470}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Science Mode - S2 orbit fit");
    if (!science_available_) {
        ImGui::TextWrapped("Phase 4 artefacts not found. Run fit_s2_tool to "
                           "generate validation/phase4/*.");
        ImGui::End();
        return;
    }
    ImGui::TextUnformatted(fit_summary_.c_str());
    ImGui::TextDisabled("model: full GR geodesic | data: Gillessen+2017 (VizieR)");

    // Orbit view (RA/Dec plane, East left).
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float W = ImGui::GetContentRegionAvail().x;
    const float H = W;
    const float cx = origin.x + W * 0.5f, cy = origin.y + H * 0.55f;
    const float scale = W / 420.0f; // mas â†’ px (S2 spans ~Â±200 mas)
    auto to_px = [&](float ra, float dec) {
        return ImVec2(cx - ra * scale, cy - dec * scale); // East left, North up
    };
    dl->AddRectFilled(origin, {origin.x + W, origin.y + H},
                      IM_COL32(10, 10, 16, 255), 4.0f);
    for (size_t i = 1; i < model_.ra.size(); ++i)
        dl->AddLine(to_px(model_.ra[i - 1], model_.dec[i - 1]),
                    to_px(model_.ra[i], model_.dec[i]),
                    IM_COL32(120, 160, 255, 200), 1.4f);
    for (size_t i = 0; i < ast_.ra.size(); ++i)
        dl->AddCircleFilled(to_px(ast_.ra[i], ast_.dec[i]), 2.0f,
                            IM_COL32(255, 200, 80, 220));
    // Sgr A*
    dl->AddCircleFilled(to_px(0, 0), 3.5f, IM_COL32(255, 60, 60, 255));
    dl->AddText({origin.x + 6, origin.y + 6}, IM_COL32(200, 200, 210, 255),
                "S2 orbit: model (blue) vs astrometry (gold), Sgr A* (red)");
    ImGui::Dummy({W, H + 8});
    ImGui::End();
}

void SimHUD::record(VkCommandBuffer cmd, uint32_t sc_index) {
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = render_pass_;
    rbi.framebuffer = framebuffers_[sc_index];
    rbi.renderArea = {{0, 0}, ctx_->swapchain_extent()};
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
}

SimHUD::~SimHUD() {
    ctx_->wait_idle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    destroy_framebuffers();
    if (imgui_pool_) vkDestroyDescriptorPool(ctx_->device(), imgui_pool_, nullptr);
    if (render_pass_) vkDestroyRenderPass(ctx_->device(), render_pass_, nullptr);
}

} // namespace sgr


