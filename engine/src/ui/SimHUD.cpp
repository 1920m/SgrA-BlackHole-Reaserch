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
    // largest apoapsis actually reached, straight off the integrated
    // tracks (no assumed ellipse) — drives the in-scene "fit to view"
    for (const auto& st : cluster_)
        for (const auto& q : st.track)
            clu_rmax_M_ = std::max(
                clu_rmax_M_,
                static_cast<double>(std::sqrt(q[0] * q[0] + q[1] * q[1] +
                                              q[2] * q[2])));
    cluster_available_ = true;
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
    } else {
        if (app_->live_.show_stars || app_->live_.science_mode)
            update_cluster_time();
        // background draw list: over the rendered scene, under the panels
        if (app_->live_.show_stars) draw_stars_overlay();
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

    double a_ref = 0.0;
    for (const auto& st : cluster_)
        if (st.name == "S2") a_ref = st.a_M;
    if (a_ref <= 0.0) a_ref = cluster_.front().a_M;
    if (a_ref <= 0.0) return;
    const double k = static_cast<double>(lp.star_scale_M) / a_ref;

    // camera frame: position + the (ê_r, ê_θ, ê_φ) triad the shader uses
    const double th0 = lp.incl_deg * 3.14159265358979 / 180.0;
    const double ph0 = lp.cam_phi;
    const double s0 = std::sin(th0), c0 = std::cos(th0);
    const double sp = std::sin(ph0), cp = std::cos(ph0);
    const double er[3] = {s0 * cp, s0 * sp, c0};
    const double eth[3] = {c0 * cp, c0 * sp, -s0};
    const double eph[3] = {-sp, cp, 0.0};
    const double cam[3] = {lp.r_cam * er[0], lp.r_cam * er[1],
                           lp.r_cam * er[2]};
    const double thalf = std::tan(0.5 * lp.fov);
    const double aspect = static_cast<double>(H) / static_cast<double>(W);
    // Schwarzschild critical impact parameter: the display occultation
    // test for a star passing behind the silhouette.
    const double b_crit = 3.0 * std::sqrt(3.0);

    struct Proj { ImVec2 p; float vis; bool ok; };
    auto project = [&](double x, double y, double z) {
        Proj o{ImVec2(0, 0), 1.0f, false};
        const double d[3] = {x - cam[0], y - cam[1], z - cam[2]};
        const double dr = d[0] * er[0] + d[1] * er[1] + d[2] * er[2];
        if (dr >= -1e-9) return o;                    // behind the camera
        const double dth = d[0] * eth[0] + d[1] * eth[1] + d[2] * eth[2];
        const double dph = d[0] * eph[0] + d[1] * eph[1] + d[2] * eph[2];
        const double sx = (-dph / dr) / thalf;
        const double sy = (dth / dr) / (thalf * aspect);
        if (std::abs(sx) > 6.0 || std::abs(sy) > 6.0) return o; // far off
        o.p = ImVec2(static_cast<float>((sx + 1.0) * 0.5 * W),
                     static_cast<float>((1.0 - sy) * 0.5 * H));
        // occultation: closest approach of the camera->star segment to
        // the hole, faded rather than switched so it does not pop
        const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (len > 1e-9) {
            const double u[3] = {d[0] / len, d[1] / len, d[2] / len};
            const double tstar = -(cam[0] * u[0] + cam[1] * u[1] +
                                   cam[2] * u[2]);
            if (tstar > 0.0 && tstar < len) {
                const double q[3] = {cam[0] + tstar * u[0],
                                     cam[1] + tstar * u[1],
                                     cam[2] + tstar * u[2]};
                const double b = std::sqrt(q[0] * q[0] + q[1] * q[1] +
                                           q[2] * q[2]);
                o.vis = static_cast<float>(
                    std::clamp((b - b_crit) / (0.45 * b_crit), 0.0, 1.0));
            }
        }
        o.ok = true;
        return o;
    };

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    for (auto& st : cluster_) {
        // orbit line (the full integrated GR track — S2's pericentre
        // advance draws its own rosette)
        Proj prev{};
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

        const double t = std::clamp(clu_epoch_, st.t0 + 1e-3, st.t1 - 1e-3);
        const TrajectorySample smp = st.model.sample_at(t);
        const Proj cur = project(smp.pos_M[0] * k, smp.pos_M[1] * k,
                                 smp.pos_M[2] * k);
        if (!cur.ok || cur.vis <= 0.02f) continue;
        const StarShade sh = star_shade(smp.redshift);
        const float rad = 3.2f + 2.2f * std::min(1.0f, std::abs(sh.u));
        dl->AddCircleFilled(cur.p, rad * 4.0f, star_col(sh, 22.0f * cur.vis));
        dl->AddCircleFilled(cur.p, rad * 2.1f, star_col(sh, 60.0f * cur.vis));
        dl->AddCircleFilled(cur.p, rad, star_col(sh, 255.0f * cur.vis));
        dl->AddCircleFilled(cur.p, rad * 0.45f,
                            IM_COL32(255, 255, 255,
                                     static_cast<int>(220 * cur.vis)));
        dl->AddText({cur.p.x + rad + 5.0f, cur.p.y - 7.0f},
                    (st.line_col & 0x00ffffffu) |
                        (static_cast<uint32_t>(200 * cur.vis) << 24),
                    st.name.c_str());
    }
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
    ImGui::SliderFloat("camera r [M]", &lp.r_cam, 15.0f, 400.0f, "%.0f");
    ImGui::SliderFloat("fov [rad]", &lp.fov, 0.2f, 1.4f, "%.2f");
    ImGui::SliderFloat("disk r_out [M]", &lp.disk_r_out, 8.0f, 30.0f, "%.1f");
    ImGui::SliderFloat("exposure [EV]", &lp.exposure_ev, -3.0f, 4.0f, "%.1f");
    ImGui::SliderFloat("T_peak [K]", &lp.t_peak_k, 2000.0f, 20000.0f, "%.0f");
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
        // Fit the whole cluster into the current frame: the outermost
        // apoapsis lands at ~60 % of the frame half-width at the hole's
        // distance (r_cam * tan(fov/2)) — the margin absorbs perspective,
        // since orbit arcs on the camera side project wider than that.
        auto fit = [&] {
            if (clu_rmax_M_ <= 0.0 || a_ref <= 0.0) return;
            const double half = lp.r_cam * std::tan(0.5 * lp.fov);
            lp.star_scale_M = static_cast<float>(
                std::clamp(a_ref * 0.60 * half / clu_rmax_M_, 1.0, 300.0));
        };
        if (!stars_fitted_) {
            fit();
            stars_fitted_ = true;
        }
        ImGui::SetNextItemWidth(126.0f);
        ImGui::SliderFloat("S2 orbit [M]", &lp.star_scale_M, 1.0f, 300.0f,
                           "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine();
        if (ImGui::SmallButton("fit")) fit();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Scale the cluster so every orbit fits the\n"
                              "current camera framing.");
        if (a_ref > 0.0)
            ImGui::TextDisabled("scale 1 : %.0f (true a_S2 = %.0f M)",
                                a_ref / std::max(lp.star_scale_M, 1e-3f),
                                a_ref);
        ImGui::SetNextItemWidth(126.0f);
        ImGui::SliderFloat("star time [x real]", &clu_speed_, 1.0e4f, 1.0e8f,
                           "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine();
        if (ImGui::SmallButton(clu_play_ ? "||" : ">")) clu_play_ = !clu_play_;
        ImGui::TextDisabled("epoch %.2f", clu_epoch_);
    }
    ImGui::Separator();
    ImGui::Checkbox("Science Mode (S2 orbit + fit)", &lp.science_mode);
    ImGui::TextDisabled("drag: orbit  |  wheel: zoom  |  keys 1-6: presets");
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


