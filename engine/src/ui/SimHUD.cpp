#include "ui/SimHUD.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "app.hpp"
#include "modes/MenuMode.hpp"
#include "vk/context.hpp"

namespace sgr {

using vk::check;

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
        draw_controls(fps);
        if (app_->live_.science_mode) draw_science();
        if (app_->live_.scope_mode) draw_scope_overlay();
    }
    ImGui::Render();
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


