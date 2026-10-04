#include "app.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <nlohmann/json.hpp>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/observers.hpp"
#include "Hotspot.hpp"
#include "Wormhole.hpp"
#include "ui/SimHUD.hpp"

namespace sgr {

using vk::check;
using vk::Context;

namespace {
// Compute → compute memory dependency (RAW and WAR between post passes and
// across frames/submissions on the shared radiance accumulator).
void compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}
} // namespace

App::App(const AppOptions& opts) : opts_(opts) {
    ctx_ = std::make_unique<Context>(opts.width, opts.height,
                                     "SGR A* — scientific simulator",
                                     opts.validation);
    bindless_ = std::make_unique<vk::BindlessTable>(*ctx_);
    {
        // Gate 17 per-frame observer parameters (binding 2). Device-local;
        // contents written with vkCmdUpdateBuffer inside each frame's
        // command buffer, so frames in flight never race on it.
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sizeof(FrameParams);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        check(vmaCreateBuffer(ctx_->allocator(), &bci, &aci, &frame_buf_,
                              &frame_buf_alloc_, nullptr),
              "vmaCreateBuffer(frame params)");
        bindless_->write_storage_buffer(frame_buf_, sizeof(FrameParams));
    }
    pipeline_ = std::make_unique<vk::ComputePipeline>(
        *ctx_, bindless_->layout(),
        std::string(SGR_SHADER_DIR) + "/" + opts.shader + ".comp.glsl",
        sizeof(PushConstants));
    if (opts.shader == "disk") {
        // Gate 14 display chain: radiance accumulator → bloom → tone map.
        const std::string dir = std::string(SGR_SHADER_DIR) + "/";
        bloom_down_ = std::make_unique<vk::ComputePipeline>(
            *ctx_, bindless_->layout(), dir + "bloom_down.comp.glsl",
            static_cast<uint32_t>(sizeof(BloomDownPC)));
        bloom_up_ = std::make_unique<vk::ComputePipeline>(
            *ctx_, bindless_->layout(), dir + "bloom_up.comp.glsl",
            static_cast<uint32_t>(sizeof(BloomUpPC)));
        tonemap_ = std::make_unique<vk::ComputePipeline>(
            *ctx_, bindless_->layout(), dir + "tonemap.comp.glsl",
            static_cast<uint32_t>(sizeof(TonemapPC)));
    }
    live_.render_scale = std::clamp(opts_.render_scale, 0.25f, 2.0f);
    create_render_target();

    // live parameters: preset defaults → per-shader defaults → CLI overrides
    apply_preset(opts_.preset); // sets r_cam too — must precede the overrides
    const bool live_path = opts_.shader == "disk";
    live_.fov = live_path ? 0.70f : 0.032f;
    if (!live_path) {
        live_.r_cam = 1000.0f;
        live_.a_star = 0.94f;
        live_.incl_deg = 80.0f;
        live_.disk_r_out = 18.0f;
        live_.exposure_ev = 0.0f;
        live_.t_peak_k = 6300.0f;
    }
    if (opts_.a_star >= 0.0f) live_.a_star = opts_.a_star;
    if (opts_.incl_deg >= 0.0f) live_.incl_deg = opts_.incl_deg;
    if (opts_.fov > 0.0f) live_.fov = opts_.fov;
    if (opts_.r_cam > 0.0f) live_.r_cam = opts_.r_cam;
    if (opts_.disk_r_out > 0.0f) live_.disk_r_out = opts_.disk_r_out;
    if (opts_.exposure_ev > -900.0f) live_.exposure_ev = opts_.exposure_ev;
    if (opts_.t_peak_k > 0.0f) {
        live_.t_peak_k = opts_.t_peak_k;   // explicit override of the
        live_.t_peak_from_mdot = false;    // Mdot-derived temperature
    }
    if (!live_path) live_.t_peak_from_mdot = false; // validation shaders
    if (opts_.rot_speed >= 0.0f) live_.rot_speed = opts_.rot_speed;
    // design range is 0-1 (the shader clamps too; beyond 1 the filament
    // mix() extrapolates into harsh negative-depth lanes)
    if (opts_.turb >= 0.0f) live_.turb = std::min(opts_.turb, 1.0f);
    live_.black_sky = opts_.black_sky;
    live_.bolometric = opts_.bolometric;
    live_.show_grid = opts_.grid;
    if (opts_.bloom >= 0.0f) live_.bloom = opts_.bloom;
    if (opts_.display == "linear") live_.tonemap = 0;
    else if (opts_.display == "agx") live_.tonemap = 1;
    else if (opts_.display == "aces") live_.tonemap = 2;
    else if (!opts_.display.empty())
        std::fprintf(stderr, "[app] unknown --display '%s' (linear|agx|aces)\n",
                     opts_.display.c_str());
    live_.show_stars = opts_.show_stars;
    live_.show_star_paths = opts_.star_paths;
    live_.focus_ride = opts_.focus_ride;
    live_.star_scale_ratio = std::max(opts_.star_scale, 1.0f);

    live_.science_mode = opts_.science;

    // Scroll = camera zoom (installed before ImGui's backend so its handler
    // chains to ours; ImGui consumes the event first when a window is
    // hovered — see the WantCaptureMouse guard in the callback).
    glfwSetWindowUserPointer(ctx_->window(), this);
    glfwSetScrollCallback(ctx_->window(), [](GLFWwindow* w, double, double dy) {
        auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
        if (!app || app->opts_.no_mouse || app->mode_ != UiMode::Simulation)
            return;
        if (app->hud_ && app->hud_->wants_mouse()) return;
        // Upper limit 1e6 M: the S-star orbits reach ~1.5e5 M, so viewing
        // the cluster at TRUE scale means flying the camera out past
        // them. The fp32 live path stays clean out there (steps are
        // logarithmic in r; the hole simply becomes a lensed dot).
        if (app->live_.wormhole) {
            // exotic demo: the wheel flies along ℓ, through the throat
            const float l = app->live_.wh_l;
            app->live_.wh_l = std::clamp(
                l - 0.08f * static_cast<float>(dy) * std::max(std::abs(l), 0.5f),
                -60.0f, 60.0f);
            return;
        }
        if (app->live_.focus_index > 0) {
            // focused on a star: zoom means distance FROM THE STAR
            app->live_.focus_dist = std::clamp(
                app->live_.focus_dist * std::pow(0.90f, static_cast<float>(dy)),
                0.05f, 20000.0f);
            return;
        }
        app->live_.r_cam = std::clamp(
            app->live_.r_cam * std::pow(0.90f, static_cast<float>(dy)),
            15.0f, 1000000.0f);
    });

    if (opts_.shader == "disk") {
        load_luts();
        if (!opts_.procedural_sky) load_sky();
    }
    if (opts_.sky_ev > -900.0f) live_.sky_ev = opts_.sky_ev;
    live_.use_sky_map = sky_loaded_;
    if (opts_.observer >= 0) live_.observer = std::clamp(opts_.observer, 0, 4);
    live_.show_ergo = opts_.ergo;
    live_.uniform_sky = opts_.uniform_sky;
    live_.flare = opts_.flare;
    if (opts_.flare_r > 0.0f) live_.flare_R = opts_.flare_r;
    live_.flare_t = opts_.flare_t;
    live_.flare_gray = opts_.flare_gray;
    live_.wormhole = opts_.wormhole;
    if (opts_.wh_l > -999.0f) live_.wh_l = opts_.wh_l;
    if (opts_.no_sky_shift) live_.sky_shift = false;
    // look basis in the local triad (outward, θ̂, φ̂): ahead = at the hole
    if (opts_.look == "back") {
        live_.fwd_r = 1.0f;  live_.fwd_th = 0.0f; live_.fwd_ph = 0.0f;
        live_.rgt_r = 0.0f;  live_.rgt_th = 0.0f; live_.rgt_ph = -1.0f;
        live_.up_r = 0.0f;   live_.up_th = -1.0f; live_.up_ph = 0.0f;
    } else if (opts_.look == "side") {
        live_.fwd_r = 0.0f;  live_.fwd_th = 0.0f; live_.fwd_ph = 1.0f;
        live_.rgt_r = 1.0f;  live_.rgt_th = 0.0f; live_.rgt_ph = 0.0f;
        live_.up_r = 0.0f;   live_.up_th = -1.0f; live_.up_ph = 0.0f;
    } else if (!opts_.look.empty() && opts_.look != "ahead") {
        std::fprintf(stderr, "[app] unknown --look '%s' (ahead|back|side)\n", opts_.look.c_str());
    }
    // Gate 24: the fall into Sgr A*
    live_.flight_r0 = opts_.flight_r0;
    live_.flight_theta = opts_.flight_theta;
    live_.flight_rest = opts_.flight_rest;
    if (opts_.flight || opts_.flight_frames > 0 || opts_.flight_r > 0.0f) {
        live_.flight = true;
        start_flight();
        if (opts_.flight_r > 0.0f && flight_) {
            const double t = flight_->tau_at_radius(opts_.flight_r);
            live_.flight_tau = std::isnan(t) ? 0.0 : t;
        }
        live_.flight_play = opts_.flight && !opts_.offscreen;
    }
    if (opts_.wormhole && opts_.incl_deg < 0.0f) {
        live_.incl_deg = 30.0f;   // as enter_exotic(): the GC behind the throat
        live_.cam_phi = 0.0f;
    }
    if (opts_.band == "230") live_.band = 1;
    else if (opts_.band == "kband") live_.band = 2;
    {
        // Gate 19: the electron density normalisation fitted by riaf_tool to
        // the EHT 230 GHz benchmark (2.4 Jy), if that output exists here
        std::ifstream in("validation/extension/riaf.json");
        if (in) {
            try {
                const auto j = nlohmann::json::parse(in);
                live_.riaf_n0 = j.at("main").at("n0").get<double>();
                live_.riaf_T0 = static_cast<float>(j.at("main").at("T0").get<double>());
                live_.riaf_beta = static_cast<float>(j.at("main").at("beta").get<double>());
                live_.riaf_fitted = true;
            } catch (const std::exception&) {
                std::fprintf(stderr, "[app] riaf.json unreadable: 230 GHz not normalised\n");
            }
        }
    }
    if (opts_.flare_amp > 0.0f) live_.flare_amp = opts_.flare_amp;
    live_.show_disk = !opts_.no_disk;

    // Composition aim (--aim X Y): yaw/pitch the ray basis so the hole sits
    // at NDC (X, Y) instead of the frame centre. Built in double precision
    // and exactly orthonormal (the shader must not renormalise it).
    if (opts_.aim_x != 0.0f || opts_.aim_y != 0.0f) {
        const double th = std::tan(0.5 * live_.fov);
        const double aspect = static_cast<double>(target_h_) / target_w_;
        const double psi = std::atan(opts_.aim_x * th);
        const double chi = std::atan(opts_.aim_y * th * aspect);
        const double f0[3] = {-1, 0, 0}, r0[3] = {0, 0, 1}, u0[3] = {0, -1, 0};
        double f1[3], r1[3], f[3], u[3];
        for (int i = 0; i < 3; ++i) {
            f1[i] = std::cos(psi) * f0[i] - std::sin(psi) * r0[i];
            r1[i] = std::sin(psi) * f0[i] + std::cos(psi) * r0[i];
        }
        for (int i = 0; i < 3; ++i) {
            f[i] = std::cos(chi) * f1[i] - std::sin(chi) * u0[i];
            u[i] = std::sin(chi) * f1[i] + std::cos(chi) * u0[i];
        }
        live_.fwd_r = static_cast<float>(f[0]);
        live_.fwd_th = static_cast<float>(f[1]);
        live_.fwd_ph = static_cast<float>(f[2]);
        live_.rgt_r = static_cast<float>(r1[0]);
        live_.rgt_th = static_cast<float>(r1[1]);
        live_.rgt_ph = static_cast<float>(r1[2]);
        live_.up_r = static_cast<float>(u[0]);
        live_.up_th = static_cast<float>(u[1]);
        live_.up_ph = static_cast<float>(u[2]);
    }

    // Phase 9: cinematic menu — load the baked hero plate (blit source only;
    // the live renderer stays idle while the menu is up).
    if (opts_.menu) {
        mode_ = UiMode::Menu;
        hero_ = create_texture_png(
            std::string(SGR_ASSET_DIR) + "/hero_plate_4k.png",
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        if (hero_.w == 0)
            std::fprintf(stderr,
                         "[app] hero plate missing — menu uses black backdrop\n");
    }

    if (opts_.hud) hud_ = std::make_unique<SimHUD>(*ctx_, *this);
}

void App::start_flight() {
    InfallStart st;
    st.kind = live_.flight_rest ? InfallStartKind::Rest : InfallStartKind::Rain;
    st.r0 = live_.flight_r0;
    st.theta0 = live_.flight_theta * 3.14159265358979323846 / 180.0;
    flight_ = std::make_unique<InfallTrajectory>(live_.a_star, st);
    live_.flight_tau = 0.0;
}

void App::enter_flight() {
    enter_simulation(false);
    live_.flight = true;
    live_.band = 0;
    // look ahead, at the hole
    live_.fwd_r = -1.0f; live_.fwd_th = 0.0f; live_.fwd_ph = 0.0f;
    live_.rgt_r = 0.0f;  live_.rgt_th = 0.0f; live_.rgt_ph = 1.0f;
    live_.up_r = 0.0f;   live_.up_th = -1.0f; live_.up_ph = 0.0f;
    start_flight();
    live_.flight_play = true;
}

void App::enter_exotic() {
    enter_simulation(false);
    live_.wormhole = true;
    live_.band = 0;           // the sky in visible light is the whole scene
    live_.incl_deg = 30.0f;   // Galactic Centre star clouds behind the throat
    live_.cam_phi = 0.0f;
}

void App::enter_simulation(bool science, bool scope) {
    live_.wormhole = false;   // the Sgr A* scenes (exotic demo: enter_exotic)
    live_.flight = false;     // the exterior cameras (in-fall: enter_flight)
    const bool was_scope = live_.scope_mode;
    live_.science_mode = science;
    live_.scope_mode = scope;
    if (scope) {
        // Photon-ring scope framing: shadow + n=1 ring fill the frame.
        live_.r_cam = 60.0f;
        live_.fov = 0.22f;
    } else if (was_scope) {
        apply_preset(live_.preset); // restore normal framing
        live_.fov = 0.70f;
    }
    mode_ = UiMode::Simulation;
}

void App::request_quit() { glfwSetWindowShouldClose(ctx_->window(), GLFW_TRUE); }

void App::apply_preset(int index) {
    if (index < 0 || index >= static_cast<int>(kDiskPresets.size())) index = 0;
    const DiskPreset& p = kDiskPresets[index];
    live_.preset = index;
    live_.style = p.style;
    live_.a_star = p.a_star;
    live_.incl_deg = p.inclination_deg;
    live_.disk_r_out = p.disk_outer_r;
    live_.exposure_ev = p.exposure_ev;
    live_.mdot_msun_yr = p.mdot_msun_yr;
    live_.t_peak_from_mdot = true;
    live_.r_cam = p.r_cam;
    live_.turb = p.turb;
    live_.bloom = p.bloom;
    live_.wb_kelvin = p.wb_kelvin;
    live_.saturation = p.saturation;
}

App::Texture App::create_texture_png(const std::string& path,
                                     VkImageUsageFlags usage,
                                     VkImageLayout final_layout) {
    int w = 0, h = 0, comp = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!px) {
        std::fprintf(stderr, "[app] missing texture %s\n", path.c_str());
        return Texture{};
    }
    const Texture tex = upload_texture(px, w, h, VK_FORMAT_R8G8B8A8_UNORM, 4,
                                       usage, final_layout);
    stbi_image_free(px);
    return tex;
}

App::Texture App::create_texture_rgba32f(const std::vector<float>& rgba, int w,
                                         int h, VkImageUsageFlags usage,
                                         VkImageLayout final_layout) {
    if (rgba.size() != 4ull * w * h) {
        std::fprintf(stderr, "[app] float texture size mismatch\n");
        return Texture{};
    }
    return upload_texture(rgba.data(), w, h, VK_FORMAT_R32G32B32A32_SFLOAT, 16,
                          usage, final_layout);
}

App::Texture App::upload_texture(const void* texels, int w, int h,
                                 VkFormat format, size_t bytes_per_texel,
                                 VkImageUsageFlags usage,
                                 VkImageLayout final_layout) {
    Texture tex;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(ctx_->allocator(), &ici, &aci, &tex.image, &tex.alloc,
                         nullptr),
          "vmaCreateImage(texture)");

    // staging upload
    const size_t bytes = bytes_per_texel * static_cast<size_t>(w) * h;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging;
    VmaAllocation salloc;
    VmaAllocationInfo sinfo;
    check(vmaCreateBuffer(ctx_->allocator(), &bci, &sai, &staging, &salloc,
                          &sinfo),
          "vmaCreateBuffer(texture staging)");
    std::memcpy(sinfo.pMappedData, texels, bytes);

    // The read of `final_layout` after the upload can be a transfer (blit
    // source) or a shader sample — cover both in the destination scope.
    const VkPipelineStageFlags dst_stage =
        final_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ? VK_PIPELINE_STAGE_TRANSFER_BIT
            : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    const VkAccessFlags dst_access =
        final_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ? VK_ACCESS_TRANSFER_READ_BIT
            : VK_ACCESS_SHADER_READ_BIT;

    auto& f = ctx_->frame(0);
    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);
    Context::image_barrier(f.cmd, tex.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    vkCmdCopyBufferToImage(f.cmd, staging, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Context::image_barrier(f.cmd, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, final_layout,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT, dst_stage, dst_access);
    vkEndCommandBuffer(f.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
    ctx_->wait_idle();
    vmaDestroyBuffer(ctx_->allocator(), staging, salloc);

    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = tex.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(ctx_->device(), &vci, nullptr, &tex.view),
              "vkCreateImageView(texture)");
    }
    tex.w = w;
    tex.h = h;
    owned_textures_.push_back(tex);
    return tex;
}

// Blackbody table for the live disk (slot 0). Built here from the physics
// library's Planck x CIE integral — the same function the CPU reference
// tracer shades with — so the GPU and CPU colours cannot drift apart.
// Texel i holds ABSOLUTE linear-sRGB radiance (Y(6500 K) = 1) at
// T_i = exp(ln 300 + i/(N−1)·ln(1e5/300)); disk.comp.glsl (BB_*) must
// mirror these constants. Absolute, not normalised: because
// g³·B_{ν/g}(T) = B_ν(gT), the observed pixel is exactly this table at
// g·T — colour and brightness in one lookup, no extra T⁴ factor.
namespace {
constexpr int kBbLutN = 1024;
constexpr double kBbTMin = 300.0, kBbTMax = 1.0e5;
} // namespace

void App::load_luts() {
    // lut_sampler_ (linear) also serves the menu's gallery thumbnails.
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(ctx_->device(), &sci, nullptr, &lut_sampler_),
          "vkCreateSampler(lut)");
    // The float table is read with texelFetch and interpolated in the
    // shader, so it never depends on optional RGBA32F filtering support.
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    check(vkCreateSampler(ctx_->device(), &sci, nullptr, &nearest_sampler_),
          "vkCreateSampler(nearest)");

    {
        std::vector<float> bb(4ull * kBbLutN);
        const double l0 = std::log(kBbTMin), l1 = std::log(kBbTMax);
        for (int i = 0; i < kBbLutN; ++i) {
            const double T = std::exp(l0 + (l1 - l0) * i / (kBbLutN - 1));
            const auto rgb = planck_to_linear_srgb(T);
            bb[4 * i + 0] = static_cast<float>(rgb[0]);
            bb[4 * i + 1] = static_cast<float>(rgb[1]);
            bb[4 * i + 2] = static_cast<float>(rgb[2]);
            bb[4 * i + 3] = 1.0f;
        }
        const Texture lut = create_texture_rgba32f(
            bb, kBbLutN, 1, VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (lut.w > 0) {
            bindless_->write_sampled_image(0, lut.view, nearest_sampler_);
            ++n_luts_;
        }
    }

    std::printf("[app] loaded %zu LUTs\n", n_luts_);
}

// Gate 15 sky: engine/assets/sky/starmap_gal_8k.sky (scripts/prepare_sky.py)
// — NASA SVS Deep Star Maps 2020 in galactic plate carrée, every mip level
// pre-built as B10G11R11 unsigned floats. Missing file → procedural stars.
void App::load_sky() {
    const std::string path =
        std::string(SGR_ASSET_DIR) + "/sky/starmap_gal_8k.sky";
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::printf("[app] no sky map (%s) — procedural stars; run "
                    "scripts/prepare_sky.py\n", path.c_str());
        return;
    }
    char magic[8] = {};
    uint32_t hdr[4] = {};
    if (std::fread(magic, 1, 8, fp) != 8 || std::memcmp(magic, "SGRSKY01", 8) ||
        std::fread(hdr, 4, 4, fp) != 4 || hdr[3] != 0 || hdr[2] == 0 ||
        hdr[2] > 16) {
        std::fprintf(stderr, "[app] bad sky file %s\n", path.c_str());
        std::fclose(fp);
        return;
    }
    const uint32_t w = hdr[0], h = hdr[1], levels = hdr[2];
    std::vector<VkBufferImageCopy> regions(levels);
    size_t total = 0;
    for (uint32_t l = 0; l < levels; ++l) {
        const uint32_t lw = std::max(1u, w >> l), lh = std::max(1u, h >> l);
        VkBufferImageCopy& r = regions[l];
        r.bufferOffset = total;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
        r.imageExtent = {lw, lh, 1};
        total += 4ull * lw * lh;
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = total;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging;
    VmaAllocation salloc;
    VmaAllocationInfo sinfo;
    check(vmaCreateBuffer(ctx_->allocator(), &bci, &sai, &staging, &salloc,
                          &sinfo),
          "vmaCreateBuffer(sky staging)");
    const size_t got = std::fread(sinfo.pMappedData, 1, total, fp);
    std::fclose(fp);
    if (got != total) {
        std::fprintf(stderr, "[app] truncated sky file %s\n", path.c_str());
        vmaDestroyBuffer(ctx_->allocator(), staging, salloc);
        return;
    }

    Texture tex;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    ici.extent = {w, h, 1};
    ici.mipLevels = levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(ctx_->allocator(), &ici, &aci, &tex.image, &tex.alloc,
                         nullptr),
          "vmaCreateImage(sky)");

    auto& f = ctx_->frame(0);
    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = tex.image;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(f.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &ib);
    vkCmdCopyBufferToImage(f.cmd, staging, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels,
                           regions.data());
    ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(f.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &ib);
    vkEndCommandBuffer(f.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    check(vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE),
          "vkQueueSubmit(sky upload)");
    ctx_->wait_idle();
    vmaDestroyBuffer(ctx_->allocator(), staging, salloc);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = tex.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};
    check(vkCreateImageView(ctx_->device(), &vci, nullptr, &tex.view),
          "vkCreateImageView(sky)");
    tex.w = static_cast<int>(w);
    tex.h = static_cast<int>(h);
    owned_textures_.push_back(tex);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;        // longitude wraps
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // poles
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = static_cast<float>(levels);
    check(vkCreateSampler(ctx_->device(), &sci, nullptr, &sky_sampler_),
          "vkCreateSampler(sky)");
    bindless_->write_sampled_image(1, tex.view, sky_sampler_);
    sky_loaded_ = true;
    std::printf("[app] sky map %ux%u (%u mips, %.0f MiB) — NASA/GSFC SVS Deep "
                "Star Maps 2020; Gaia DR2: ESA/Gaia/DPAC\n",
                w, h, levels, total / 1048576.0);
}

App::~App() {
    if (ctx_) ctx_->wait_idle();
    hud_.reset();
    for (auto& t : owned_textures_) {
        if (t.view) vkDestroyImageView(ctx_->device(), t.view, nullptr);
        if (t.image) vmaDestroyImage(ctx_->allocator(), t.image, t.alloc);
    }
    if (lut_sampler_) vkDestroySampler(ctx_->device(), lut_sampler_, nullptr);
    if (nearest_sampler_)
        vkDestroySampler(ctx_->device(), nearest_sampler_, nullptr);
    if (sky_sampler_) vkDestroySampler(ctx_->device(), sky_sampler_, nullptr);
    if (frame_buf_) vmaDestroyBuffer(ctx_->allocator(), frame_buf_, frame_buf_alloc_);
    destroy_render_target();
}

void App::create_render_target() {
    if (opts_.offscreen) {
        target_w_ = static_cast<uint32_t>(opts_.width);
        target_h_ = static_cast<uint32_t>(opts_.height);
    } else {
        // live_.render_scale is runtime-tunable (auto-res / HUD slider);
        // > 1 renders oversized and the LINEAR blit downsamples (cheap SSAA).
        const float s = std::clamp(live_.render_scale, 0.25f, 2.0f);
        target_w_ = std::max(16u, static_cast<uint32_t>(
                                      ctx_->swapchain_extent().width * s));
        target_h_ = std::max(16u, static_cast<uint32_t>(
                                      ctx_->swapchain_extent().height * s));
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {target_w_, target_h_, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(ctx_->allocator(), &ici, &aci, &target_,
                         &target_alloc_, nullptr),
          "vmaCreateImage(target)");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = target_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(ctx_->device(), &vci, nullptr, &target_view_),
          "vkCreateImageView(target)");

    bindless_->write_storage_image(kSlotLdr, target_view_);
    target_initialised_ = false;

    if (post_enabled()) {
        // Linear radiance accumulator (fp32: hundreds of progressive samples
        // average without drift) + rgba16f bloom pyramid at 1/2, 1/4, ...
        hdr_ = create_storage_image(target_w_, target_h_,
                                    VK_FORMAT_R32G32B32A32_SFLOAT, kSlotHdr);
        uint32_t w = target_w_, h = target_h_;
        for (uint32_t k = 0; k < kMaxBloomLevels; ++k) {
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
            if (w < 4 || h < 4) break;
            bloom_.push_back(create_storage_image(
                w, h, VK_FORMAT_R16G16B16A16_SFLOAT, kSlotBloom0 + k));
        }
        post_initialised_ = false;
        accum_n_ = 0;
        accum_sig_.clear();
    }
}

void App::destroy_render_target() {
    if (target_view_) vkDestroyImageView(ctx_->device(), target_view_, nullptr);
    if (target_) vmaDestroyImage(ctx_->allocator(), target_, target_alloc_);
    target_view_ = VK_NULL_HANDLE;
    target_ = VK_NULL_HANDLE;
    destroy_storage_image(hdr_);
    for (auto& b : bloom_) destroy_storage_image(b);
    bloom_.clear();
}

App::StorageImage App::create_storage_image(uint32_t w, uint32_t h,
                                            VkFormat format, uint32_t slot) {
    StorageImage img;
    img.w = w;
    img.h = h;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check(vmaCreateImage(ctx_->allocator(), &ici, &aci, &img.image, &img.alloc,
                         nullptr),
          "vmaCreateImage(storage)");
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(ctx_->device(), &vci, nullptr, &img.view),
          "vkCreateImageView(storage)");
    bindless_->write_storage_image(slot, img.view);
    return img;
}

void App::destroy_storage_image(StorageImage& img) {
    if (img.view) vkDestroyImageView(ctx_->device(), img.view, nullptr);
    if (img.image) vmaDestroyImage(ctx_->allocator(), img.image, img.alloc);
    img = StorageImage{};
}

// Menu frame: no compute — blit the baked hero plate with a slow pan/zoom
// (subtle background motion per the design reference), then the ImGui pass.
void App::record_menu(VkCommandBuffer cmd, uint32_t sc_index, float t) {
    const VkImage sc_img = ctx_->swapchain_images()[sc_index];
    const uint32_t sw = ctx_->swapchain_extent().width;
    const uint32_t sh = ctx_->swapchain_extent().height;

    Context::image_barrier(cmd, sc_img, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT);

    if (hero_.w > 0) {
        // Ken Burns crop: aspect-correct window into the 4K plate, drifting
        // slowly. Amplitudes stay well inside the over-rendered margins.
        const float zoom = 0.90f + 0.035f * std::sin(0.031f * t);
        const float cx = 0.5f + 0.040f * std::sin(0.017f * t);
        const float cy = 0.5f + 0.030f * std::sin(0.011f * t + 1.7f);
        const float fit = std::min(static_cast<float>(hero_.w) / sw,
                                   static_cast<float>(hero_.h) / sh);
        const float crop_w = sw * fit * zoom;
        const float crop_h = sh * fit * zoom;
        const float x0 = std::clamp(cx * hero_.w - 0.5f * crop_w, 0.0f,
                                    hero_.w - crop_w);
        const float y0 = std::clamp(cy * hero_.h - 0.5f * crop_h, 0.0f,
                                    hero_.h - crop_h);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[0] = {static_cast<int32_t>(x0),
                              static_cast<int32_t>(y0), 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(x0 + crop_w),
                              static_cast<int32_t>(y0 + crop_h), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {static_cast<int32_t>(sw), static_cast<int32_t>(sh),
                              1};
        vkCmdBlitImage(cmd, hero_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       sc_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);
    } else {
        // No plate baked yet: clear to near-black.
        VkClearColorValue clear{{0.01f, 0.01f, 0.015f, 1.0f}};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, sc_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clear, 1, &range);
    }

    if (hud_) {
        hud_->record(cmd, sc_index);
    } else {
        Context::image_barrier(cmd, sc_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }
}

void App::record(VkCommandBuffer cmd, uint32_t sc_index, float t) {
    if (mode_ == UiMode::Menu) {
        record_menu(cmd, sc_index, t);
        return;
    }
    // display image → GENERAL (first use: from UNDEFINED). The source scope
    // covers last frame's blit read of it (write-after-read).
    Context::image_barrier(
        cmd, target_,
        target_initialised_ ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                            : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    target_initialised_ = true;

    PushConstants pc = make_push_constants(t);
    if (post_enabled()) {
        // Gate 14: trace into the radiance accumulator, then the display
        // chain (bloom → tone map) writes the 8-bit image blitted below.
        const FrameParams fp = make_frame_params();
        record_frame_params(cmd, fp);
        prepare_post_images(cmd);
        update_accumulation(pc, fp);
        pc.image_index = kSlotHdr;
        record_trace(cmd, pc, 0, target_h_);
        compute_barrier(cmd);
        record_post(cmd);
    } else {
        pc.image_index = kSlotLdr;
        record_trace(cmd, pc, 0, target_h_);
    }

    // compute writes → blit read
    Context::image_barrier(cmd, target_, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_READ_BIT);

    const VkImage sc_img = ctx_->swapchain_images()[sc_index];
    Context::image_barrier(cmd, sc_img, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(target_w_),
                          static_cast<int32_t>(target_h_), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(ctx_->swapchain_extent().width),
                          static_cast<int32_t>(ctx_->swapchain_extent().height), 1};
    vkCmdBlitImage(cmd, target_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc_img,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   target_w_ == ctx_->swapchain_extent().width
                       ? VK_FILTER_NEAREST
                       : VK_FILTER_LINEAR);

    if (hud_) {
        // The HUD render pass consumes TRANSFER_DST and ends in PRESENT_SRC.
        hud_->record(cmd, sc_index);
    } else {
        Context::image_barrier(cmd, sc_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }
}

void App::record_trace(VkCommandBuffer cmd, PushConstants pc,
                       uint32_t rows_from, uint32_t rows) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_->pipeline());
    const VkDescriptorSet set = bindless_->set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_->layout(), 0, 1, &set, 0, nullptr);
    pc.y_offset = rows_from;
    vkCmdPushConstants(cmd, pipeline_->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdDispatch(cmd, (target_w_ + 7) / 8, (rows + 7) / 8, 1);
}

void App::prepare_post_images(VkCommandBuffer cmd) {
    if (post_initialised_) {
        compute_barrier(cmd);
        return;
    }
    auto to_general = [&](VkImage img) {
        Context::image_barrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT);
    };
    to_general(hdr_.image);
    for (const auto& b : bloom_) to_general(b.image);
    post_initialised_ = true;
}

// Exposure metering. Linear/science: absolute (1, identical to the CPU
// reference). Filmic: metered like a camera — EV 0 places the luminance of
// the un-beamed peak-temperature disk at 1.0 (~2.5 stops over mid-grey), so
// a 7000 K and a 23000 K disk are both well exposed and the Doppler contrast
// between the two sides stays on the tone curve.
double App::display_meter() const {
    // false-colour bands (radio, near-infrared) are not metered on a disk
    const bool linear = live_.science_mode || live_.tonemap == 0 || live_.band != 0;
    if (linear || opts_.shader != "disk") return 1.0;
    const double T = effective_t_peak();
    double meter;
    if (live_.bolometric) {
        meter = std::pow(T / 6500.0, 4.0);
    } else {
        const auto rgb = planck_to_linear_srgb(T);
        meter = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    }
    // no luminous disk to meter (e.g. r_out inside the ISCO, T_peak = 0):
    // fall back to the absolute scale instead of a ~1e12 exposure
    return (meter > 1e-9) ? meter : 1.0;
}

// Gate 17: the camera's local frame. A pixel along local direction n of an
// observer with orthonormal tetrad (e0; e_i) records k_rec = e0 − n^i e_i;
// the shader traces −k_rec backward in Kerr-Schild coordinates. The tetrad
// is the ZAMO frame at BL (r_cam, incl, φ = 0), Lorentz-boosted by the
// observer's velocity v relative to the ZAMO — aberration and the
// Doppler shift of everything seen follow automatically.
FrameParams App::make_frame_params(ObserverState* st) const {
    const double a = live_.a_star;
    const double r = live_.r_cam;
    const double th = live_.incl_deg * 3.14159265358979323846 / 180.0;
    const KerrMetric bl(1.0, a);

    // Observer 3-velocity relative to the ZAMO (sgr_physics/observers.hpp,
    // validated in test_observers.cpp; the same code feeds the paper).
    std::array<double, 3> v{};
    switch (static_cast<Observer>(live_.observer)) {
        case Observer::Orbiting:
            v = observer_velocity(bl, ObserverKind::Orbiting, r, th);
            break;
        case Observer::Infall:
            v = observer_velocity(bl, ObserverKind::Infall, r, th);
            break;
        case Observer::Hovering:
            v = observer_velocity(bl, ObserverKind::Hovering, r, th);
            break;
        case Observer::RideStar:
            // moving with the focused S-star: ride_v = (ṙ, r θ̇, r sinθ φ̇)
            // of its full-GR orbit in coordinate time (SimHUD)
            if (live_.ride_v_valid) {
                const double sth = std::max(std::sin(th), 1e-9);
                v = observer_velocity(bl, ObserverKind::Moving, r, th,
                                      {live_.ride_v[0], live_.ride_v[1] / r,
                                       live_.ride_v[2] / (r * sth)});
            }
            break;
        case Observer::Zamo:
        default:
            break;
    }
    // invalid (v ≥ c here): observer_frame falls back to the ZAMO frame
    const ObserverFrame of = observer_frame(bl, r, th, v);

    // look legs from the triad components the camera basis is built in
    using V4 = std::array<double, 4>;
    auto leg = [&](float c_r, float c_th, float c_ph) {
        V4 out{};
        for (int m = 0; m < 4; ++m)
            out[m] = c_r * of.e[1][m] + c_th * of.e[2][m] + c_ph * of.e[3][m];
        return out;
    };
    const V4 fwd = leg(live_.fwd_r, live_.fwd_th, live_.fwd_ph);
    const V4 rgt = leg(live_.rgt_r, live_.rgt_th, live_.rgt_ph);
    const V4 up = leg(live_.up_r, live_.up_th, live_.up_ph);

    const KerrSchild ks(1.0, std::clamp(a, -0.9999, 0.9999), -1);
    FrameParams fp{};
    const Vec4 X = ks.point_from_bl(0.0, r, th, 0.0);
    auto put = [](float* dst, const Vec4& src) {
        for (int m = 0; m < 4; ++m) dst[m] = static_cast<float>(src[m]);
    };
    put(fp.cam_x, X);
    put(fp.e0, ks.vector_from_bl(r, th, 0.0, of.e[0]));
    put(fp.e_fwd, ks.vector_from_bl(r, th, 0.0, fwd));
    put(fp.e_rgt, ks.vector_from_bl(r, th, 0.0, rgt));
    put(fp.e_up, ks.vector_from_bl(r, th, 0.0, up));
    fp.ks_chart = -1.0f;
    // Gate 24: the in-faller's camera replaces the exterior observer. Its
    // event and frame come from the fp64 worldline in the INGOING chart
    // (regular through r₊); the look basis is applied to its legs exactly as
    // to the ZAMO triad above.
    if (live_.flight && flight_) {
        const InfallSample s = flight_->at(live_.flight_tau);
        const auto e = infall_tetrad(flight_->chart(), s.X, s.U);
        auto legf = [&](float c_r, float c_th, float c_ph) {
            Vec4 out{};
            for (int m = 0; m < 4; ++m)
                out[m] = c_r * e[1][m] + c_th * e[2][m] + c_ph * e[3][m];
            return out;
        };
        put(fp.cam_x, s.X);
        put(fp.e0, s.U);
        put(fp.e_fwd, legf(live_.fwd_r, live_.fwd_th, live_.fwd_ph));
        put(fp.e_rgt, legf(live_.rgt_r, live_.rgt_th, live_.rgt_ph));
        put(fp.e_up, legf(live_.up_r, live_.up_th, live_.up_ph));
        fp.ks_chart = 1.0f;
    }
    fp.flags = (live_.sky_shift ? 1u : 0u) | (live_.show_ergo ? 2u : 0u);
    fp.max_steps = static_cast<uint32_t>(std::clamp(opts_.max_steps, 100, 100000));

    // Gate 18 hot spot: orbit constants and comoving metric from the fp64
    // CPU twin (science/flares/Hotspot.cpp), so both trace the same emitter.
    if (live_.flare && live_.flare_R > bl.r_isco(true)) {
        HotspotParams hp;
        hp.a_star = std::clamp(a, -0.9999, 0.9999);
        hp.R = live_.flare_R;
        hp.sigma = live_.flare_sigma;
        hp.alpha = live_.flare_alpha;
        const Hotspot hs(hp);
        const auto h = hs.comoving_metric();
        const auto c0 = hs.centre(0.0);
        const float s0[4] = {live_.flare_R, live_.flare_sigma, 0.0f, live_.flare_amp};
        const float s1[4] = {static_cast<float>(hs.omega()), static_cast<float>(hs.ut()),
                             live_.flare_t, live_.flare_alpha};
        const float h0[4] = {static_cast<float>(h[0]), static_cast<float>(h[1]),
                             static_cast<float>(h[2]), static_cast<float>(h[3])};
        const float h1[4] = {static_cast<float>(h[4]), static_cast<float>(h[5]),
                             static_cast<float>(c0[0]), static_cast<float>(c0[1])};
        std::memcpy(fp.spot0, s0, sizeof(s0));
        std::memcpy(fp.spot1, s1, sizeof(s1));
        std::memcpy(fp.spot_h0, h0, sizeof(h0));
        std::memcpy(fp.spot_h1, h1, sizeof(h1));
        fp.flags |= 4u;
    }
    // Gate 19: the hot accretion flow in the 230 GHz band
    if (live_.band == 1) {
        const double ri = bl.r_isco(true);
        const float r0[4] = {static_cast<float>(live_.riaf_n0), live_.riaf_T0, live_.riaf_beta,
                             live_.riaf_nu_ghz};
        const float r1[4] = {static_cast<float>(bl.energy_circular(ri, true)),
                             static_cast<float>(bl.angmom_circular(ri, true)),
                             static_cast<float>(ri), 60.0f};
        const float r2[4] = {static_cast<float>(constants::sgrA_rg_m * 100.0), 0.0f, 0.0f, 0.0f};
        std::memcpy(fp.riaf0, r0, sizeof(r0));
        std::memcpy(fp.riaf1, r1, sizeof(r1));
        std::memcpy(fp.riaf2, r2, sizeof(r2));
        fp.flags |= 8u;
    }
    // Gate 21: the exotic wormhole demo (replaces the Kerr scene entirely)
    if (live_.wormhole) {
        const double M = live_.wh_W / DnegWormhole::lensing_width_closed_form(1.0);
        const float w0[4] = {1.0f, live_.wh_a, static_cast<float>(M), live_.wh_l};
        std::memcpy(fp.wh0, w0, sizeof(w0));
        fp.flags = (fp.flags & ~(4u | 8u)) | 16u;
    }

    if (st) {
        st->valid = of.valid;
        st->alpha = of.alpha;
        st->omega = of.omega;
        st->v = of.speed;
        st->gamma = of.gamma;
        st->dtau_dt = of.dtau_dt();         // 1/u^t (BL t = far-away time)
        st->r_ergo = bl.r_ergosphere(th);
    }
    return fp;
}

void App::record_frame_params(VkCommandBuffer cmd, const FrameParams& fp) {
    // previous frames' shader reads → this update → this frame's reads
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
    vkCmdUpdateBuffer(cmd, frame_buf_, 0, sizeof(FrameParams), &fp);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}

TonemapPC App::make_tonemap_pc() const {
    TonemapPC tp{};
    tp.hdr_slot = kSlotHdr;
    tp.ldr_slot = kSlotLdr;
    tp.bloom_slot = kSlotBloom0;
    tp.w = target_w_;
    tp.h = target_h_;
    tp.bloom_w = bloom_.empty() ? 1 : bloom_[0].w;
    tp.bloom_h = bloom_.empty() ? 1 : bloom_[0].h;
    // the 230 GHz band is already a false-colour display: keep it linear
    tp.op = (live_.science_mode || live_.band == 1) ? 0u : static_cast<uint32_t>(live_.tonemap);
    tp.exposure = static_cast<float>(std::exp2(live_.exposure_ev) / display_meter());
    tp.bloom_strength = (tp.op == 0u || bloom_.empty())
                            ? 0.0f
                            : std::clamp(live_.bloom, 0.0f, 1.0f);
    tp.bloom_norm = bloom_.empty() ? 1.0f : 1.0f / static_cast<float>(bloom_.size());
    tp.saturation = live_.saturation;
    // Camera white balance: gains that render a blackbody at wb_kelvin as
    // neutral grey of the same luminance (what a camera's WB setting does).
    static thread_local float cached_wb = -1.0f;
    static thread_local std::array<float, 3> gains{1.0f, 1.0f, 1.0f};
    if (cached_wb != live_.wb_kelvin) {
        const auto rgb = planck_to_linear_srgb(
            std::clamp(static_cast<double>(live_.wb_kelvin), 3000.0, 20000.0));
        const double Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        for (int c = 0; c < 3; ++c)
            gains[c] = static_cast<float>(
                std::clamp(Y / std::max(rgb[c], 1e-12), 0.2, 5.0));
        cached_wb = live_.wb_kelvin;
    }
    tp.wb_r = gains[0];
    tp.wb_g = gains[1];
    tp.wb_b = gains[2];
    return tp;
}

void App::record_post(VkCommandBuffer cmd) {
    const VkDescriptorSet set = bindless_->set();
    const TonemapPC tp = make_tonemap_pc();

    if (tp.bloom_strength > 0.0f) {
        // downsample chain: accumulator → level 0 → level 1 → ...
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          bloom_down_->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bloom_down_->layout(), 0, 1, &set, 0, nullptr);
        for (size_t k = 0; k < bloom_.size(); ++k) {
            BloomDownPC bp{};
            bp.src_slot = k == 0 ? kSlotHdr : kSlotBloom0 + static_cast<uint32_t>(k) - 1;
            bp.dst_slot = kSlotBloom0 + static_cast<uint32_t>(k);
            bp.src_w = k == 0 ? target_w_ : bloom_[k - 1].w;
            bp.src_h = k == 0 ? target_h_ : bloom_[k - 1].h;
            bp.dst_w = bloom_[k].w;
            bp.dst_h = bloom_[k].h;
            bp.src_is_hdr = k == 0 ? 1u : 0u;
            vkCmdPushConstants(cmd, bloom_down_->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp), &bp);
            vkCmdDispatch(cmd, (bp.dst_w + 7) / 8, (bp.dst_h + 7) / 8, 1);
            compute_barrier(cmd);
        }
        // upsample-accumulate chain: coarsest → level 0
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          bloom_up_->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bloom_up_->layout(), 0, 1, &set, 0, nullptr);
        for (size_t k = bloom_.size() - 1; k-- > 0;) {
            BloomUpPC up{};
            up.src_slot = kSlotBloom0 + static_cast<uint32_t>(k) + 1;
            up.dst_slot = kSlotBloom0 + static_cast<uint32_t>(k);
            up.src_w = bloom_[k + 1].w;
            up.src_h = bloom_[k + 1].h;
            up.dst_w = bloom_[k].w;
            up.dst_h = bloom_[k].h;
            up.radius = 1.0f;
            vkCmdPushConstants(cmd, bloom_up_->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
            vkCmdDispatch(cmd, (up.dst_w + 7) / 8, (up.dst_h + 7) / 8, 1);
            compute_barrier(cmd);
        }
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemap_->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            tonemap_->layout(), 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, tonemap_->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(tp), &tp);
    vkCmdDispatch(cmd, (target_w_ + 7) / 8, (target_h_ + 7) / 8, 1);
}

void App::update_accumulation(PushConstants& pc, const FrameParams& fp) {
    // Scene signature = everything the RADIANCE depends on (push constants
    // + observer frame). Display-only settings (exposure, bloom, tone
    // curve) do not restart accumulation.
    PushConstants key = pc;
    key.time_s = 0.0f;
    key.exposure_ev = 0.0f;
    key.y_offset = 0;
    key.accum_index = 0;
    key.accum_weight = 0.0f;
    FrameParams fkey = fp;
    fkey.spot1[2] = 0.0f;   // a playing flare is animated, not a new scene
    std::vector<unsigned char> sig(sizeof(key) + sizeof(fkey));
    std::memcpy(sig.data(), &key, sizeof(key));
    std::memcpy(sig.data() + sizeof(key), &fkey, sizeof(fkey));
    const bool same = accum_sig_ == sig;
    if (!same || !live_.accumulate) accum_n_ = 0;
    accum_sig_ = std::move(sig);
    // A time-dependent scene (filaments/clouds advected at rot_speed) keeps
    // an exponential window of ~8 frames instead of a converging mean.
    const bool animated =
        (live_.rot_speed > 0.0f &&
         (live_.turb > 0.0f || (live_.style & kStyleClouds) != 0u)) ||
        (live_.flare && live_.flare_play && live_.flare_speed > 0.0f);
    const uint32_t cap = animated ? 8u : 4096u;
    pc.accum_index = accum_n_;
    pc.accum_weight = 1.0f / static_cast<float>(std::min(accum_n_ + 1, cap));
    if (live_.accumulate) ++accum_n_;
}

void App::save_screenshot() {
    ctx_->wait_idle();
    // With the HUD active, capture the composited swapchain image (UI
    // included); otherwise read back the compute target directly.
    const bool from_swapchain = static_cast<bool>(hud_);
    const VkImage src_img = from_swapchain
                                ? ctx_->swapchain_images()[last_sc_index_]
                                : target_;
    const uint32_t sw = from_swapchain ? ctx_->swapchain_extent().width : target_w_;
    const uint32_t sh = from_swapchain ? ctx_->swapchain_extent().height : target_h_;
    const uint32_t target_w_save = target_w_, target_h_save = target_h_;
    target_w_ = sw;
    target_h_ = sh;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 4ull * target_w_ * target_h_;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buf;
    VmaAllocation alloc;
    VmaAllocationInfo ainfo;
    check(vmaCreateBuffer(ctx_->allocator(), &bci, &aci, &buf, &alloc, &ainfo),
          "vmaCreateBuffer(screenshot)");

    auto& f = ctx_->frame(0);
    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);
    if (from_swapchain) {
        Context::image_barrier(f.cmd, src_img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {target_w_, target_h_, 1};
    vkCmdCopyImageToBuffer(f.cmd, src_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buf, 1, &region);
    vkEndCommandBuffer(f.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
    ctx_->wait_idle();

    // Swapchain is BGRA; the compute target is RGBA.
    const int ir = from_swapchain ? 2 : 0;
    const int ib = from_swapchain ? 0 : 2;
    std::vector<unsigned char> rgb(3ull * target_w_ * target_h_);
    const auto* src = static_cast<const unsigned char*>(ainfo.pMappedData);
    for (size_t i = 0; i < static_cast<size_t>(target_w_) * target_h_; ++i) {
        rgb[3 * i + 0] = src[4 * i + ir];
        rgb[3 * i + 1] = src[4 * i + 1];
        rgb[3 * i + 2] = src[4 * i + ib];
    }
    if (stbi_write_png(opts_.screenshot_path.c_str(), target_w_, target_h_, 3,
                       rgb.data(), 3 * target_w_))
        std::printf("[app] screenshot written: %s\n",
                    opts_.screenshot_path.c_str());
    else
        std::fprintf(stderr, "[app] screenshot FAILED (missing directory?): %s\n",
                     opts_.screenshot_path.c_str());
    vmaDestroyBuffer(ctx_->allocator(), buf, alloc);
    target_w_ = target_w_save;
    target_h_ = target_h_save;
}

float App::effective_t_peak() const {
    if (!live_.t_peak_from_mdot) return live_.t_peak_k;
    // Physical Novikov-Thorne peak temperature for Sgr A*'s mass at the
    // chosen accretion rate (physics/src/disk_model.cpp), cached.
    static thread_local float c_mdot = -1.0f, c_a = -10.0f, c_rout = -1.0f,
                              c_t = 0.0f;
    if (c_mdot != live_.mdot_msun_yr || c_a != live_.a_star ||
        c_rout != live_.disk_r_out) {
        c_t = static_cast<float>(nt_peak_temperature_k(
            constants::sgrA_mass_msun, live_.mdot_msun_yr, live_.a_star,
            live_.disk_r_out));
        c_mdot = live_.mdot_msun_yr;
        c_a = live_.a_star;
        c_rout = live_.disk_r_out;
    }
    return c_t;
}

PushConstants App::make_push_constants(float t) const {
    // Live params (HUD-editable) drive everything; offscreen/diagnostic modes
    // run with the CLI values they were initialised from.
    const KerrMetric metric(1.0, live_.a_star);
    const double r_isco = metric.r_isco(true);
    PushConstants pc{};
    pc.image_index = 0;
    pc.width = target_w_;
    pc.height = target_h_;
    pc.time_s = t;
    pc.a_star = live_.a_star;
    pc.r_cam = live_.r_cam;
    pc.incl_deg = live_.incl_deg;
    pc.fov = live_.fov;
    pc.disk_r_in = static_cast<float>(r_isco);
    // hidden disk: an empty annulus (r_out < r_in) — the exposure meter and
    // T_peak keep using the preset's disk so the display stays calibrated
    pc.disk_r_out = (live_.show_disk && live_.band == 0) ? live_.disk_r_out : 0.0f;
    pc.exposure_ev = live_.exposure_ev;
    pc.shade_mode = opts_.shader == "disk" ? live_.style : opts_.shade_mode;
    pc.y_offset = 0;
    pc.t_peak_k = effective_t_peak();
    pc.turb = live_.turb;
    pc.cam_phi = live_.cam_phi;
    if (live_.flight && flight_) {
        // the shader's escape radius and edge-on step tightening follow
        // the in-faller (its azimuth is already in the event)
        const InfallSample s = flight_->at(live_.flight_tau);
        pc.r_cam = static_cast<float>(s.r);
        pc.incl_deg = static_cast<float>(s.theta * 180.0 / 3.14159265358979323846);
        pc.cam_phi = 0.0f;
    }
    // 1.0 on the slider = real Sgr A* wall-clock: one geometric time unit
    // (GM/c³) is 21.17 s for M = 4.297e6 Msun.
    pc.time_scale = live_.rot_speed / 21.17f;
    // radio and near-infrared bands: the optical Milky Way map is not what
    // those cameras see — black sky; no thin disk outside the visible band
    pc.sky_off = (live_.black_sky || live_.band != 0) ? 1u : 0u;
    pc.fwd_r = live_.fwd_r;
    pc.fwd_th = live_.fwd_th;
    pc.fwd_ph = live_.fwd_ph;
    pc.rgt_r = live_.rgt_r;
    pc.rgt_th = live_.rgt_th;
    pc.rgt_ph = live_.rgt_ph;
    pc.up_r = live_.up_r;
    pc.up_th = live_.up_th;
    pc.up_ph = live_.up_ph;
    pc.display_flags = (live_.bolometric ? 1u : 0u) | (live_.show_grid ? 2u : 0u) |
                       ((sky_loaded_ && live_.use_sky_map) ? 4u : 0u) |
                       (live_.uniform_sky ? 8u : 0u) |
                       (live_.flare_gray ? 16u : 0u);
    // The Milky Way map is in its own display units (no physical flux ratio
    // to the disk exists — it is the sky seen from Earth). Scaling by the
    // camera meter keeps it at its native look whatever the disk's T_peak.
    pc.sky_gain = static_cast<float>(std::exp2(live_.sky_ev) * display_meter());

    // Page-Thorne normalisation, cached against the parameters it depends on.
    static thread_local double cached_a = -10.0, cached_rout = -1.0,
                               cached_fmax = 1.0;
    if (cached_a != live_.a_star || cached_rout != live_.disk_r_out) {
        const NovikovThorneDisk disk(live_.a_star, r_isco, live_.disk_r_out);
        double fmax = 0.0;
        for (int i = 0; i <= 512; ++i) {
            const double r = r_isco + (live_.disk_r_out - r_isco) * i / 512.0;
            fmax = std::max(fmax, disk.flux(r));
        }
        cached_a = live_.a_star;
        cached_rout = live_.disk_r_out;
        cached_fmax = std::max(fmax, 1e-12);
    }
    pc.flux_max = static_cast<float>(cached_fmax);
    return pc;
}

// Offscreen batch render: exact width×height target, one queue submission per
// row band (keeps each GPU packet well under the ~2 s Windows TDR watchdog for
// the double-precision geodesic shader), then screenshot and exit.
int App::run_offscreen() {
    auto& f = ctx_->frame(0);
    const uint32_t band_rows = 16;
    PushConstants pc = make_push_constants(0.0f);
    pc.image_index = post_enabled() ? kSlotHdr : kSlotLdr;
    // Progressive samples (Gate 14): sample s is R2-jittered in the shader
    // and folded into the running mean; s = 0 is the un-jittered image.
    const uint32_t spp =
        post_enabled() ? static_cast<uint32_t>(std::max(1, opts_.spp)) : 1u;

    auto submit = [&](auto&& body) {
        vkResetCommandBuffer(f.cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(f.cmd, &bi);
        body(f.cmd);
        vkEndCommandBuffer(f.cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &f.cmd;
        check(vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE),
              "vkQueueSubmit(offscreen)");
        ctx_->wait_idle();
    };

    // Gate 18 movies: --movie N renders N frames, the observing clock
    // advanced by --movie-dt M each, written as <name>_0000.png ...
    // Gate 24: --flight-frames N records the fall. Frames are spaced evenly
    // in log(r − r₋) from the start to the stop just above r₋, so the
    // recording slows down as the faller nears the horizon (each frame's
    // proper time and radius go to <name>_frames.csv).
    const bool flight_movie = opts_.flight_frames > 0 && flight_;
    const int n_frames = flight_movie ? opts_.flight_frames : std::max(1, opts_.movie_frames);
    const std::string shot = opts_.screenshot_path;
    std::FILE* flight_csv = nullptr;
    if (flight_movie && !shot.empty()) {
        const size_t dot = shot.find_last_of('.');
        const std::string csv = (dot == std::string::npos ? shot : shot.substr(0, dot)) + "_frames.csv";
        flight_csv = std::fopen(csv.c_str(), "w");
        if (flight_csv) std::fprintf(flight_csv, "frame,tau_M,r_M,theta_deg,t_bl_M\n");
    }
    for (int fr = 0; fr < n_frames; ++fr) {
        if (flight_movie) {
            const double rm = flight_->chart().r_minus();
            const double r_start = flight_->samples().front().r;
            const double r_end = flight_->samples().back().r;
            const double u = n_frames > 1 ? static_cast<double>(fr) / (n_frames - 1) : 0.0;
            const double rr = rm + (r_start - rm) * std::pow((r_end - rm) / (r_start - rm), u);
            const double t = fr == 0 ? 0.0 : fr == n_frames - 1 ? flight_->tau_end()
                                                                 : flight_->tau_at_radius(rr);
            live_.flight_tau = std::isnan(t) ? flight_->tau_end() : t;
            const InfallSample s = flight_->at(live_.flight_tau);
            if (flight_csv)
                std::fprintf(flight_csv, "%d,%.10g,%.10g,%.8g,%.10g\n", fr, s.tau, s.r,
                             s.theta * 180.0 / 3.14159265358979323846, s.t_bl);
            const uint32_t idx = pc.image_index;
            pc = make_push_constants(0.0f);
            pc.image_index = idx;
            if (!shot.empty()) {
                char tag[16];
                std::snprintf(tag, sizeof(tag), "_%04d", fr);
                const size_t dot = shot.find_last_of('.');
                opts_.screenshot_path = (dot == std::string::npos)
                                            ? shot + tag
                                            : shot.substr(0, dot) + tag + shot.substr(dot);
            }
        } else if (opts_.movie_frames > 0) {
            live_.flare_t = opts_.flare_t + static_cast<float>(fr) * opts_.movie_dt;
            if (!shot.empty()) {
                char idx[16];
                std::snprintf(idx, sizeof(idx), "_%04d", fr);
                const size_t dot = shot.find_last_of('.');
                opts_.screenshot_path = (dot == std::string::npos)
                                            ? shot + idx
                                            : shot.substr(0, dot) + idx + shot.substr(dot);
            }
        }
        submit([&](VkCommandBuffer cmd) {
            Context::image_barrier(cmd, target_, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_SHADER_WRITE_BIT);
            if (post_enabled()) {
                record_frame_params(cmd, make_frame_params());
                prepare_post_images(cmd);
            }
        });
        for (uint32_t s = 0; s < spp; ++s) {
            pc.accum_index = s;
            pc.accum_weight = 1.0f / static_cast<float>(s + 1);
            for (uint32_t y0 = 0; y0 < target_h_; y0 += band_rows) {
                const uint32_t rows = std::min(band_rows, target_h_ - y0);
                submit([&](VkCommandBuffer cmd) {
                    compute_barrier(cmd); // previous sample's accumulator writes
                    record_trace(cmd, pc, y0, rows);
                });
            }
            if (spp > 1 && (s + 1) % 16 == 0)
                std::printf("[app] offscreen sample %u/%u\n", s + 1, spp);
        }
        if (post_enabled()) {
            submit([&](VkCommandBuffer cmd) {
                compute_barrier(cmd);
                record_post(cmd);
            });
        }
        // GENERAL → TRANSFER_SRC for the readback
        {
            vkResetCommandBuffer(f.cmd, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(f.cmd, &bi);
            Context::image_barrier(f.cmd, target_, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_ACCESS_TRANSFER_READ_BIT);
            vkEndCommandBuffer(f.cmd);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &f.cmd;
            check(vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE),
                  "vkQueueSubmit(offscreen barrier)");
            ctx_->wait_idle();
        }
        if (!opts_.screenshot_path.empty()) save_screenshot();
    }
    opts_.screenshot_path = shot;
    if (flight_csv) std::fclose(flight_csv);
    std::printf("[app] offscreen render %ux%u complete, validation errors: %d, "
                "warnings: %d\n",
                target_w_, target_h_, Context::validation_errors.load(),
                Context::validation_warnings.load());
    return Context::validation_errors.load();
}

int App::run() {
    if (opts_.offscreen) return run_offscreen();
    const auto t0 = std::chrono::steady_clock::now();
    int frame_count = 0;
    int frame_slot = 0;
    auto fps_t0 = t0;
    int fps_frames = 0;

    double fps_now = 0.0;
    auto prev_frame = t0;
    while (!glfwWindowShouldClose(ctx_->window())) {
        glfwPollEvents();
        if (opts_.max_frames >= 0 && frame_count >= opts_.max_frames) break;
        {
            // Gate 18: the hot spot's observing clock (time-warped)
            const auto now = std::chrono::steady_clock::now();
            const double dt_wall = std::chrono::duration<double>(now - prev_frame).count();
            prev_frame = now;
            if (live_.flare && live_.flare_play && mode_ == UiMode::Simulation)
                live_.flare_t += static_cast<float>(std::min(dt_wall, 0.25) * live_.flare_speed);
            // Gate 24: the in-faller's proper time (time-warped)
            if (live_.flight && live_.flight_play && flight_ && mode_ == UiMode::Simulation) {
                live_.flight_tau = std::min(
                    live_.flight_tau + std::min(dt_wall, 0.25) * live_.flight_warp,
                    flight_->tau_end());
                if (live_.flight_tau >= flight_->tau_end()) live_.flight_play = false;
            }
        }

        // Esc (edge-triggered): sim → menu when the menu exists, else quit;
        // in the menu itself Esc quits.
        const bool esc =
            glfwGetKey(ctx_->window(), GLFW_KEY_ESCAPE) == GLFW_PRESS;
        if (esc && !esc_prev_) {
            if (opts_.menu && mode_ == UiMode::Simulation)
                return_to_menu();
            else
                break;
        }
        esc_prev_ = esc;

        // Scripted transition demo (Gate 9): menu → sim → menu.
        if (opts_.demo_frames > 0) {
            if (frame_count == opts_.demo_frames)
                enter_simulation(true);
            else if (frame_count == 2 * opts_.demo_frames)
                return_to_menu();
        }

        if (mode_ == UiMode::Simulation) {
            for (int k = 0; k < static_cast<int>(kDiskPresets.size()); ++k)
                if (glfwGetKey(ctx_->window(), GLFW_KEY_1 + k) == GLFW_PRESS)
                    apply_preset(k);

            // Mouse-drag orbit: horizontal = azimuth, vertical = inclination
            // (full range — above and below the disk plane; a small dead zone
            // at exactly 90° avoids the degenerate edge-on camera).
            if (!opts_.no_mouse) {
                double mx = 0.0, my = 0.0;
                glfwGetCursorPos(ctx_->window(), &mx, &my);
                const bool lmb = glfwGetMouseButton(ctx_->window(),
                                                    GLFW_MOUSE_BUTTON_LEFT) ==
                                 GLFW_PRESS;
                const bool ui_mouse = hud_ && hud_->wants_mouse();
                if (lmb && dragging_ && live_.focus_index > 0) {
                    // focused: orbit the STAR, not the hole (SimHUD derives
                    // r_cam/incl/cam_phi from these each frame)
                    live_.focus_yaw -=
                        static_cast<float>(mx - drag_x_) * 0.006f;
                    live_.focus_pitch = std::clamp(
                        live_.focus_pitch +
                            static_cast<float>(my - drag_y_) * 0.006f,
                        -1.45f, 1.45f);
                } else if (lmb && dragging_) {
                    live_.cam_phi += static_cast<float>(mx - drag_x_) * 0.005f;
                    float incl = live_.incl_deg +
                                 static_cast<float>(my - drag_y_) * 0.20f;
                    incl = std::clamp(incl, 1.0f, 179.0f);
                    if (incl > 89.3f && incl < 90.7f)
                        incl = (my - drag_y_) >= 0.0 ? 90.7f : 89.3f;
                    live_.incl_deg = incl;
                }
                dragging_ = lmb && !ui_mouse;
                drag_x_ = mx;
                drag_y_ = my;
            }
        } else {
            dragging_ = false;
        }
        // Render-scale change: rebuild the target between frames, before
        // ImGui starts building this frame's draw data (keeps the rebuild
        // from interleaving with any per-frame CPU state).
        if (target_rebuild_) {
            target_rebuild_ = false;
            live_.render_scale = std::clamp(live_.render_scale, 0.25f, 2.0f);
            ctx_->wait_idle();
            destroy_render_target();
            create_render_target();
        }

        if (hud_) hud_->new_frame(fps_now);

        auto& f = ctx_->frame(frame_slot);
        vkWaitForFences(ctx_->device(), 1, &f.in_flight, VK_TRUE, UINT64_MAX);

        uint32_t sc_index = 0;
        VkResult acq = vkAcquireNextImageKHR(ctx_->device(), ctx_->swapchain(),
                                             UINT64_MAX, f.image_available,
                                             VK_NULL_HANDLE, &sc_index);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            ctx_->recreate_swapchain();
            destroy_render_target();
            create_render_target();
            if (hud_) hud_->on_swapchain_recreated();
            continue;
        }
        check(acq, "vkAcquireNextImageKHR");
        vkResetFences(ctx_->device(), 1, &f.in_flight);

        vkResetCommandBuffer(f.cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(f.cmd, &bi);
        const float t = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        record(f.cmd, sc_index, t);
        vkEndCommandBuffer(f.cmd);

        const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkSemaphore present_sem = ctx_->render_finished(sc_index);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &f.image_available;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &f.cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &present_sem;
        check(vkQueueSubmit(ctx_->queue(), 1, &si, f.in_flight), "vkQueueSubmit");

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &present_sem;
        pi.swapchainCount = 1;
        VkSwapchainKHR sc = ctx_->swapchain();
        pi.pSwapchains = &sc;
        pi.pImageIndices = &sc_index;
        VkResult pres = vkQueuePresentKHR(ctx_->queue(), &pi);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            ctx_->recreate_swapchain();
            destroy_render_target();
            create_render_target();
            if (hud_) hud_->on_swapchain_recreated();
        } else {
            check(pres, "vkQueuePresentKHR");
        }

        ++frame_count;
        ++fps_frames;
        last_sc_index_ = sc_index;
        frame_slot = (frame_slot + 1) % vk::kFramesInFlight;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - fps_t0).count() > 2.0) {
            fps_now = fps_frames /
                      std::chrono::duration<double>(now - fps_t0).count();
            std::printf("[app] %.1f fps (scale %.2f)\n", fps_now,
                        live_.render_scale);
            fps_t0 = now;
            fps_frames = 0;

            // Auto resolution: hold ~13 fps (user-chosen) by scaling the
            // internal render target. Cost per frame ~ pixels ~ scale², so
            // sqrt(fps/target) is the equalising step; blend 70% toward it
            // and ignore <5% moves to avoid oscillation. Menu mode is a
            // cheap blit — never scale on its numbers.
            if (live_.auto_res && mode_ == UiMode::Simulation &&
                fps_now > 0.0) {
                if (autores_skip_) {
                    // this sample includes the rebuild stall — not a real
                    // performance measurement
                    autores_skip_ = false;
                } else {
                    const float target = 13.0f;
                    float want =
                        live_.render_scale *
                        std::clamp(std::sqrt(static_cast<float>(fps_now) /
                                             target),
                                   0.74f, 1.35f); // bounded step per sample
                    want = std::clamp(want, 0.25f, 2.0f);
                    const float next = live_.render_scale +
                                       0.7f * (want - live_.render_scale);
                    if (std::fabs(next - live_.render_scale) >
                        0.05f * live_.render_scale) {
                        live_.render_scale = next;
                        request_target_rebuild();
                        autores_skip_ = true;
                    }
                }
            }
        }
    }

    if (!opts_.screenshot_path.empty()) save_screenshot();
    ctx_->wait_idle();

    std::printf("[app] frames rendered: %d, validation errors: %d, warnings: %d\n",
                frame_count, Context::validation_errors.load(),
                Context::validation_warnings.load());
    return Context::validation_errors.load();
}

} // namespace sgr
