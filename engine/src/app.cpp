#include "app.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/metric.hpp"
#include "ui/SimHUD.hpp"

namespace sgr {

using vk::check;
using vk::Context;

App::App(const AppOptions& opts) : opts_(opts) {
    ctx_ = std::make_unique<Context>(opts.width, opts.height,
                                     "SGR A* — scientific simulator",
                                     opts.validation);
    bindless_ = std::make_unique<vk::BindlessTable>(*ctx_);
    pipeline_ = std::make_unique<vk::ComputePipeline>(
        *ctx_, bindless_->layout(),
        std::string(SGR_SHADER_DIR) + "/" + opts.shader + ".comp.glsl",
        sizeof(PushConstants));
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
    if (opts_.t_peak_k > 0.0f) live_.t_peak_k = opts_.t_peak_k;
    if (opts_.rot_speed >= 0.0f) live_.rot_speed = opts_.rot_speed;
    // design range is 0-1 (the shader clamps too; beyond 1 the filament
    // mix() extrapolates into harsh negative-depth lanes)
    if (opts_.turb >= 0.0f) live_.turb = std::min(opts_.turb, 1.0f);
    live_.black_sky = opts_.black_sky;
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

    if (opts_.shader == "disk") load_luts();

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

void App::enter_simulation(bool science, bool scope) {
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
    live_.lut_slot = p.lut_slot;
    live_.a_star = p.a_star;
    live_.incl_deg = p.inclination_deg;
    live_.disk_r_out = p.disk_outer_r;
    live_.exposure_ev = p.exposure_ev;
    live_.t_peak_k = p.t_peak_k;
    live_.r_cam = p.r_cam;
    live_.turb = p.turb;
}

App::Texture App::create_texture_png(const std::string& path,
                                     VkImageUsageFlags usage,
                                     VkImageLayout final_layout) {
    Texture tex;
    int w = 0, h = 0, comp = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!px) {
        std::fprintf(stderr, "[app] missing texture %s\n", path.c_str());
        return tex;
    }
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
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
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 4ull * w * h;
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
    std::memcpy(sinfo.pMappedData, px, 4ull * w * h);
    stbi_image_free(px);

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
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(ctx_->device(), &vci, nullptr, &tex.view),
              "vkCreateImageView(texture)");
    }
    tex.w = w;
    tex.h = h;
    owned_textures_.push_back(tex);
    return tex;
}

void App::load_luts() {
    const char* names[] = {"disk_colormap", "firestorm", "gargantua", "arctic",
                           "nebula", "cinematic"};
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(ctx_->device(), &sci, nullptr, &lut_sampler_),
          "vkCreateSampler(lut)");

    for (uint32_t slot = 0; slot < 6; ++slot) {
        const std::string path = std::string(SGR_ASSET_DIR) + "/luts/" +
                                 names[slot] + ".png";
        const Texture lut = create_texture_png(
            path, VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (lut.w == 0) continue;
        bindless_->write_sampled_image(slot, lut.view, lut_sampler_);
        ++n_luts_;
    }
    std::printf("[app] loaded %zu LUTs\n", n_luts_);
}

App::~App() {
    if (ctx_) ctx_->wait_idle();
    hud_.reset();
    for (auto& t : owned_textures_) {
        if (t.view) vkDestroyImageView(ctx_->device(), t.view, nullptr);
        if (t.image) vmaDestroyImage(ctx_->allocator(), t.image, t.alloc);
    }
    if (lut_sampler_) vkDestroySampler(ctx_->device(), lut_sampler_, nullptr);
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

    bindless_->write_storage_image(0, target_view_);
    target_initialised_ = false;
}

void App::destroy_render_target() {
    if (target_view_) vkDestroyImageView(ctx_->device(), target_view_, nullptr);
    if (target_) vmaDestroyImage(ctx_->allocator(), target_, target_alloc_);
    target_view_ = VK_NULL_HANDLE;
    target_ = VK_NULL_HANDLE;
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
    // storage image → GENERAL (first use: from UNDEFINED)
    Context::image_barrier(
        cmd, target_,
        target_initialised_ ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                            : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    target_initialised_ = true;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_->pipeline());
    const VkDescriptorSet set = bindless_->set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_->layout(), 0, 1, &set, 0, nullptr);

    PushConstants pc = make_push_constants(t);
    vkCmdPushConstants(cmd, pipeline_->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdDispatch(cmd, (target_w_ + 7) / 8, (target_h_ + 7) / 8, 1);

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
    pc.disk_r_out = live_.disk_r_out;
    pc.exposure_ev = live_.exposure_ev;
    pc.shade_mode = opts_.shader == "disk" ? live_.lut_slot : opts_.shade_mode;
    pc.y_offset = 0;
    pc.t_peak_k = live_.t_peak_k;
    pc.turb = live_.turb;
    pc.cam_phi = live_.cam_phi;
    // 1.0 on the slider = real Sgr A* wall-clock: one geometric time unit
    // (GM/c³) is 21.17 s for M = 4.297e6 Msun.
    pc.time_scale = live_.rot_speed / 21.17f;
    pc.sky_off = live_.black_sky ? 1u : 0u;
    pc.fwd_r = live_.fwd_r;
    pc.fwd_th = live_.fwd_th;
    pc.fwd_ph = live_.fwd_ph;
    pc.rgt_r = live_.rgt_r;
    pc.rgt_th = live_.rgt_th;
    pc.rgt_ph = live_.rgt_ph;
    pc.up_r = live_.up_r;
    pc.up_th = live_.up_th;
    pc.up_ph = live_.up_ph;

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

    bool first = true;
    for (uint32_t y0 = 0; y0 < target_h_; y0 += band_rows) {
        vkResetCommandBuffer(f.cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(f.cmd, &bi);
        if (first) {
            Context::image_barrier(f.cmd, target_, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_SHADER_WRITE_BIT);
            first = false;
        }
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline_->pipeline());
        const VkDescriptorSet set = bindless_->set();
        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_->layout(), 0, 1, &set, 0, nullptr);
        pc.y_offset = y0;
        vkCmdPushConstants(f.cmd, pipeline_->layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        const uint32_t rows = std::min(band_rows, target_h_ - y0);
        vkCmdDispatch(f.cmd, (target_w_ + 7) / 8, (rows + 7) / 8, 1);
        vkEndCommandBuffer(f.cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &f.cmd;
        check(vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE),
              "vkQueueSubmit(offscreen band)");
        ctx_->wait_idle();
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
    while (!glfwWindowShouldClose(ctx_->window())) {
        glfwPollEvents();
        if (opts_.max_frames >= 0 && frame_count >= opts_.max_frames) break;

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
            for (int k = 0; k < 6; ++k)
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
