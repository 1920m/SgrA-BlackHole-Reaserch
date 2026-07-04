#include "app.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
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

    live_.science_mode = opts_.science;
    if (opts_.shader == "disk") load_luts();
    if (opts_.hud) hud_ = std::make_unique<SimHUD>(*ctx_, *this);
}

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
}

void App::load_luts() {
    const char* names[] = {"disk_colormap", "firestorm", "gargantua", "arctic",
                           "nebula"};
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(ctx_->device(), &sci, nullptr, &lut_sampler_),
          "vkCreateSampler(lut)");

    for (uint32_t slot = 0; slot < 5; ++slot) {
        const std::string path = std::string(SGR_ASSET_DIR) + "/luts/" +
                                 names[slot] + ".png";
        int w = 0, h = 0, comp = 0;
        unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 4);
        if (!px) {
            std::fprintf(stderr, "[app] missing LUT %s\n", path.c_str());
            continue;
        }
        LutTexture lut;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        check(vmaCreateImage(ctx_->allocator(), &ici, &aci, &lut.image,
                             &lut.alloc, nullptr),
              "vmaCreateImage(lut)");

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
              "vmaCreateBuffer(lut staging)");
        std::memcpy(sinfo.pMappedData, px, 4ull * w * h);
        stbi_image_free(px);

        auto& f = ctx_->frame(0);
        vkResetCommandBuffer(f.cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(f.cmd, &bi);
        Context::image_barrier(f.cmd, lut.image, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {static_cast<uint32_t>(w),
                              static_cast<uint32_t>(h), 1};
        vkCmdCopyBufferToImage(f.cmd, staging, lut.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        Context::image_barrier(f.cmd, lut.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_READ_BIT);
        vkEndCommandBuffer(f.cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &f.cmd;
        vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
        ctx_->wait_idle();
        vmaDestroyBuffer(ctx_->allocator(), staging, salloc);

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = lut.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(ctx_->device(), &vci, nullptr, &lut.view),
              "vkCreateImageView(lut)");
        bindless_->write_sampled_image(slot, lut.view, lut_sampler_);
        luts_.push_back(lut);
    }
    std::printf("[app] loaded %zu LUTs\n", luts_.size());
}

App::~App() {
    if (ctx_) ctx_->wait_idle();
    hud_.reset();
    for (auto& l : luts_) {
        if (l.view) vkDestroyImageView(ctx_->device(), l.view, nullptr);
        if (l.image) vmaDestroyImage(ctx_->allocator(), l.image, l.alloc);
    }
    if (lut_sampler_) vkDestroySampler(ctx_->device(), lut_sampler_, nullptr);
    destroy_render_target();
}

void App::create_render_target() {
    if (opts_.offscreen) {
        target_w_ = static_cast<uint32_t>(opts_.width);
        target_h_ = static_cast<uint32_t>(opts_.height);
    } else {
        const float s = std::clamp(opts_.render_scale, 0.2f, 1.0f);
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

void App::record(VkCommandBuffer cmd, uint32_t sc_index, float t) {
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
    stbi_write_png(opts_.screenshot_path.c_str(), target_w_, target_h_, 3,
                   rgb.data(), 3 * target_w_);
    std::printf("[app] screenshot written: %s\n", opts_.screenshot_path.c_str());
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
        if (glfwGetKey(ctx_->window(), GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        for (int k = 0; k < 5; ++k)
            if (glfwGetKey(ctx_->window(), GLFW_KEY_1 + k) == GLFW_PRESS)
                apply_preset(k);
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
            std::printf("[app] %.1f fps\n", fps_now);
            fps_t0 = now;
            fps_frames = 0;
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
