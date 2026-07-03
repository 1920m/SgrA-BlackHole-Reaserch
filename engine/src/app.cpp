#include "app.hpp"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <stb_image_write.h>

#include "sgr_physics/metric.hpp"

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
}

App::~App() {
    if (ctx_) ctx_->wait_idle();
    destroy_render_target();
}

void App::create_render_target() {
    target_w_ = ctx_->swapchain_extent().width;
    target_h_ = ctx_->swapchain_extent().height;

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

    const KerrMetric metric(1.0, opts_.a_star);
    PushConstants pc{};
    pc.image_index = 0;
    pc.width = target_w_;
    pc.height = target_h_;
    pc.time_s = t;
    pc.a_star = opts_.a_star;
    pc.r_cam = opts_.r_cam;
    pc.incl_deg = opts_.incl_deg;
    pc.fov = opts_.fov;
    pc.disk_r_in = static_cast<float>(metric.r_isco(true));
    pc.disk_r_out = opts_.disk_r_out;
    pc.exposure_ev = opts_.exposure_ev;
    pc.shade_mode = opts_.shade_mode;
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
                   VK_FILTER_NEAREST);

    Context::image_barrier(cmd, sc_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
}

void App::save_screenshot() {
    ctx_->wait_idle();
    // Copy the storage image to a host-visible buffer.
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
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {target_w_, target_h_, 1};
    vkCmdCopyImageToBuffer(f.cmd, target_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buf, 1, &region);
    vkEndCommandBuffer(f.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
    ctx_->wait_idle();

    std::vector<unsigned char> rgb(3ull * target_w_ * target_h_);
    const auto* src = static_cast<const unsigned char*>(ainfo.pMappedData);
    for (size_t i = 0; i < static_cast<size_t>(target_w_) * target_h_; ++i) {
        rgb[3 * i + 0] = src[4 * i + 0];
        rgb[3 * i + 1] = src[4 * i + 1];
        rgb[3 * i + 2] = src[4 * i + 2];
    }
    stbi_write_png(opts_.screenshot_path.c_str(), target_w_, target_h_, 3,
                   rgb.data(), 3 * target_w_);
    std::printf("[app] screenshot written: %s\n", opts_.screenshot_path.c_str());
    vmaDestroyBuffer(ctx_->allocator(), buf, alloc);
}

int App::run() {
    const auto t0 = std::chrono::steady_clock::now();
    int frame_count = 0;
    int frame_slot = 0;
    auto fps_t0 = t0;
    int fps_frames = 0;

    while (!glfwWindowShouldClose(ctx_->window())) {
        glfwPollEvents();
        if (opts_.max_frames >= 0 && frame_count >= opts_.max_frames) break;

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
        } else {
            check(pres, "vkQueuePresentKHR");
        }

        ++frame_count;
        ++fps_frames;
        frame_slot = (frame_slot + 1) % vk::kFramesInFlight;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - fps_t0).count() > 2.0) {
            std::printf("[app] %.1f fps\n",
                        fps_frames /
                            std::chrono::duration<double>(now - fps_t0).count());
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
