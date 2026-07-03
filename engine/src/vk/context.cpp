#include "vk/context.hpp"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#include <cstdio>
#include <stdexcept>

namespace sgr::vk {

std::atomic<int> Context::validation_errors{0};
std::atomic<int> Context::validation_warnings{0};

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: VkResult " +
                                 std::to_string(static_cast<int>(r)));
}

namespace {
VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        Context::validation_errors++;
        std::fprintf(stderr, "[VULKAN ERROR] %s\n", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        Context::validation_warnings++;
        std::fprintf(stderr, "[vulkan warn] %s\n", data->pMessage);
    }
    return VK_FALSE;
}
} // namespace

Context::Context(int width, int height, const char* title,
                 bool enable_validation) {
    if (volkInitialize() != VK_SUCCESS)
        throw std::runtime_error("volkInitialize failed (no Vulkan loader)");

    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) throw std::runtime_error("glfwCreateWindow failed");

    vkb::InstanceBuilder ib;
    ib.set_app_name("sgr-a-research-sim")
        .require_api_version(1, 2, 0)
        .set_debug_callback(debug_cb);
    if (enable_validation) ib.request_validation_layers(true);
    auto inst_ret = ib.build();
    if (!inst_ret)
        throw std::runtime_error("instance: " + inst_ret.error().message());
    vkb::Instance vkb_inst = inst_ret.value();
    instance_ = vkb_inst.instance;
    messenger_ = vkb_inst.debug_messenger;
    validation_enabled_ = enable_validation;
    volkLoadInstance(instance_);

    check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
          "glfwCreateWindowSurface");

    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
    features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    features.shaderFloat64 = VK_TRUE; // double-precision geodesics (Phase 7)

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.descriptorIndexing = VK_TRUE;
    f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;

    vkb::PhysicalDeviceSelector sel(vkb_inst);
    auto phys_ret = sel.set_surface(surface_)
                        .set_minimum_version(1, 2)
                        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                        .set_required_features(features)
                        .set_required_features_12(f12)
                        .select();
    if (!phys_ret)
        throw std::runtime_error("physical device: " + phys_ret.error().message());
    physical_ = phys_ret.value().physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_, &props);
    std::printf("[vk] device: %s (api %u.%u.%u)\n", props.deviceName,
                VK_VERSION_MAJOR(props.apiVersion),
                VK_VERSION_MINOR(props.apiVersion),
                VK_VERSION_PATCH(props.apiVersion));

    auto dev_ret = vkb::DeviceBuilder(phys_ret.value()).build();
    if (!dev_ret)
        throw std::runtime_error("device: " + dev_ret.error().message());
    vkb::Device vkb_dev = dev_ret.value();
    device_ = vkb_dev.device;
    volkLoadDevice(device_);

    queue_ = vkb_dev.get_queue(vkb::QueueType::graphics).value();
    queue_family_ = vkb_dev.get_queue_index(vkb::QueueType::graphics).value();

    VmaVulkanFunctions vf{};
    vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.instance = instance_;
    aci.physicalDevice = physical_;
    aci.device = device_;
    aci.pVulkanFunctions = &vf;
    aci.vulkanApiVersion = VK_API_VERSION_1_2;
    check(vmaCreateAllocator(&aci, &allocator_), "vmaCreateAllocator");

    create_swapchain();

    for (auto& f : frames_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = queue_family_;
        check(vkCreateCommandPool(device_, &pci, nullptr, &f.pool),
              "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = f.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &cai, &f.cmd),
              "vkAllocateCommandBuffers");
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(vkCreateSemaphore(device_, &sci, nullptr, &f.image_available), "sem");
        check(vkCreateSemaphore(device_, &sci, nullptr, &f.render_finished), "sem");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(device_, &fci, nullptr, &f.in_flight), "fence");
    }
}

void Context::create_swapchain() {
    vkb::SwapchainBuilder sb(physical_, device_, surface_);
    auto sc_ret = sb.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM,
                                         VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
                      .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                      .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                      .build();
    if (!sc_ret)
        throw std::runtime_error("swapchain: " + sc_ret.error().message());
    vkb::Swapchain vkb_sc = sc_ret.value();
    swapchain_ = vkb_sc.swapchain;
    swapchain_format_ = vkb_sc.image_format;
    swapchain_extent_ = vkb_sc.extent;
    sc_images_ = vkb_sc.get_images().value();
}

void Context::destroy_swapchain() {
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void Context::recreate_swapchain() {
    wait_idle();
    destroy_swapchain();
    create_swapchain();
}

void Context::wait_idle() const { vkDeviceWaitIdle(device_); }

void Context::image_barrier(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout,
                            VkPipelineStageFlags src_stage,
                            VkAccessFlags src_access,
                            VkPipelineStageFlags dst_stage,
                            VkAccessFlags dst_access) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr,
                         1, &b);
}

Context::~Context() {
    if (device_) {
        wait_idle();
        for (auto& f : frames_) {
            if (f.in_flight) vkDestroyFence(device_, f.in_flight, nullptr);
            if (f.image_available) vkDestroySemaphore(device_, f.image_available, nullptr);
            if (f.render_finished) vkDestroySemaphore(device_, f.render_finished, nullptr);
            if (f.pool) vkDestroyCommandPool(device_, f.pool, nullptr);
        }
        destroy_swapchain();
        if (allocator_) vmaDestroyAllocator(allocator_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) {
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (messenger_) {
            auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (fn) fn(instance_, messenger_, nullptr);
        }
        vkDestroyInstance(instance_, nullptr);
    }
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

} // namespace sgr::vk
