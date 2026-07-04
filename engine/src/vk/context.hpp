#pragma once
// Vulkan context: volk + vk-bootstrap + VMA, swapchain, per-frame sync.
// Validation layers are requested in Debug (or --validation) and every
// messenger callback above INFO severity is counted — Gate 6/9 require the
// error count to be exactly zero at exit.

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace sgr::vk {

inline constexpr int kFramesInFlight = 2;

struct FrameSync {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;
};

class Context {
public:
    // Creates window + instance + device + swapchain. Throws on failure.
    Context(int width, int height, const char* title, bool enable_validation);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    GLFWwindow* window() const { return window_; }
    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physical() const { return physical_; }
    VmaAllocator allocator() const { return allocator_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return queue_family_; }

    VkSwapchainKHR swapchain() const { return swapchain_; }
    VkFormat swapchain_format() const { return swapchain_format_; }
    VkExtent2D swapchain_extent() const { return swapchain_extent_; }
    const std::vector<VkImage>& swapchain_images() const { return sc_images_; }

    FrameSync& frame(int i) { return frames_[i]; }
    // Present-wait semaphores are owned per swapchain image (their lifetime is
    // tied to the presented image, not the frame slot — validation-clean
    // semaphore reuse, see VUID-vkQueueSubmit-pSignalSemaphores-00067).
    VkSemaphore render_finished(uint32_t image_index) const {
        return render_finished_[image_index];
    }

    void recreate_swapchain();
    void wait_idle() const;

    // Validation bookkeeping (zero required by the gates).
    static std::atomic<int> validation_errors;
    static std::atomic<int> validation_warnings;
    bool validation_enabled() const { return validation_enabled_; }

    // sync2-shaped helper implemented with sync1 (hardware constraint; see
    // engine/CMakeLists.txt note). Swap body for vkCmdPipelineBarrier2 when a
    // Vulkan 1.3 driver is available.
    static void image_barrier(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout old_layout, VkImageLayout new_layout,
                              VkPipelineStageFlags src_stage,
                              VkAccessFlags src_access,
                              VkPipelineStageFlags dst_stage,
                              VkAccessFlags dst_access);

private:
    void create_swapchain();
    void destroy_swapchain();

    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent_{};
    std::vector<VkImage> sc_images_;
    std::vector<VkSemaphore> render_finished_;

    FrameSync frames_[kFramesInFlight];
    bool validation_enabled_ = false;
};

void check(VkResult r, const char* what);

} // namespace sgr::vk
