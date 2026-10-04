#pragma once
// Bindless descriptor table: one global set with
// UPDATE_AFTER_BIND storage-image and sampled-image arrays. Resources are
// registered once and addressed by index through push constants.

#include <volk.h>

#include <cstdint>

namespace sgr::vk {

class Context;

inline constexpr uint32_t kMaxStorageImages = 16;
inline constexpr uint32_t kMaxSampledImages = 16;

class BindlessTable {
public:
    explicit BindlessTable(Context& ctx);
    ~BindlessTable();
    BindlessTable(const BindlessTable&) = delete;
    BindlessTable& operator=(const BindlessTable&) = delete;

    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }

    // Register resources at a slot (update-after-bind: legal while in flight
    // as long as the slot isn't concurrently accessed).
    void write_storage_image(uint32_t slot, VkImageView view);
    void write_sampled_image(uint32_t slot, VkImageView view, VkSampler sampler);
    // Binding 2: the per-frame parameter storage buffer (one, written once;
    // contents updated with vkCmdUpdateBuffer each frame).
    void write_storage_buffer(VkBuffer buffer, VkDeviceSize range);

private:
    VkDevice device_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

} // namespace sgr::vk
