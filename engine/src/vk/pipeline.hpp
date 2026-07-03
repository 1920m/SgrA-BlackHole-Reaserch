#pragma once
// Compute pipeline from GLSL source, compiled at startup via shaderc.

#include <volk.h>

#include <cstdint>
#include <string>

namespace sgr::vk {

class Context;

// Compile GLSL compute source to SPIR-V and build the pipeline.
class ComputePipeline {
public:
    ComputePipeline(Context& ctx, VkDescriptorSetLayout set_layout,
                    const std::string& glsl_path, uint32_t push_constant_size);
    ~ComputePipeline();
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

} // namespace sgr::vk
