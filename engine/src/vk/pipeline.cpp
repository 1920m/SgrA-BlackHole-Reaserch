#include "vk/pipeline.hpp"

#include <shaderc/shaderc.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "vk/context.hpp"

namespace sgr::vk {

namespace {
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open shader " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
} // namespace

ComputePipeline::ComputePipeline(Context& ctx, VkDescriptorSetLayout set_layout,
                                 const std::string& glsl_path,
                                 uint32_t push_constant_size)
    : device_(ctx.device()) {
    const std::string src = read_file(glsl_path);

    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,
                              shaderc_env_version_vulkan_1_2);
    const auto result = compiler.CompileGlslToSpv(
        src, shaderc_compute_shader, glsl_path.c_str(), opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("shader compile failed:\n" +
                                 result.GetErrorMessage());
    const std::vector<uint32_t> spirv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode = spirv.data();
    VkShaderModule module;
    check(vkCreateShaderModule(device_, &smci, nullptr, &module),
          "vkCreateShaderModule");

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = push_constant_size ? 1 : 0;
    plci.pPushConstantRanges = &pcr;
    check(vkCreatePipelineLayout(device_, &plci, nullptr, &layout_),
          "vkCreatePipelineLayout");

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = layout_;
    check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                   &pipeline_),
          "vkCreateComputePipelines");
    vkDestroyShaderModule(device_, module, nullptr);
}

ComputePipeline::~ComputePipeline() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
}

} // namespace sgr::vk
