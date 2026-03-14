#include "utils/vulkan_compute_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace VulkanComputeUtils {

bool read_binary_file(const std::string& path, std::vector<char>& out_data) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << "\n";
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        std::cerr << "File is empty or unreadable: " << path << "\n";
        return false;
    }

    out_data.resize(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(out_data.data(), size);
    return file.good();
}

bool create_shader_module(VkDevice device, const std::vector<char>& spirv_code, VkShaderModule& out_shader_module) {
    if (spirv_code.empty() || spirv_code.size() % sizeof(uint32_t) != 0) {
        return false;
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(spirv_code.data());

    return vkCreateShaderModule(device, &create_info, nullptr, &out_shader_module) == VK_SUCCESS;
}

bool load_shader_module_from_file(VkDevice device, const std::string& path, VkShaderModule& out_shader_module) {
    std::vector<char> spirv_code;
    if (!read_binary_file(path, spirv_code)) {
        return false;
    }

    return create_shader_module(device, spirv_code, out_shader_module);
}

bool create_descriptor_set_layout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                  VkDescriptorSetLayout& out_layout) {
    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.bindingCount = static_cast<uint32_t>(bindings.size());
    create_info.pBindings = bindings.empty() ? nullptr : bindings.data();

    return vkCreateDescriptorSetLayout(device, &create_info, nullptr, &out_layout) == VK_SUCCESS;
}

bool create_descriptor_pool(VkDevice device, const std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets,
                            VkDescriptorPool& out_pool) {
    VkDescriptorPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.maxSets = max_sets;
    create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    create_info.pPoolSizes = pool_sizes.empty() ? nullptr : pool_sizes.data();

    return vkCreateDescriptorPool(device, &create_info, nullptr, &out_pool) == VK_SUCCESS;
}

bool allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptor_pool,
                             VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& out_set) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout;

    return vkAllocateDescriptorSets(device, &alloc_info, &out_set) == VK_SUCCESS;
}

void update_descriptor_set_buffers(VkDevice device, VkDescriptorSet descriptor_set,
                                   const std::vector<DescriptorBufferBindingUpdate>& updates) {
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(updates.size());

    for (const auto& update : updates) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = update.binding;
        write.descriptorCount = 1;
        write.descriptorType = update.descriptor_type;
        write.pBufferInfo = &update.buffer_info;
        writes.push_back(write);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool create_pipeline_layout(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
                            const std::vector<VkPushConstantRange>& push_constant_ranges,
                            VkPipelineLayout& out_pipeline_layout) {
    VkPipelineLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    create_info.pSetLayouts = descriptor_set_layouts.empty() ? nullptr : descriptor_set_layouts.data();
    create_info.pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size());
    create_info.pPushConstantRanges = push_constant_ranges.empty() ? nullptr : push_constant_ranges.data();

    return vkCreatePipelineLayout(device, &create_info, nullptr, &out_pipeline_layout) == VK_SUCCESS;
}

bool create_pipeline_layout(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
                            VkPipelineLayout& out_pipeline_layout) {
    const std::vector<VkPushConstantRange> no_push_constants{};
    return create_pipeline_layout(device, descriptor_set_layouts, no_push_constants, out_pipeline_layout);
}

bool create_compute_pipeline(VkDevice device, VkShaderModule shader_module, VkPipelineLayout pipeline_layout,
                             const char* entry_point, VkPipeline& out_pipeline) {
    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = entry_point;

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;

    return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out_pipeline) == VK_SUCCESS;
}

float query_timestamp_period(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    return props.limits.timestampPeriod;
}

std::string resolve_shader_path(const std::string& user_path, const std::string& shader_name) {
    if (!user_path.empty() && std::filesystem::exists(user_path)) {
        return user_path;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.reserve(12);

#if defined(PROJECT_SHADER_BINARY_DIR)
    candidates.emplace_back(std::filesystem::path(PROJECT_SHADER_BINARY_DIR) / shader_name);
#endif

#if defined(PROJECT_SHADER_SOURCE_DIR)
    candidates.emplace_back(std::filesystem::path(PROJECT_SHADER_SOURCE_DIR) / shader_name);
#endif

    candidates.emplace_back(std::filesystem::path("build") / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("build") / "windows-x64" / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("build") / "debug" / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("build") / "release" / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("out") / "build" / "x64-Debug" / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("out") / "build" / "x64-Release" / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("build_cli11") / "shaders" / shader_name);
    candidates.emplace_back(std::filesystem::path("shaders") / shader_name);
    candidates.emplace_back(std::filesystem::path("..") / "shaders" / shader_name);

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.generic_string();
        }
    }

    return {};
}

uint32_t compute_group_count_1d(uint32_t element_count, uint32_t local_size_x) {
    if (element_count == 0U || local_size_x == 0U) {
        return 0U;
    }

    return (element_count + local_size_x - 1U) / local_size_x;
}

void record_transfer_write_to_compute_read_write_barrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                                                         VkDeviceSize size) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0U;
    barrier.size = size;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 1U,
                         &barrier, 0U, nullptr);
}

void record_compute_write_to_transfer_read_barrier(VkCommandBuffer command_buffer, VkBuffer buffer, VkDeviceSize size) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0U;
    barrier.size = size;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U,
                         nullptr, 1U, &barrier, 0U, nullptr);
}

} // namespace VulkanComputeUtils
