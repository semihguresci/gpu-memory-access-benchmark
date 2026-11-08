#pragma once

#include "utils/gpu_timestamp_timer.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace VulkanComputeUtils {

struct DescriptorBufferBindingUpdate {
    uint32_t binding = 0;
    VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo buffer_info{};
};

bool read_binary_file(const std::string& path, std::vector<char>& out_data);

bool create_shader_module(VkDevice device, const std::vector<char>& spirv_code, VkShaderModule& out_shader_module);

bool load_shader_module_from_file(VkDevice device, const std::string& path, VkShaderModule& out_shader_module);

bool create_descriptor_set_layout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                  VkDescriptorSetLayout& out_layout);

bool create_descriptor_pool(VkDevice device, const std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets,
                            VkDescriptorPool& out_pool);

bool allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptor_pool,
                             VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& out_set);

void update_descriptor_set_buffers(VkDevice device, VkDescriptorSet descriptor_set,
                                   const std::vector<DescriptorBufferBindingUpdate>& updates);

bool create_pipeline_layout(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
                            const std::vector<VkPushConstantRange>& push_constant_ranges,
                            VkPipelineLayout& out_pipeline_layout);

bool create_pipeline_layout(VkDevice device, const std::vector<VkDescriptorSetLayout>& descriptor_set_layouts,
                            VkPipelineLayout& out_pipeline_layout);

bool create_compute_pipeline(VkDevice device, VkShaderModule shader_module, VkPipelineLayout pipeline_layout,
                             const char* entry_point, VkPipeline& out_pipeline);

float query_timestamp_period(VkPhysicalDevice physical_device);

std::string resolve_shader_path(const std::string& user_path, const std::string& shader_name);

uint32_t compute_group_count_1d(uint32_t element_count, uint32_t local_size_x);

bool validate_compute_requirements(const VkPhysicalDeviceLimits& limits, const std::array<uint32_t, 3>& local_size,
                                   uint32_t shared_memory_bytes, std::string& error);

// For a compute pipeline with one descriptor set of storage buffers.
bool validate_storage_buffer_descriptor_requirements(const VkPhysicalDeviceLimits& limits, uint32_t descriptor_count,
                                                     std::string& error);

void record_transfer_write_to_compute_read_write_barrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                                                         VkDeviceSize size);

void record_compute_write_to_transfer_read_barrier(VkCommandBuffer command_buffer, VkBuffer buffer, VkDeviceSize size);

void record_compute_write_to_host_read_barrier(VkCommandBuffer command_buffer);
void record_compute_write_to_host_read_barrier(VkCommandBuffer command_buffer, VkBuffer buffer, VkDeviceSize size);
void record_transfer_write_to_host_read_barrier(VkCommandBuffer command_buffer, VkBuffer buffer, VkDeviceSize size);

} // namespace VulkanComputeUtils
