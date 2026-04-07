#include "experiments/prefix_sum_scan_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "22_prefix_sum_scan";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTargetLogicalCount = 65536U;
constexpr uint32_t kMaxBlockElements = kWorkgroupSize * 8U;
constexpr uint32_t kReferenceBlockCount = 256U;
constexpr uint32_t kDispatchCount = 3U;
constexpr uint32_t kSourcePatternMultiplier = 17U;
constexpr uint32_t kSourcePatternOffset = 23U;
constexpr uint32_t kSourcePatternModulus = 251U;
constexpr uint32_t kSentinelValue = 0xA5A5A5A5U;
constexpr std::array<uint32_t, 4> kItemsPerThreadValues = {1U, 2U, 4U, 8U};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource output_buffer{};
    BufferResource block_totals_buffer{};
    BufferResource block_prefix_buffer{};
    BufferResource block_scan_scratch_buffer{};
    void* input_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
    void* block_totals_mapped_ptr = nullptr;
    void* block_prefix_mapped_ptr = nullptr;
};

struct ScanPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet main_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSet block_descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct ApplyPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t element_count = 0U;
    uint32_t block_elements = 0U;
    uint32_t block_count = 0U;
    uint32_t items_per_thread = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_variant_name(uint32_t items_per_thread) {
    return "items_per_thread_" + std::to_string(items_per_thread);
}

std::string make_case_name(uint32_t items_per_thread, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(items_per_thread) + "_elements_" +
           std::to_string(logical_count);
}

uint32_t source_pattern_value(uint32_t index) {
    return ((index * kSourcePatternMultiplier) + kSourcePatternOffset) % kSourcePatternModulus;
}

void fill_source_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_sentinel_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, kSentinelValue);
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes) {
    const uint64_t fixed_overhead_bytes =
        (static_cast<uint64_t>(kReferenceBlockCount) * sizeof(uint32_t) * 2U) + sizeof(uint32_t);
    if (max_buffer_bytes <= fixed_overhead_bytes) {
        return 0U;
    }

    const uint64_t main_buffer_budget_bytes = static_cast<uint64_t>(max_buffer_bytes) - fixed_overhead_bytes;
    const uint64_t buffer_elements = main_buffer_budget_bytes / (sizeof(uint32_t) * 2U);
    if (buffer_elements < kMaxBlockElements) {
        return 0U;
    }

    const uint64_t capped_elements = std::min<uint64_t>(buffer_elements, static_cast<uint64_t>(kTargetLogicalCount));
    const uint64_t rounded_elements = capped_elements - (capped_elements % kMaxBlockElements);
    if (rounded_elements < kMaxBlockElements) {
        return 0U;
    }

    return static_cast<uint32_t>(rounded_elements);
}

uint32_t compute_block_elements(uint32_t items_per_thread) {
    return kWorkgroupSize * items_per_thread;
}

uint32_t compute_block_count(uint32_t logical_count, uint32_t block_elements) {
    if (logical_count == 0U || block_elements == 0U) {
        return 0U;
    }

    return (logical_count + block_elements - 1U) / block_elements;
}

VkDeviceSize compute_buffer_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t compute_cpu_scan(uint32_t value, uint32_t input) {
    return value + input;
}

std::vector<uint32_t> build_reference_scan(const uint32_t* input_values, uint32_t logical_count) {
    std::vector<uint32_t> reference(logical_count, 0U);
    uint32_t running_sum = 0U;
    for (uint32_t index = 0U; index < logical_count; ++index) {
        running_sum = compute_cpu_scan(running_sum, input_values[index]);
        reference[index] = running_sum;
    }

    return reference;
}

std::vector<uint32_t> build_reference_block_totals(const uint32_t* input_values, uint32_t logical_count,
                                                   uint32_t block_elements) {
    const uint32_t block_count = compute_block_count(logical_count, block_elements);
    std::vector<uint32_t> block_totals(block_count, 0U);

    for (uint32_t block_index = 0U; block_index < block_count; ++block_index) {
        const uint32_t block_start = block_index * block_elements;
        const uint32_t block_end = std::min(logical_count, block_start + block_elements);
        uint32_t running_sum = 0U;
        for (uint32_t element_index = block_start; element_index < block_end; ++element_index) {
            running_sum = compute_cpu_scan(running_sum, input_values[element_index]);
        }
        block_totals[block_index] = running_sum;
    }

    return block_totals;
}

std::vector<uint32_t> build_reference_block_prefix(const std::vector<uint32_t>& block_totals) {
    std::vector<uint32_t> prefix(block_totals.size(), 0U);
    uint32_t running_sum = 0U;
    for (std::size_t index = 0; index < block_totals.size(); ++index) {
        running_sum = compute_cpu_scan(running_sum, block_totals[index]);
        prefix[index] = running_sum;
    }

    return prefix;
}

bool validate_values(const uint32_t* actual_values, const std::vector<uint32_t>& expected_values) {
    for (std::size_t index = 0; index < expected_values.size(); ++index) {
        if (actual_values[index] != expected_values[index]) {
            return false;
        }
    }

    return true;
}

bool create_buffer_resources(VulkanContext& context, uint32_t logical_count, uint32_t block_count,
                             BufferResources& out_resources) {
    const VkDeviceSize main_buffer_bytes = compute_buffer_bytes(logical_count);
    const VkDeviceSize block_buffer_bytes = compute_buffer_bytes(std::max(block_count, kReferenceBlockCount));
    const VkDeviceSize scratch_buffer_bytes = compute_buffer_bytes(1U);

    if (!create_buffer_resource(
            context.physical_device(), context.device(), main_buffer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create prefix sum scan input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), main_buffer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create prefix sum scan output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), block_buffer_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_totals_buffer)) {
        std::cerr << "Failed to create prefix sum scan block totals buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), block_buffer_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_prefix_buffer)) {
        std::cerr << "Failed to create prefix sum scan block prefix buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), scratch_buffer_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_scan_scratch_buffer)) {
        std::cerr << "Failed to create prefix sum scan scratch buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "prefix sum scan input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_scan_scratch_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "prefix sum scan output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_scan_scratch_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.block_totals_buffer, "prefix sum scan block totals buffer",
                           out_resources.block_totals_mapped_ptr)) {
        if (out_resources.output_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.output_buffer.memory);
            out_resources.output_mapped_ptr = nullptr;
        }
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_scan_scratch_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.block_prefix_buffer, "prefix sum scan block prefix buffer",
                           out_resources.block_prefix_mapped_ptr)) {
        if (out_resources.block_totals_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.block_totals_buffer.memory);
            out_resources.block_totals_mapped_ptr = nullptr;
        }
        if (out_resources.output_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.output_buffer.memory);
            out_resources.output_mapped_ptr = nullptr;
        }
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_scan_scratch_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_totals_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.block_prefix_mapped_ptr != nullptr && resources.block_prefix_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.block_prefix_buffer.memory);
        resources.block_prefix_mapped_ptr = nullptr;
    }

    if (resources.block_totals_mapped_ptr != nullptr && resources.block_totals_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.block_totals_buffer.memory);
        resources.block_totals_mapped_ptr = nullptr;
    }

    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }

    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.block_scan_scratch_buffer);
    destroy_buffer_resource(context.device(), resources.block_prefix_buffer);
    destroy_buffer_resource(context.device(), resources.block_totals_buffer);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_scan_descriptor_set(VulkanContext& context, VkDescriptorSet descriptor_set,
                                const BufferResource& src_buffer, const BufferResource& dst_buffer,
                                const BufferResource& totals_buffer) {
    const VkDescriptorBufferInfo src_info{
        src_buffer.buffer,
        0U,
        src_buffer.size,
    };
    const VkDescriptorBufferInfo dst_info{
        dst_buffer.buffer,
        0U,
        dst_buffer.size,
    };
    const VkDescriptorBufferInfo totals_info{
        totals_buffer.buffer,
        0U,
        totals_buffer.size,
    };

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = src_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = dst_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = totals_info,
                                                          },
                                                      });
}

void update_apply_descriptor_set(VulkanContext& context, VkDescriptorSet descriptor_set,
                                 const BufferResource& prefix_buffer, const BufferResource& output_buffer) {
    const VkDescriptorBufferInfo prefix_info{
        prefix_buffer.buffer,
        0U,
        prefix_buffer.size,
    };
    const VkDescriptorBufferInfo output_info{
        output_buffer.buffer,
        0U,
        output_buffer.size,
    };

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = prefix_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = output_info,
                                                          },
                                                      });
}

bool create_scan_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                    const BufferResources& buffers, ScanPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load prefix sum scan shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create prefix sum scan descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 2U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create prefix sum scan descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.main_descriptor_set)) {
        std::cerr << "Failed to allocate prefix sum scan main descriptor set.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.block_descriptor_set)) {
        std::cerr << "Failed to allocate prefix sum scan block descriptor set.\n";
        return false;
    }

    update_scan_descriptor_set(context, out_resources.main_descriptor_set, buffers.input_buffer, buffers.output_buffer,
                               buffers.block_totals_buffer);
    update_scan_descriptor_set(context, out_resources.block_descriptor_set, buffers.block_totals_buffer,
                               buffers.block_prefix_buffer, buffers.block_scan_scratch_buffer);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create prefix sum scan pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create prefix sum scan compute pipeline.\n";
        return false;
    }

    return true;
}

bool create_apply_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                     const BufferResources& buffers, ApplyPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load prefix sum apply shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create prefix sum apply descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create prefix sum apply descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate prefix sum apply descriptor set.\n";
        return false;
    }

    update_apply_descriptor_set(context, out_resources.descriptor_set, buffers.block_prefix_buffer,
                                buffers.output_buffer);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create prefix sum apply pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create prefix sum apply compute pipeline.\n";
        return false;
    }

    return true;
}

void destroy_scan_pipeline_resources(VulkanContext& context, ScanPipelineResources& resources) {
    if (resources.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), resources.pipeline, nullptr);
        resources.pipeline = VK_NULL_HANDLE;
    }

    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }

    if (resources.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), resources.descriptor_pool, nullptr);
        resources.descriptor_pool = VK_NULL_HANDLE;
    }

    if (resources.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), resources.descriptor_set_layout, nullptr);
        resources.descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (resources.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), resources.shader_module, nullptr);
        resources.shader_module = VK_NULL_HANDLE;
    }

    resources.main_descriptor_set = VK_NULL_HANDLE;
    resources.block_descriptor_set = VK_NULL_HANDLE;
}

void destroy_apply_pipeline_resources(VulkanContext& context, ApplyPipelineResources& resources) {
    if (resources.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), resources.pipeline, nullptr);
        resources.pipeline = VK_NULL_HANDLE;
    }

    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }

    if (resources.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), resources.descriptor_pool, nullptr);
        resources.descriptor_pool = VK_NULL_HANDLE;
    }

    if (resources.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), resources.descriptor_set_layout, nullptr);
        resources.descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (resources.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), resources.shader_module, nullptr);
        resources.shader_module = VK_NULL_HANDLE;
    }

    resources.descriptor_set = VK_NULL_HANDLE;
}

void record_compute_write_to_compute_read_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         1U, &barrier, 0U, nullptr, 0U, nullptr);
}

double compute_effective_gbps(uint32_t logical_count, uint32_t block_count, double dispatch_ms) {
    const uint64_t total_bytes =
        (static_cast<uint64_t>(logical_count) * 16U) + (static_cast<uint64_t>(block_count) * 16U);
    return compute_effective_gbps_from_bytes(total_bytes, dispatch_ms);
}

void append_case_notes(std::string& notes, uint32_t items_per_thread, uint32_t logical_count, uint32_t block_elements,
                       uint32_t block_count, uint32_t scan_block_count, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "scan_strategy=hierarchical_two_stage");
    append_note(notes, "inclusive_scan=true");
    append_note(notes, "workgroup_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "items_per_thread=" + std::to_string(items_per_thread));
    append_note(notes, "block_elements=" + std::to_string(block_elements));
    append_note(notes, "block_count=" + std::to_string(block_count));
    append_note(notes, "scan_block_count=" + std::to_string(scan_block_count));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
    append_note(notes, "block_totals_buffer_elements=" + std::to_string(kReferenceBlockCount));
    append_note(notes, "validation_mode=exact_uint32_wrapping");
    append_note(notes, "input_pattern=deterministic_affine_mod251");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_variant(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
                 const ScanPipelineResources& scan_resources, const ApplyPipelineResources& apply_resources,
                 uint32_t items_per_thread, uint32_t logical_count, PrefixSumScanExperimentOutput& output,
                 bool verbose_progress) {
    const uint32_t block_elements = compute_block_elements(items_per_thread);
    const uint32_t block_count = compute_block_count(logical_count, block_elements);
    const uint32_t scan_block_count = compute_block_count(block_count, kWorkgroupSize);

    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* block_totals_values = static_cast<uint32_t*>(buffers.block_totals_mapped_ptr);
    auto* block_prefix_values = static_cast<uint32_t*>(buffers.block_prefix_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || block_totals_values == nullptr ||
        block_prefix_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for items_per_thread=" << items_per_thread
                  << ".\n";
        return false;
    }

    const std::string variant_name = make_variant_name(items_per_thread);
    const std::vector<uint32_t> reference_scan = build_reference_scan(input_values, logical_count);
    const std::vector<uint32_t> reference_block_totals =
        build_reference_block_totals(input_values, logical_count, block_elements);
    const std::vector<uint32_t> reference_block_prefix = build_reference_block_prefix(reference_block_totals);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", block_elements=" << block_elements
                  << ", block_count=" << block_count << ", scan_block_count=" << scan_block_count
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const PushConstants main_push_constants{
        logical_count,
        block_elements,
        block_count,
        items_per_thread,
    };
    const PushConstants block_push_constants{
        block_count,
        kWorkgroupSize,
        1U,
        1U,
    };

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_sentinel_values(output_values, logical_count);
        fill_sentinel_values(block_totals_values, std::max(block_count, kReferenceBlockCount));
        fill_sentinel_values(block_prefix_values, std::max(block_count, kReferenceBlockCount));

        const double dispatch_ms = context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline_layout, 0U,
                                    1U, &scan_resources.main_descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, scan_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(main_push_constants)), &main_push_constants);
            vkCmdDispatch(command_buffer, block_count, 1U, 1U);

            record_compute_write_to_compute_read_barrier(command_buffer);

            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline_layout, 0U,
                                    1U, &scan_resources.block_descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, scan_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(block_push_constants)), &block_push_constants);
            vkCmdDispatch(command_buffer, 1U, 1U, 1U);

            record_compute_write_to_compute_read_barrier(command_buffer);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, apply_resources.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, apply_resources.pipeline_layout, 0U,
                                    1U, &apply_resources.descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, apply_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(main_push_constants)), &main_push_constants);
            vkCmdDispatch(command_buffer, block_count, 1U, 1U);
        });

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_values(output_values, reference_scan) &&
                             validate_values(block_prefix_values, reference_block_prefix) &&
                             validate_values(block_totals_values, reference_block_totals);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_elements=" << logical_count << ", block_elements=" << block_elements
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_sentinel_values(output_values, logical_count);
        fill_sentinel_values(block_totals_values, std::max(block_count, kReferenceBlockCount));
        fill_sentinel_values(block_prefix_values, std::max(block_count, kReferenceBlockCount));

        const double dispatch_ms = context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline_layout, 0U,
                                    1U, &scan_resources.main_descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, scan_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(main_push_constants)), &main_push_constants);
            vkCmdDispatch(command_buffer, block_count, 1U, 1U);

            record_compute_write_to_compute_read_barrier(command_buffer);

            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_resources.pipeline_layout, 0U,
                                    1U, &scan_resources.block_descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, scan_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(block_push_constants)), &block_push_constants);
            vkCmdDispatch(command_buffer, 1U, 1U, 1U);

            record_compute_write_to_compute_read_barrier(command_buffer);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, apply_resources.pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, apply_resources.pipeline_layout, 0U,
                                    1U, &apply_resources.descriptor_set, 0U, nullptr);
            vkCmdPushConstants(command_buffer, apply_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(main_push_constants)), &main_push_constants);
            vkCmdDispatch(command_buffer, block_count, 1U, 1U);
        });

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_values(output_values, reference_scan) &&
                             validate_values(block_prefix_values, reference_block_prefix) &&
                             validate_values(block_totals_values, reference_block_totals);

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;

        std::string notes;
        append_case_notes(notes, items_per_thread, logical_count, block_elements, block_count, scan_block_count,
                          correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(logical_count, block_count, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });

        dispatch_samples.push_back(dispatch_ms);
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(items_per_thread, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_outputs=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

PrefixSumScanExperimentOutput run_prefix_sum_scan_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                             const PrefixSumScanExperimentConfig& config) {
    PrefixSumScanExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "prefix sum scan experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string scan_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.scan_shader_path, "22_prefix_sum_scan_scan.comp.spv");
    const std::string add_offsets_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.add_offsets_shader_path, "22_prefix_sum_scan_add_offsets.comp.spv");
    if (scan_shader_path.empty() || add_offsets_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shaders for prefix sum scan experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = determine_logical_count(config.max_buffer_bytes);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for prefix sum scan experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t largest_block_elements = compute_block_elements(kItemsPerThreadValues.back());
    const uint32_t largest_block_count = compute_block_count(logical_count, largest_block_elements);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] scan shader: " << scan_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] add-offsets shader: " << add_offsets_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_outputs=" << logical_count
                  << ", largest_block_elements=" << largest_block_elements
                  << ", largest_block_count=" << largest_block_count
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, logical_count, compute_block_count(logical_count, kWorkgroupSize * 8U),
                                 buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* block_totals_values = static_cast<uint32_t*>(buffers.block_totals_mapped_ptr);
    auto* block_prefix_values = static_cast<uint32_t*>(buffers.block_prefix_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || block_totals_values == nullptr ||
        block_prefix_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for one or more buffers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_source_values(input_values, logical_count);
    fill_sentinel_values(output_values, logical_count);
    fill_sentinel_values(block_totals_values, kReferenceBlockCount);
    fill_sentinel_values(block_prefix_values, kReferenceBlockCount);

    ScanPipelineResources scan_resources{};
    if (!create_scan_pipeline_resources(context, scan_shader_path, buffers, scan_resources)) {
        destroy_scan_pipeline_resources(context, scan_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    ApplyPipelineResources apply_resources{};
    if (!create_apply_pipeline_resources(context, add_offsets_shader_path, buffers, apply_resources)) {
        destroy_apply_pipeline_resources(context, apply_resources);
        destroy_scan_pipeline_resources(context, scan_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const uint32_t items_per_thread : kItemsPerThreadValues) {
        if (!run_variant(context, runner, buffers, scan_resources, apply_resources, items_per_thread, logical_count,
                         output, config.verbose_progress)) {
            output.all_points_correct = false;
        }
    }

    destroy_apply_pipeline_resources(context, apply_resources);
    destroy_scan_pipeline_resources(context, scan_resources);
    destroy_buffer_resources(context, buffers);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
