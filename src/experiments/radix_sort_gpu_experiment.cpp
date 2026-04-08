#include "experiments/radix_sort_gpu_experiment.hpp"

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
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "34_radix_sort_gpu";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;

// Candidate problem sizes (in elements). The scan kernel launches a single
// workgroup of 256 threads that iterates over all blocks, so num_blocks
// (= elements / kWorkgroupSize) must fit within the sequential loop budget.
// This benchmark currently samples sizes up to 64K elements (256 blocks) to
// keep the single-workgroup scan fast while still exercising meaningful
// problem sizes.
constexpr std::array<uint32_t, 5> kCandidateProblemSizes = {
    4096U, 8192U, 16384U, 32768U, 65536U,
};

// Implementation limit for the single-workgroup scan used by
// build_problem_sizes(): up to 1024 blocks (262144 elements at the current
// workgroup size).
constexpr uint32_t kMaxBlocks = 1024U;

enum class VariantKind : uint32_t {
    Radix8Bit, // 8-bit digit width, 4 passes over 32-bit keys
    Radix4Bit, // 4-bit digit width, 8 passes over 32-bit keys
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    uint32_t radix_bits;  // 8 or 4
    uint32_t radix_size;  // 256 or 16
    uint32_t radix_mask;  // 0xFF or 0x0F
    uint32_t num_passes;  // 4 or 8
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::Radix8Bit, "8bit_4pass", 8U, 256U, 0xFFU, 4U},
    {VariantKind::Radix4Bit, "4bit_8pass", 4U, 16U, 0x0FU, 8U},
}};

// Push constants for the count (histogram) shader.
struct CountPushConstants {
    uint32_t element_count = 0U;
    uint32_t num_blocks = 0U;
    uint32_t bit_offset = 0U;
    uint32_t radix_size = 0U;
    uint32_t radix_mask = 0U;
};

// Push constants for the scan (prefix sum) shader.
struct ScanPushConstants {
    uint32_t num_blocks = 0U;
    uint32_t radix_size = 0U;
};

// Push constants for the scatter shader.
struct ScatterPushConstants {
    uint32_t element_count = 0U;
    uint32_t num_blocks = 0U;
    uint32_t bit_offset = 0U;
    uint32_t radix_mask = 0U;
};

struct BufferResources {
    // Ping-pong key buffers (sorted output alternates between these).
    BufferResource keys_ping{};
    BufferResource keys_pong{};
    // histogram[digit * num_blocks + block] — per-block digit counts.
    BufferResource histogram{};
    // block_prefix[digit * num_blocks + block] — exclusive prefix per block per digit.
    BufferResource block_prefix{};
    // digit_starts[digit] — global exclusive prefix per digit.
    BufferResource digit_starts{};

    void* keys_ping_mapped_ptr = nullptr;
    void* keys_pong_mapped_ptr = nullptr;
};

// A pipeline consists of the three pass kernels (count, scan, scatter), all
// sharing the same descriptor set layout and pool.
struct PipelineResources {
    VkDescriptorSetLayout count_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool count_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet count_descriptor_set_ping = VK_NULL_HANDLE; // reads from keys_ping
    VkDescriptorSet count_descriptor_set_pong = VK_NULL_HANDLE; // reads from keys_pong
    VkShaderModule count_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout count_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline count_pipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout scan_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool scan_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet scan_descriptor_set = VK_NULL_HANDLE;
    VkShaderModule scan_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout scan_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline scan_pipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout scatter_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool scatter_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet scatter_descriptor_set_ping = VK_NULL_HANDLE; // keys_ping → keys_pong
    VkDescriptorSet scatter_descriptor_set_pong = VK_NULL_HANDLE; // keys_pong → keys_ping
    VkShaderModule scatter_shader_module = VK_NULL_HANDLE;
    VkPipelineLayout scatter_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline scatter_pipeline = VK_NULL_HANDLE;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    // Mix the index so the initial distribution exercises all digit buckets.
    uint32_t value = index;
    value ^= value >> 16U;
    value *= 0x45D9F3BU;
    value ^= value >> 16U;
    return value;
}

VkDeviceSize compute_keys_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * sizeof(uint32_t);
}

VkDeviceSize compute_histogram_span_bytes(uint32_t radix_size, uint32_t num_blocks) {
    return static_cast<VkDeviceSize>(radix_size) * static_cast<VkDeviceSize>(num_blocks) * sizeof(uint32_t);
}

VkDeviceSize compute_digit_starts_span_bytes(uint32_t radix_size) {
    return static_cast<VkDeviceSize>(radix_size) * sizeof(uint32_t);
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_problem_size, uint32_t max_radix_size,
                             uint32_t max_num_blocks, BufferResources& out_resources) {
    const VkDeviceSize keys_bytes = compute_keys_span_bytes(max_problem_size);
    const VkDeviceSize hist_bytes = compute_histogram_span_bytes(max_radix_size, max_num_blocks);
    const VkDeviceSize digit_starts_bytes = compute_digit_starts_span_bytes(max_radix_size);

    if (!create_buffer_resource(context.physical_device(), context.device(), keys_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.keys_ping)) {
        std::cerr << "[" << kExperimentId << "] Failed to create keys_ping buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), keys_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.keys_pong)) {
        std::cerr << "[" << kExperimentId << "] Failed to create keys_pong buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), hist_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.histogram)) {
        std::cerr << "[" << kExperimentId << "] Failed to create histogram buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), hist_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_prefix)) {
        std::cerr << "[" << kExperimentId << "] Failed to create block_prefix buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), digit_starts_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.digit_starts)) {
        std::cerr << "[" << kExperimentId << "] Failed to create digit_starts buffer.\n";
        return false;
    }

    if (!map_buffer_memory(context, out_resources.keys_ping, "radix sort keys_ping", out_resources.keys_ping_mapped_ptr)) {
        return false;
    }
    if (!map_buffer_memory(context, out_resources.keys_pong, "radix sort keys_pong", out_resources.keys_pong_mapped_ptr)) {
        return false;
    }
    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.keys_pong_mapped_ptr != nullptr && resources.keys_pong.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.keys_pong.memory);
        resources.keys_pong_mapped_ptr = nullptr;
    }
    if (resources.keys_ping_mapped_ptr != nullptr && resources.keys_ping.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.keys_ping.memory);
        resources.keys_ping_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.digit_starts);
    destroy_buffer_resource(context.device(), resources.block_prefix);
    destroy_buffer_resource(context.device(), resources.histogram);
    destroy_buffer_resource(context.device(), resources.keys_pong);
    destroy_buffer_resource(context.device(), resources.keys_ping);
}

void update_count_descriptor_set(VulkanContext& context, VkBuffer keys_src, VkDeviceSize keys_size,
                                 const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo keys_info{keys_src, 0U, keys_size};
    const VkDescriptorBufferInfo hist_info{buffers.histogram.buffer, 0U, buffers.histogram.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, keys_info}, {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hist_info}});
}

void update_scan_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo hist_info{buffers.histogram.buffer, 0U, buffers.histogram.size};
    const VkDescriptorBufferInfo prefix_info{buffers.block_prefix.buffer, 0U, buffers.block_prefix.size};
    const VkDescriptorBufferInfo starts_info{buffers.digit_starts.buffer, 0U, buffers.digit_starts.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, hist_info},
         {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, prefix_info},
         {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, starts_info}});
}

void update_scatter_descriptor_set(VulkanContext& context, VkBuffer keys_src, VkDeviceSize keys_src_size,
                                   const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo src_info{keys_src, 0U, keys_src_size};
    const bool src_is_ping = (keys_src == buffers.keys_ping.buffer);
    const VkDescriptorBufferInfo dst_info{
        src_is_ping ? buffers.keys_pong.buffer : buffers.keys_ping.buffer, 0U,
        src_is_ping ? buffers.keys_pong.size : buffers.keys_ping.size};
    const VkDescriptorBufferInfo prefix_info{buffers.block_prefix.buffer, 0U, buffers.block_prefix.size};
    const VkDescriptorBufferInfo starts_info{buffers.digit_starts.buffer, 0U, buffers.digit_starts.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, src_info},
         {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dst_info},
         {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, prefix_info},
         {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, starts_info}});
}

bool create_pipeline_resources(VulkanContext& context, const std::string& count_shader_path,
                               const std::string& scan_shader_path, const std::string& scatter_shader_path,
                               const BufferResources& buffers, PipelineResources& out_resources) {
    // -- count pipeline --
    {
        const std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                              out_resources.count_descriptor_set_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create count descriptor set layout.\n";
            return false;
        }
        // Two descriptor sets: one for reading keys_ping, one for reading keys_pong.
        const std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4U}};
        if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 2U,
                                                        out_resources.count_descriptor_pool)) {
            std::cerr << "[" << kExperimentId << "] Failed to create count descriptor pool.\n";
            return false;
        }
        if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.count_descriptor_pool,
                                                         out_resources.count_descriptor_set_layout,
                                                         out_resources.count_descriptor_set_ping)) {
            std::cerr << "[" << kExperimentId << "] Failed to allocate count ping descriptor set.\n";
            return false;
        }
        if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.count_descriptor_pool,
                                                         out_resources.count_descriptor_set_layout,
                                                         out_resources.count_descriptor_set_pong)) {
            std::cerr << "[" << kExperimentId << "] Failed to allocate count pong descriptor set.\n";
            return false;
        }
        update_count_descriptor_set(context, buffers.keys_ping.buffer, buffers.keys_ping.size, buffers,
                                    out_resources.count_descriptor_set_ping);
        update_count_descriptor_set(context, buffers.keys_pong.buffer, buffers.keys_pong.size, buffers,
                                    out_resources.count_descriptor_set_pong);

        const std::vector<VkPushConstantRange> pc_ranges = {
            {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(CountPushConstants))}};
        if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.count_descriptor_set_layout},
                                                        pc_ranges, out_resources.count_pipeline_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create count pipeline layout.\n";
            return false;
        }
        if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), count_shader_path,
                                                              out_resources.count_shader_module)) {
            std::cerr << "[" << kExperimentId << "] Failed to load count shader: " << count_shader_path << "\n";
            return false;
        }
        if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.count_shader_module,
                                                         out_resources.count_pipeline_layout, "main",
                                                         out_resources.count_pipeline)) {
            std::cerr << "[" << kExperimentId << "] Failed to create count compute pipeline.\n";
            return false;
        }
    }

    // -- scan pipeline --
    {
        const std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                              out_resources.scan_descriptor_set_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scan descriptor set layout.\n";
            return false;
        }
        const std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U}};
        if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U,
                                                        out_resources.scan_descriptor_pool)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scan descriptor pool.\n";
            return false;
        }
        if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.scan_descriptor_pool,
                                                         out_resources.scan_descriptor_set_layout,
                                                         out_resources.scan_descriptor_set)) {
            std::cerr << "[" << kExperimentId << "] Failed to allocate scan descriptor set.\n";
            return false;
        }
        update_scan_descriptor_set(context, buffers, out_resources.scan_descriptor_set);

        const std::vector<VkPushConstantRange> pc_ranges = {
            {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(ScanPushConstants))}};
        if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.scan_descriptor_set_layout},
                                                        pc_ranges, out_resources.scan_pipeline_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scan pipeline layout.\n";
            return false;
        }
        if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), scan_shader_path,
                                                              out_resources.scan_shader_module)) {
            std::cerr << "[" << kExperimentId << "] Failed to load scan shader: " << scan_shader_path << "\n";
            return false;
        }
        if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.scan_shader_module,
                                                         out_resources.scan_pipeline_layout, "main",
                                                         out_resources.scan_pipeline)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scan compute pipeline.\n";
            return false;
        }
    }

    // -- scatter pipeline --
    {
        const std::vector<VkDescriptorSetLayoutBinding> bindings = {
            {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                              out_resources.scatter_descriptor_set_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scatter descriptor set layout.\n";
            return false;
        }
        // Two descriptor sets: one for ping→pong, one for pong→ping.
        const std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8U}};
        if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 2U,
                                                        out_resources.scatter_descriptor_pool)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scatter descriptor pool.\n";
            return false;
        }
        if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.scatter_descriptor_pool,
                                                         out_resources.scatter_descriptor_set_layout,
                                                         out_resources.scatter_descriptor_set_ping)) {
            std::cerr << "[" << kExperimentId << "] Failed to allocate scatter ping descriptor set.\n";
            return false;
        }
        if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.scatter_descriptor_pool,
                                                         out_resources.scatter_descriptor_set_layout,
                                                         out_resources.scatter_descriptor_set_pong)) {
            std::cerr << "[" << kExperimentId << "] Failed to allocate scatter pong descriptor set.\n";
            return false;
        }
        update_scatter_descriptor_set(context, buffers.keys_ping.buffer, buffers.keys_ping.size, buffers,
                                      out_resources.scatter_descriptor_set_ping);
        update_scatter_descriptor_set(context, buffers.keys_pong.buffer, buffers.keys_pong.size, buffers,
                                      out_resources.scatter_descriptor_set_pong);

        const std::vector<VkPushConstantRange> pc_ranges = {
            {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(ScatterPushConstants))}};
        if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.scatter_descriptor_set_layout},
                                                        pc_ranges, out_resources.scatter_pipeline_layout)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scatter pipeline layout.\n";
            return false;
        }
        if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), scatter_shader_path,
                                                              out_resources.scatter_shader_module)) {
            std::cerr << "[" << kExperimentId << "] Failed to load scatter shader: " << scatter_shader_path << "\n";
            return false;
        }
        if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.scatter_shader_module,
                                                         out_resources.scatter_pipeline_layout, "main",
                                                         out_resources.scatter_pipeline)) {
            std::cerr << "[" << kExperimentId << "] Failed to create scatter compute pipeline.\n";
            return false;
        }
    }
    return true;
}

void destroy_pipeline_resources(VulkanContext& context, PipelineResources& r) {
    auto destroy_pipeline = [&](VkPipeline& p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device(), p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    auto destroy_layout = [&](VkPipelineLayout& l) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context.device(), l, nullptr);
            l = VK_NULL_HANDLE;
        }
    };
    auto destroy_pool = [&](VkDescriptorPool& p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(context.device(), p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    auto destroy_dsl = [&](VkDescriptorSetLayout& dsl) {
        if (dsl != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context.device(), dsl, nullptr);
            dsl = VK_NULL_HANDLE;
        }
    };
    auto destroy_shader = [&](VkShaderModule& sm) {
        if (sm != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device(), sm, nullptr);
            sm = VK_NULL_HANDLE;
        }
    };

    destroy_pipeline(r.scatter_pipeline);
    destroy_layout(r.scatter_pipeline_layout);
    destroy_pool(r.scatter_descriptor_pool);
    destroy_dsl(r.scatter_descriptor_set_layout);
    destroy_shader(r.scatter_shader_module);
    r.scatter_descriptor_set_ping = VK_NULL_HANDLE;
    r.scatter_descriptor_set_pong = VK_NULL_HANDLE;

    destroy_pipeline(r.scan_pipeline);
    destroy_layout(r.scan_pipeline_layout);
    destroy_pool(r.scan_descriptor_pool);
    destroy_dsl(r.scan_descriptor_set_layout);
    destroy_shader(r.scan_shader_module);
    r.scan_descriptor_set = VK_NULL_HANDLE;

    destroy_pipeline(r.count_pipeline);
    destroy_layout(r.count_pipeline_layout);
    destroy_pool(r.count_descriptor_pool);
    destroy_dsl(r.count_descriptor_set_layout);
    destroy_shader(r.count_shader_module);
    r.count_descriptor_set_ping = VK_NULL_HANDLE;
    r.count_descriptor_set_pong = VK_NULL_HANDLE;
}

void record_compute_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &barrier, 0U, nullptr, 0U, nullptr);
}

// Run one complete radix sort (all passes) and return GPU time in ms.
// ping_is_input: if true the keys_ping buffer contains the current input;
//                after each pass the source alternates (ping-pong).
double run_sort(VulkanContext& context, const PipelineResources& pipelines, const VariantDescriptor& variant,
                uint32_t element_count, uint32_t num_blocks) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer cmd) {
        bool reading_ping = true;

        for (uint32_t pass = 0U; pass < variant.num_passes; ++pass) {
            const uint32_t bit_offset = pass * variant.radix_bits;

            // -- Count pass --
            const CountPushConstants count_pc{element_count, num_blocks, bit_offset, variant.radix_size,
                                              variant.radix_mask};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.count_pipeline);
            VkDescriptorSet count_ds =
                reading_ping ? pipelines.count_descriptor_set_ping : pipelines.count_descriptor_set_pong;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.count_pipeline_layout, 0U, 1U,
                                    &count_ds, 0U, nullptr);
            vkCmdPushConstants(cmd, pipelines.count_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                static_cast<uint32_t>(sizeof(count_pc)), &count_pc);
            vkCmdDispatch(cmd, num_blocks, 1U, 1U);

            record_compute_barrier(cmd);

            // -- Scan pass (single workgroup) --
            const ScanPushConstants scan_pc{num_blocks, variant.radix_size};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.scan_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.scan_pipeline_layout, 0U, 1U,
                                    &pipelines.scan_descriptor_set, 0U, nullptr);
            vkCmdPushConstants(cmd, pipelines.scan_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                static_cast<uint32_t>(sizeof(scan_pc)), &scan_pc);
            vkCmdDispatch(cmd, 1U, 1U, 1U);

            record_compute_barrier(cmd);

            // -- Scatter pass --
            const ScatterPushConstants scatter_pc{element_count, num_blocks, bit_offset, variant.radix_mask};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.scatter_pipeline);
            VkDescriptorSet scatter_ds =
                reading_ping ? pipelines.scatter_descriptor_set_ping : pipelines.scatter_descriptor_set_pong;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.scatter_pipeline_layout, 0U, 1U,
                                    &scatter_ds, 0U, nullptr);
            vkCmdPushConstants(cmd, pipelines.scatter_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                static_cast<uint32_t>(sizeof(scatter_pc)), &scatter_pc);
            vkCmdDispatch(cmd, num_blocks, 1U, 1U);

            if (pass + 1U < variant.num_passes) {
                record_compute_barrier(cmd);
            }

            reading_ping = !reading_ping;
        }
    });
}

std::vector<uint32_t> cpu_reference_sort(const uint32_t* keys, uint32_t count) {
    std::vector<uint32_t> sorted(keys, keys + count);
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

// After a full sort, the output lives in ping if num_passes is even, pong otherwise.
bool validate_sort_result(const BufferResources& buffers, uint32_t element_count, uint32_t num_passes,
                          const std::vector<uint32_t>& reference_sorted) {
    const uint32_t* result_ptr = (num_passes % 2U == 0U)
                                     ? static_cast<const uint32_t*>(buffers.keys_ping_mapped_ptr)
                                     : static_cast<const uint32_t*>(buffers.keys_pong_mapped_ptr);
    if (result_ptr == nullptr) {
        return false;
    }
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (result_ptr[index] != reference_sorted[index]) {
            return false;
        }
    }
    return true;
}

void fill_keys(uint32_t* keys, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        keys[index] = input_pattern_value(index);
    }
}

void record_case_notes(std::string& notes, const VariantDescriptor& variant, uint32_t element_count,
                       uint32_t num_blocks, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("variant=") + variant.variant_name);
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "num_blocks=" + std::to_string(num_blocks));
    append_note(notes, "radix_bits=" + std::to_string(variant.radix_bits));
    append_note(notes, "num_passes=" + std::to_string(variant.num_passes));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipelines, const VariantDescriptor& variant, uint32_t element_count,
              RadixSortGpuExperimentOutput& output, bool verbose_progress) {
    const uint32_t num_blocks = element_count / kWorkgroupSize;
    auto* keys_ptr = static_cast<uint32_t*>(buffers.keys_ping_mapped_ptr);
    if (keys_ptr == nullptr) {
        std::cerr << "[" << kExperimentId << "] keys_ping mapped pointer is null.\n";
        return false;
    }

    fill_keys(keys_ptr, element_count);
    const std::vector<uint32_t> reference_sorted = cpu_reference_sort(keys_ptr, element_count);

    const uint64_t bytes_per_sort =
        static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 2ULL * static_cast<uint64_t>(variant.num_passes);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_keys(keys_ptr, element_count);
        const double ms = run_sort(context, pipelines, variant, element_count, num_blocks);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(ms);
            const bool correct =
                dispatch_ok && validate_sort_result(buffers, element_count, variant.num_passes, reference_sorted);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant.variant_name << " n=" << element_count << " ms=" << ms
                      << " correct=" << (correct ? "pass" : "fail") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        fill_keys(keys_ptr, element_count);
        const auto start = std::chrono::high_resolution_clock::now();
        const double dispatch_ms = run_sort(context, pipelines, variant, element_count, num_blocks);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> e2e_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && validate_sort_result(buffers, element_count, variant.num_passes, reference_sorted);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, variant, element_count, num_blocks, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant.variant_name,
            .problem_size = element_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = e2e_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(bytes_per_sort, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + variant.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

std::vector<uint32_t> build_problem_sizes(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    // `max_buffer_bytes` is a per-buffer budget. Validate each candidate against
    // the largest individual scratch buffers required by the radix sort:
    // - ping/pong key buffers: element_count * sizeof(uint32_t)
    // - histogram / block_prefix buffers: radix_size * num_blocks * sizeof(uint32_t)
    // The fixed-size digit_starts buffer is smaller than the histogram buffers for
    // the same radix size, so it is covered by the histogram check.
    constexpr uint64_t kMaxRadixSize = 256ULL;
    const uint64_t max_buffer_bytes_u64 = static_cast<uint64_t>(max_buffer_bytes);
    const uint64_t max_elements_from_dispatch =
        static_cast<uint64_t>(max_dispatch_groups_x) * static_cast<uint64_t>(kWorkgroupSize);
    const uint64_t max_elements_from_blocks = static_cast<uint64_t>(kMaxBlocks) * kWorkgroupSize;
    const uint64_t max_elements =
        std::min({max_elements_from_dispatch, max_elements_from_blocks, static_cast<uint64_t>(UINT32_MAX)});

    std::vector<uint32_t> sizes;
    for (const uint32_t candidate : kCandidateProblemSizes) {
        if (candidate > max_elements || (candidate % kWorkgroupSize != 0U)) {
            continue;
        }

        const uint64_t key_buffer_bytes = static_cast<uint64_t>(candidate) * sizeof(uint32_t);
        const uint64_t num_blocks = static_cast<uint64_t>(candidate) / static_cast<uint64_t>(kWorkgroupSize);
        const uint64_t histogram_buffer_bytes = kMaxRadixSize * num_blocks * sizeof(uint32_t);

        if (key_buffer_bytes <= max_buffer_bytes_u64 && histogram_buffer_bytes <= max_buffer_bytes_u64) {
            sizes.push_back(candidate);
        }
    }
    return sizes;
}

} // namespace

RadixSortGpuExperimentOutput run_radix_sort_gpu_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                           const RadixSortGpuExperimentConfig& config) {
    RadixSortGpuExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "[" << kExperimentId << "] GPU timestamp support required.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string count_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.count_shader_path, "34_radix_sort_count.comp.spv");
    const std::string scan_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.scan_shader_path, "34_radix_sort_scan.comp.spv");
    const std::string scatter_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.scatter_shader_path, "34_radix_sort_scatter.comp.spv");

    if (count_shader_path.empty() || scan_shader_path.empty() || scatter_shader_path.empty()) {
        std::cerr << "[" << kExperimentId << "] Could not locate one or more SPIR-V shaders.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_props{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_props);

    const std::vector<uint32_t> problem_sizes =
        build_problem_sizes(config.max_buffer_bytes, device_props.limits.maxComputeWorkGroupCount[0]);
    if (problem_sizes.empty()) {
        std::cerr << "[" << kExperimentId << "] Scratch budget too small for radix sort experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t max_problem_size = *std::max_element(problem_sizes.begin(), problem_sizes.end());
    const uint32_t max_num_blocks = max_problem_size / kWorkgroupSize;

    // Allocate buffers for the largest 8-bit radix variant (radix_size=256).
    BufferResources buffers{};
    if (!create_buffer_resources(context, max_problem_size, 256U, max_num_blocks, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    PipelineResources pipelines{};
    if (!create_pipeline_resources(context, count_shader_path, scan_shader_path, scatter_shader_path, buffers,
                                   pipelines)) {
        destroy_pipeline_resources(context, pipelines);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const uint32_t problem_size : problem_sizes) {
        for (const VariantDescriptor& variant : kVariantDescriptors) {
            if (!run_case(context, runner, buffers, pipelines, variant, problem_size, output,
                          config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipelines);
    destroy_buffer_resources(context, buffers);
    return output;
}
