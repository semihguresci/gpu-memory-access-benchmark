#include "experiments/stream_compaction_experiment.hpp"

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
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "24_stream_compaction";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCountAtomic = 1U;
constexpr uint32_t kDispatchCountThreeStage = 3U;
constexpr uint32_t kTargetLogicalCount = kWorkgroupSize * kWorkgroupSize;
constexpr uint32_t kSentinelValue = 0xDEADBEEFU;
constexpr uint32_t kCounterSentinelValue = 0U;
constexpr uint32_t kInputXorSeed = 0xA511E9B3U;
constexpr uint32_t kInputMultiplier = 0x9E3779B9U;
constexpr uint32_t kInputAddend = 0x7F4A7C15U;
constexpr uint32_t kHashMul0 = 0x7FEB352DU;
constexpr uint32_t kHashMul1 = 0x846CA68BU;
constexpr uint32_t kRatioDivisor = 100U;
constexpr uint32_t kCounterElements = 1U;

constexpr std::array<uint32_t, 5> kValidRatioPercents = {5U, 25U, 50U, 75U, 95U};

enum class ImplementationKind : std::uint8_t {
    GlobalAtomicAppend,
    ThreeStage,
};

struct ImplementationDescriptor {
    ImplementationKind kind;
    const char* implementation_name;
};

constexpr std::array<ImplementationDescriptor, 2> kImplementationDescriptors = {{
    {ImplementationKind::GlobalAtomicAppend, "global_atomic_append"},
    {ImplementationKind::ThreeStage, "three_stage"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource output_buffer{};
    BufferResource local_prefix_buffer{};
    BufferResource block_counts_buffer{};
    BufferResource block_prefix_buffer{};
    BufferResource counter_buffer{};
    void* input_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
    void* local_prefix_mapped_ptr = nullptr;
    void* block_counts_mapped_ptr = nullptr;
    void* block_prefix_mapped_ptr = nullptr;
    void* counter_mapped_ptr = nullptr;
};

struct DescriptorResources {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct PipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t element_count = 0U;
    uint32_t block_count = 0U;
    uint32_t valid_ratio_percent = 0U;
    uint32_t pattern_seed = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct ReferenceData {
    std::vector<uint32_t> input_values;
    std::vector<uint32_t> local_prefix_values;
    std::vector<uint32_t> block_counts;
    std::vector<uint32_t> block_prefix_values;
    std::vector<uint32_t> compacted_values;
    uint32_t valid_count = 0U;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

const char* implementation_name(ImplementationKind implementation_kind) {
    for (const auto& descriptor : kImplementationDescriptors) {
        if (descriptor.kind == implementation_kind) {
            return descriptor.implementation_name;
        }
    }

    return "unknown";
}

std::string make_variant_name(ImplementationKind implementation_kind, uint32_t valid_ratio_percent) {
    return std::string(implementation_name(implementation_kind)) + "_ratio_" + std::to_string(valid_ratio_percent);
}

std::string make_case_name(ImplementationKind implementation_kind, uint32_t valid_ratio_percent,
                           uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(implementation_kind, valid_ratio_percent) +
           "_elements_" + std::to_string(logical_count);
}

uint32_t hash_index(uint32_t index, uint32_t pattern_seed) {
    uint32_t value = index ^ pattern_seed;
    value ^= value >> 16U;
    value *= kHashMul0;
    value ^= value >> 15U;
    value *= kHashMul1;
    value ^= value >> 16U;
    return value;
}

uint32_t generate_input_value(uint32_t index, uint32_t pattern_seed) {
    return hash_index(index, pattern_seed) ^ ((index * kInputMultiplier) + kInputAddend) ^ kInputXorSeed;
}

bool is_index_valid(uint32_t index, uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    return (hash_index(index, pattern_seed) % kRatioDivisor) < valid_ratio_percent;
}

uint32_t compute_block_count(uint32_t logical_count) {
    return VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
}

VkDeviceSize compute_buffer_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_required_total_bytes(uint32_t logical_count) {
    const uint32_t block_count = compute_block_count(logical_count);
    return (compute_buffer_span_bytes(logical_count) * 3U) + (compute_buffer_span_bytes(block_count) * 2U) +
           compute_buffer_span_bytes(kCounterElements);
}

uint32_t determine_logical_count(VkDeviceSize total_budget_bytes, uint32_t max_dispatch_groups_x) {
    const uint32_t max_group_count =
        std::min(kWorkgroupSize, std::min(max_dispatch_groups_x, kTargetLogicalCount / kWorkgroupSize));
    for (uint32_t group_count = max_group_count; group_count > 0U; --group_count) {
        const uint32_t logical_count = group_count * kWorkgroupSize;
        if (compute_required_total_bytes(logical_count) <= total_budget_bytes) {
            return logical_count;
        }
    }

    return 0U;
}

uint64_t compute_estimated_total_bytes(ImplementationKind implementation_kind, uint32_t logical_count,
                                       uint32_t block_count, uint32_t valid_count) {
    const uint64_t n = logical_count;
    const uint64_t b = block_count;
    const uint64_t v = valid_count;
    if (implementation_kind == ImplementationKind::GlobalAtomicAppend) {
        return (n * sizeof(uint32_t)) + (v * sizeof(uint32_t)) + (v * sizeof(uint32_t) * 2U);
    }

    return ((3U * n) + v + (3U * b) + 1U) * sizeof(uint32_t);
}

double compute_effective_gbps(ImplementationKind implementation_kind, uint32_t logical_count, uint32_t block_count,
                              uint32_t valid_count, double dispatch_ms) {
    return compute_effective_gbps_from_bytes(
        compute_estimated_total_bytes(implementation_kind, logical_count, block_count, valid_count), dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, uint32_t logical_count, uint32_t block_count,
                             BufferResources& out_resources) {
    const VkDeviceSize logical_span_bytes = compute_buffer_span_bytes(logical_count);
    const VkDeviceSize block_span_bytes = compute_buffer_span_bytes(block_count);
    const VkDeviceSize counter_span_bytes = compute_buffer_span_bytes(kCounterElements);

    if (!create_buffer_resource(
            context.physical_device(), context.device(), logical_span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create stream compaction input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), logical_span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create stream compaction output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), logical_span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.local_prefix_buffer)) {
        std::cerr << "Failed to create stream compaction local-prefix buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), block_span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_counts_buffer)) {
        std::cerr << "Failed to create stream compaction block-count buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.local_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), block_span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_prefix_buffer)) {
        std::cerr << "Failed to create stream compaction block-prefix buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.block_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.local_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), counter_span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.counter_buffer)) {
        std::cerr << "Failed to create stream compaction counter buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.block_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.local_prefix_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "stream compaction input buffer",
                           out_resources.input_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_buffer, "stream compaction output buffer",
                           out_resources.output_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.local_prefix_buffer, "stream compaction local-prefix buffer",
                           out_resources.local_prefix_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.block_counts_buffer, "stream compaction block-count buffer",
                           out_resources.block_counts_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.block_prefix_buffer, "stream compaction block-prefix buffer",
                           out_resources.block_prefix_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.counter_buffer, "stream compaction counter buffer",
                           out_resources.counter_mapped_ptr)) {
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.counter_mapped_ptr != nullptr && resources.counter_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.counter_buffer.memory);
        resources.counter_mapped_ptr = nullptr;
    }
    if (resources.block_prefix_mapped_ptr != nullptr && resources.block_prefix_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.block_prefix_buffer.memory);
        resources.block_prefix_mapped_ptr = nullptr;
    }
    if (resources.block_counts_mapped_ptr != nullptr && resources.block_counts_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.block_counts_buffer.memory);
        resources.block_counts_mapped_ptr = nullptr;
    }
    if (resources.local_prefix_mapped_ptr != nullptr && resources.local_prefix_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.local_prefix_buffer.memory);
        resources.local_prefix_mapped_ptr = nullptr;
    }
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.counter_buffer);
    destroy_buffer_resource(context.device(), resources.block_prefix_buffer);
    destroy_buffer_resource(context.device(), resources.block_counts_buffer);
    destroy_buffer_resource(context.device(), resources.local_prefix_buffer);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    const VkDescriptorBufferInfo local_prefix_info{buffers.local_prefix_buffer.buffer, 0U,
                                                   buffers.local_prefix_buffer.size};
    const VkDescriptorBufferInfo block_counts_info{buffers.block_counts_buffer.buffer, 0U,
                                                   buffers.block_counts_buffer.size};
    const VkDescriptorBufferInfo block_prefix_info{buffers.block_prefix_buffer.buffer, 0U,
                                                   buffers.block_prefix_buffer.size};
    const VkDescriptorBufferInfo counter_info{buffers.counter_buffer.buffer, 0U, buffers.counter_buffer.size};

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                          {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, local_prefix_info},
                                                          {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, block_counts_info},
                                                          {4U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, block_prefix_info},
                                                          {5U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, counter_info},
                                                      });
}

bool create_descriptor_resources(VulkanContext& context, const BufferResources& buffers,
                                 DescriptorResources& out_resources) {
    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create stream compaction descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create stream compaction descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate stream compaction descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);
    return true;
}

void destroy_descriptor_resources(VulkanContext& context, DescriptorResources& resources) {
    if (resources.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), resources.descriptor_pool, nullptr);
        resources.descriptor_pool = VK_NULL_HANDLE;
    }
    if (resources.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), resources.descriptor_set_layout, nullptr);
        resources.descriptor_set_layout = VK_NULL_HANDLE;
    }
    resources.descriptor_set = VK_NULL_HANDLE;
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                               VkDescriptorSetLayout descriptor_set_layout, PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load stream compaction shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {descriptor_set_layout}, push_constant_ranges,
                                                    out_resources.pipeline_layout)) {
        std::cerr << "Failed to create stream compaction pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create stream compaction compute pipeline.\n";
        return false;
    }

    return true;
}

void destroy_pipeline_resources(VulkanContext& context, PipelineResources& resources) {
    if (resources.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), resources.pipeline, nullptr);
        resources.pipeline = VK_NULL_HANDLE;
    }
    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }
    if (resources.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), resources.shader_module, nullptr);
        resources.shader_module = VK_NULL_HANDLE;
    }
}

void record_compute_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         1U, &barrier, 0U, nullptr, 0U, nullptr);
}

void fill_values(uint32_t* values, uint32_t count, uint32_t value) {
    std::fill_n(values, count, value);
}

ReferenceData build_reference_data(uint32_t logical_count, uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    ReferenceData reference;
    reference.input_values.resize(logical_count, 0U);
    reference.local_prefix_values.resize(logical_count, 0U);
    reference.compacted_values.reserve(logical_count);

    const uint32_t block_count = compute_block_count(logical_count);
    reference.block_counts.resize(block_count, 0U);
    reference.block_prefix_values.resize(block_count, 0U);

    for (uint32_t index = 0U; index < logical_count; ++index) {
        const uint32_t block_index = index / kWorkgroupSize;
        const uint32_t local_prefix = reference.block_counts[block_index];
        const bool valid = is_index_valid(index, valid_ratio_percent, pattern_seed);

        reference.input_values[index] = generate_input_value(index, pattern_seed);
        reference.local_prefix_values[index] = local_prefix;
        if (valid) {
            reference.compacted_values.push_back(reference.input_values[index]);
            ++reference.block_counts[block_index];
        }
    }

    uint32_t running_prefix = 0U;
    for (uint32_t block_index = 0U; block_index < block_count; ++block_index) {
        reference.block_prefix_values[block_index] = running_prefix;
        running_prefix += reference.block_counts[block_index];
    }
    reference.valid_count = static_cast<uint32_t>(reference.compacted_values.size());
    return reference;
}

bool validate_input_values(const uint32_t* actual_values, const std::vector<uint32_t>& expected_values) {
    for (std::size_t index = 0; index < expected_values.size(); ++index) {
        if (actual_values[index] != expected_values[index]) {
            return false;
        }
    }

    return true;
}

bool validate_output_sentinels(const uint32_t* actual_values, uint32_t begin_index, uint32_t logical_count) {
    for (uint32_t index = begin_index; index < logical_count; ++index) {
        if (actual_values[index] != kSentinelValue) {
            return false;
        }
    }

    return true;
}

bool validate_uint_vector_prefix(const uint32_t* actual_values, const std::vector<uint32_t>& expected_values) {
    for (std::size_t index = 0; index < expected_values.size(); ++index) {
        if (actual_values[index] != expected_values[index]) {
            return false;
        }
    }

    return true;
}

bool validate_atomic_output(const uint32_t* output_values, uint32_t logical_count, const ReferenceData& reference,
                            uint32_t observed_count) {
    if (observed_count != reference.valid_count || observed_count > logical_count) {
        return false;
    }

    std::vector<uint32_t> actual_values(output_values, output_values + observed_count);
    std::vector<uint32_t> expected_values = reference.compacted_values;
    std::sort(actual_values.begin(), actual_values.end());
    std::sort(expected_values.begin(), expected_values.end());
    if (actual_values != expected_values) {
        return false;
    }

    return validate_output_sentinels(output_values, observed_count, logical_count);
}

bool validate_three_stage_outputs(const BufferResources& buffers, uint32_t logical_count,
                                  const ReferenceData& reference) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    const auto* output_values = static_cast<const uint32_t*>(buffers.output_mapped_ptr);
    const auto* local_prefix_values = static_cast<const uint32_t*>(buffers.local_prefix_mapped_ptr);
    const auto* block_counts_values = static_cast<const uint32_t*>(buffers.block_counts_mapped_ptr);
    const auto* block_prefix_values = static_cast<const uint32_t*>(buffers.block_prefix_mapped_ptr);
    const auto* counter_values = static_cast<const uint32_t*>(buffers.counter_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || local_prefix_values == nullptr ||
        block_counts_values == nullptr || block_prefix_values == nullptr || counter_values == nullptr) {
        return false;
    }

    if (!validate_input_values(input_values, reference.input_values)) {
        return false;
    }
    if (!validate_uint_vector_prefix(local_prefix_values, reference.local_prefix_values)) {
        return false;
    }
    if (!validate_uint_vector_prefix(block_counts_values, reference.block_counts)) {
        return false;
    }
    if (!validate_uint_vector_prefix(block_prefix_values, reference.block_prefix_values)) {
        return false;
    }
    if (counter_values[0] != reference.valid_count) {
        return false;
    }
    if (!validate_uint_vector_prefix(output_values, reference.compacted_values)) {
        return false;
    }
    return validate_output_sentinels(output_values, reference.valid_count, logical_count);
}

bool validate_atomic_outputs(const BufferResources& buffers, uint32_t logical_count, const ReferenceData& reference) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    const auto* output_values = static_cast<const uint32_t*>(buffers.output_mapped_ptr);
    const auto* counter_values = static_cast<const uint32_t*>(buffers.counter_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || counter_values == nullptr) {
        return false;
    }

    if (!validate_input_values(input_values, reference.input_values)) {
        return false;
    }

    return validate_atomic_output(output_values, logical_count, reference, counter_values[0]);
}

void reset_buffers_for_iteration(const BufferResources& buffers, uint32_t logical_count, uint32_t block_count) {
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* local_prefix_values = static_cast<uint32_t*>(buffers.local_prefix_mapped_ptr);
    auto* block_counts_values = static_cast<uint32_t*>(buffers.block_counts_mapped_ptr);
    auto* block_prefix_values = static_cast<uint32_t*>(buffers.block_prefix_mapped_ptr);
    auto* counter_values = static_cast<uint32_t*>(buffers.counter_mapped_ptr);

    fill_values(output_values, logical_count, kSentinelValue);
    fill_values(local_prefix_values, logical_count, kSentinelValue);
    fill_values(block_counts_values, block_count, kSentinelValue);
    fill_values(block_prefix_values, block_count, kSentinelValue);
    fill_values(counter_values, kCounterElements, kCounterSentinelValue);
}

void write_input_values(const BufferResources& buffers, const ReferenceData& reference) {
    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    std::copy(reference.input_values.begin(), reference.input_values.end(), input_values);
}

double run_atomic_pipeline(VulkanContext& context, const DescriptorResources& descriptors,
                           const PipelineResources& atomic_pipeline, uint32_t logical_count,
                           uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    const uint32_t group_count_x = compute_block_count(logical_count);
    const PushConstants push_constants{
        logical_count,
        group_count_x,
        valid_ratio_percent,
        pattern_seed,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, atomic_pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, atomic_pipeline.pipeline_layout, 0U, 1U,
                                &descriptors.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, atomic_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

double run_three_stage_pipeline(VulkanContext& context, const DescriptorResources& descriptors,
                                const PipelineResources& stage1_pipeline, const PipelineResources& stage2_pipeline,
                                const PipelineResources& stage3_pipeline, uint32_t logical_count,
                                uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    const uint32_t block_count = compute_block_count(logical_count);
    const PushConstants push_constants{
        logical_count,
        block_count,
        valid_ratio_percent,
        pattern_seed,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage1_pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage1_pipeline.pipeline_layout, 0U, 1U,
                                &descriptors.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, stage1_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, block_count, 1U, 1U);

        record_compute_barrier(command_buffer);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage2_pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage2_pipeline.pipeline_layout, 0U, 1U,
                                &descriptors.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, stage2_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, 1U, 1U, 1U);

        record_compute_barrier(command_buffer);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage3_pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, stage3_pipeline.pipeline_layout, 0U, 1U,
                                &descriptors.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, stage3_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, block_count, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, ImplementationKind implementation_kind,
                       const StreamCompactionExperimentConfig& config, uint32_t logical_count, uint32_t block_count,
                       uint32_t valid_ratio_percent, const ReferenceData& reference, bool correctness_pass,
                       bool dispatch_ok) {
    append_note(notes, std::string("implementation=") + implementation_name(implementation_kind));
    append_note(notes, "valid_ratio_percent=" + std::to_string(valid_ratio_percent));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "valid_count=" + std::to_string(reference.valid_count));
    append_note(notes, "block_count=" + std::to_string(block_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "dispatch_count=" + std::to_string(implementation_kind == ImplementationKind::ThreeStage
                                                              ? kDispatchCountThreeStage
                                                              : kDispatchCountAtomic));
    append_note(notes, "stable_ordering=" +
                           std::string(implementation_kind == ImplementationKind::ThreeStage ? "true" : "false"));
    append_note(notes, "pattern_seed=" + std::to_string(config.pattern_seed));
    append_note(notes, "input_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_buffer_span_bytes(logical_count))));
    append_note(notes, "output_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_buffer_span_bytes(logical_count))));
    append_note(notes, "local_prefix_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_buffer_span_bytes(logical_count))));
    append_note(notes, "block_counts_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_buffer_span_bytes(block_count))));
    append_note(notes, "block_prefix_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_buffer_span_bytes(block_count))));
    append_note(notes, "counter_span_bytes=" + std::to_string(static_cast<unsigned long long>(
                                                   compute_buffer_span_bytes(kCounterElements))));
    append_note(notes,
                "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(config.max_buffer_bytes)));
    append_note(notes, "estimated_total_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_estimated_total_bytes(
                               implementation_kind, logical_count, block_count, reference.valid_count))));
    append_note(notes, "throughput_basis=input_elements_per_second");
    append_note(notes, "validation_mode=" + std::string(implementation_kind == ImplementationKind::ThreeStage
                                                            ? "exact_stable_compaction"
                                                            : "exact_unordered_compaction"));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const DescriptorResources& descriptors, const PipelineResources& atomic_pipeline,
              const PipelineResources& stage1_pipeline, const PipelineResources& stage2_pipeline,
              const PipelineResources& stage3_pipeline, ImplementationKind implementation_kind,
              uint32_t valid_ratio_percent, const StreamCompactionExperimentConfig& config, uint32_t logical_count,
              StreamCompactionExperimentOutput& output) {
    const uint32_t block_count = compute_block_count(logical_count);
    const std::string variant_name = make_variant_name(implementation_kind, valid_ratio_percent);
    const ReferenceData reference = build_reference_data(logical_count, valid_ratio_percent, config.pattern_seed);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", valid_count=" << reference.valid_count
                  << ", block_count=" << block_count << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        write_input_values(buffers, reference);
        reset_buffers_for_iteration(buffers, logical_count, block_count);

        const double dispatch_ms =
            implementation_kind == ImplementationKind::ThreeStage
                ? run_three_stage_pipeline(context, descriptors, stage1_pipeline, stage2_pipeline, stage3_pipeline,
                                           logical_count, valid_ratio_percent, config.pattern_seed)
                : run_atomic_pipeline(context, descriptors, atomic_pipeline, logical_count, valid_ratio_percent,
                                      config.pattern_seed);

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_ok =
            dispatch_ok && (implementation_kind == ImplementationKind::ThreeStage
                                ? validate_three_stage_outputs(buffers, logical_count, reference)
                                : validate_atomic_outputs(buffers, logical_count, reference));

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << (correctness_ok ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !correctness_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_elements=" << logical_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", correctness_ok=" << (correctness_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        write_input_values(buffers, reference);
        reset_buffers_for_iteration(buffers, logical_count, block_count);

        const double dispatch_ms =
            implementation_kind == ImplementationKind::ThreeStage
                ? run_three_stage_pipeline(context, descriptors, stage1_pipeline, stage2_pipeline, stage3_pipeline,
                                           logical_count, valid_ratio_percent, config.pattern_seed)
                : run_atomic_pipeline(context, descriptors, atomic_pipeline, logical_count, valid_ratio_percent,
                                      config.pattern_seed);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && (implementation_kind == ImplementationKind::ThreeStage
                                ? validate_three_stage_outputs(buffers, logical_count, reference)
                                : validate_atomic_outputs(buffers, logical_count, reference));

        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, implementation_kind, config, logical_count, block_count, valid_ratio_percent,
                          reference, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = logical_count,
            .dispatch_count =
                implementation_kind == ImplementationKind::ThreeStage ? kDispatchCountThreeStage : kDispatchCountAtomic,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, 1U, dispatch_ms),
            .gbps = compute_effective_gbps(implementation_kind, logical_count, block_count, reference.valid_count,
                                           dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        make_case_name(implementation_kind, valid_ratio_percent, logical_count), dispatch_samples));
    return true;
}

} // namespace

StreamCompactionExperimentOutput run_stream_compaction_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                                  const StreamCompactionExperimentConfig& config) {
    StreamCompactionExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "stream compaction experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string atomic_append_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.atomic_append_shader_path, "24_stream_compaction_atomic_append.comp.spv");
    const std::string stage1_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.stage1_shader_path, "24_stream_compaction_stage1_flag_scan.comp.spv");
    const std::string stage2_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.stage2_shader_path, "24_stream_compaction_stage2_block_scan.comp.spv");
    const std::string stage3_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.stage3_shader_path, "24_stream_compaction_stage3_scatter.comp.spv");
    if (atomic_append_shader_path.empty() || stage1_shader_path.empty() || stage2_shader_path.empty() ||
        stage3_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shaders for stream compaction experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count < kWorkgroupSize) {
        std::cerr << "Scratch buffer too small for stream compaction experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t block_count = compute_block_count(logical_count);
    if (block_count > kWorkgroupSize) {
        std::cerr << "Stream compaction experiment requires block_count <= " << kWorkgroupSize
                  << " for the single-workgroup block scan stage.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] atomic shader: " << atomic_append_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] stage1 shader: " << stage1_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] stage2 shader: " << stage2_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] stage3 shader: " << stage3_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_elements=" << logical_count << ", block_count=" << block_count
                  << ", required_total_bytes=" << compute_required_total_bytes(logical_count)
                  << ", scratch_size_bytes=" << config.max_buffer_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, logical_count, block_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    DescriptorResources descriptors{};
    if (!create_descriptor_resources(context, buffers, descriptors)) {
        destroy_descriptor_resources(context, descriptors);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    PipelineResources atomic_pipeline{};
    PipelineResources stage1_pipeline{};
    PipelineResources stage2_pipeline{};
    PipelineResources stage3_pipeline{};
    if (!create_pipeline_resources(context, atomic_append_shader_path, descriptors.descriptor_set_layout,
                                   atomic_pipeline) ||
        !create_pipeline_resources(context, stage1_shader_path, descriptors.descriptor_set_layout, stage1_pipeline) ||
        !create_pipeline_resources(context, stage2_shader_path, descriptors.descriptor_set_layout, stage2_pipeline) ||
        !create_pipeline_resources(context, stage3_shader_path, descriptors.descriptor_set_layout, stage3_pipeline)) {
        destroy_pipeline_resources(context, stage3_pipeline);
        destroy_pipeline_resources(context, stage2_pipeline);
        destroy_pipeline_resources(context, stage1_pipeline);
        destroy_pipeline_resources(context, atomic_pipeline);
        destroy_descriptor_resources(context, descriptors);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const uint32_t valid_ratio_percent : kValidRatioPercents) {
        for (const auto& descriptor : kImplementationDescriptors) {
            if (!run_case(context, runner, buffers, descriptors, atomic_pipeline, stage1_pipeline, stage2_pipeline,
                          stage3_pipeline, descriptor.kind, valid_ratio_percent, config, logical_count, output)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, stage3_pipeline);
    destroy_pipeline_resources(context, stage2_pipeline);
    destroy_pipeline_resources(context, stage1_pipeline);
    destroy_pipeline_resources(context, atomic_pipeline);
    destroy_descriptor_resources(context, descriptors);
    destroy_buffer_resources(context, buffers);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
