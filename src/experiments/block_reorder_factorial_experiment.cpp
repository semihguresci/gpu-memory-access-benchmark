#include "experiments/block_reorder_factorial_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "47_block_reorder_factorial";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kWithinOrdered = 0U;
constexpr uint32_t kWithinReversed = 1U;
constexpr uint32_t kBlocksSequential = 0U;
constexpr uint32_t kBlocksShuffled = 1U;
constexpr uint32_t kSourcePatternSeed = 0x9E3779B9U;
constexpr uint32_t kBlockPermutationSeed = 0x47E1D5C0U;
constexpr uint32_t kSweepChunkElements = 8U;
// The eviction sweep streams a dedicated buffer that exceeds the sum of all
// cache levels on the reference device (TU104: ~2.5 MiB L2 plus per-SM L1).
// Reserve this buffer from the total scratch budget before sizing the five
// workload/staging buffers. It remains live for both residency modes.
constexpr VkDeviceSize kEvictionBufferBytes = 64ULL * 1024ULL * 1024ULL;

constexpr const char* kResidencyWarm = "warm";
constexpr const char* kResidencyEvicted = "evicted_buffer_sweep";

// Every block crosses subgroup boundaries, so reversal changes the address
// footprint of each subgroup rather than only permuting its request lanes.
constexpr std::array<uint32_t, 5> kBlockSubgroupCounts = {2U, 4U, 8U, 16U, 32U};

std::array<uint32_t, kBlockSubgroupCounts.size()> build_block_widths(uint32_t subgroup_size) {
    std::array<uint32_t, kBlockSubgroupCounts.size()> widths{};
    std::ranges::transform(kBlockSubgroupCounts, widths.begin(), [subgroup_size](uint32_t count) {
        return subgroup_size * count;
    });
    return widths;
}

// Bit-reversal within the low `bits` bits of `lane` reproduces the GLSL
// expression bitfieldReverse(lane) >> (32u - bits) for lane < 2^bits.
uint32_t reversed_lane(uint32_t lane, uint32_t bits) {
    uint32_t reversed = 0U;
    for (uint32_t bit = 0U; bit < bits; ++bit) {
        reversed = (reversed << 1U) | (lane & 1U);
        lane >>= 1U;
    }
    return reversed;
}

// log2 of a power-of-two block width, mirroring block_bit_count() in the shader.
uint32_t block_bit_count(uint32_t block_width) {
    uint32_t bits = 0U;
    while ((1ULL << bits) < block_width) {
        ++bits;
    }
    return bits;
}

uint32_t hash32(uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

uint32_t source_pattern_value(uint32_t index) {
    return hash32((index * 747796405U) ^ kSourcePatternSeed);
}

uint32_t transform_value(uint32_t input_value, uint32_t source_index) {
    // Must stay in lockstep with transform_value() in
    // 47_block_reorder_factorial.comp.
    return (input_value ^ (source_index * 1664525U)) + 1013904223U;
}

uint64_t xorshift64_state(uint64_t state) {
    state ^= state << 12U;
    state ^= state >> 25U;
    state ^= state << 27U;
    return state;
}

uint32_t next_prng_value(uint64_t& state) {
    state = xorshift64_state(state);
    return static_cast<uint32_t>((state * 0x2545F4914F6CDD1DULL) >> 32U);
}

void fill_random_permutation(uint32_t* permutation, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        permutation[index] = index;
    }

    uint64_t prng_state = kBlockPermutationSeed;
    for (uint32_t index = element_count; index > 1U; --index) {
        const uint32_t swap_target = next_prng_value(prng_state) % index;
        std::swap(permutation[index - 1U], permutation[swap_target]);
    }
}

struct AccessVariantDescriptor {
    std::string variant_name;
    uint32_t block_width = 0U;
    uint32_t subgroup_size = 0U;
    uint32_t within_mode = kWithinOrdered;
    uint32_t block_mode = kBlocksSequential;
};

struct BufferResources {
    BufferResource source_buffer{};
    BufferResource output_buffer{};
    BufferResource block_perm_buffer{};
    BufferResource eviction_buffer{};
    BufferResource staging_in{};
    BufferResource staging_out{};
    void* staging_in_mapped_ptr = nullptr;
    void* staging_out_mapped_ptr = nullptr;
};

struct MainPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct SweepPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t element_count = 0U;
    uint32_t block_width = 0U;
    uint32_t within_mode = kWithinOrdered;
    uint32_t block_bits = 0U;
};

struct SweepPushConstants {
    uint32_t eviction_element_count = 0U;
    uint32_t output_element_count = 0U;
    uint32_t unused_0 = 0U;
    uint32_t unused_1 = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));
static_assert(sizeof(SweepPushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint64_t compute_output_checksum(const uint32_t* output_values, uint32_t element_count) {
    constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = kOffsetBasis;
    for (uint32_t index = 0U; index < element_count; ++index) {
        hash ^= static_cast<uint64_t>(output_values[index]);
        hash *= kPrime;
    }
    return hash;
}

bool validate_outputs(const uint32_t* output_values, const std::vector<uint32_t>& reference, uint32_t element_count) {
    if (output_values == nullptr || reference.size() < element_count) {
        return false;
    }
    return std::equal(reference.begin(), reference.begin() + element_count, output_values);
}

uint32_t determine_element_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x,
                                 uint32_t element_multiple) {
    // After reserving the eviction buffer, five buffers share the remainder:
    // source, output, block permutation, and the two
    // staging copies. Source, output, and staging store element_count uint32
    // values; block permutation stores one index per block. Round every size
    // to a multiple of both the workgroup and largest supported block width.
    const uint64_t element_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped =
        std::min({element_capacity, dispatch_capacity, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    const uint64_t rounded = capped - (capped % element_multiple);
    if (rounded < element_multiple) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_element_count, uint32_t element_multiple) {
    const std::array<uint32_t, 7> base_sizes = {
        65536U, 262144U, 1048576U, 4194304U, 8388608U, 16777216U, 33554432U,
    };
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_element_count && candidate % element_multiple == 0U) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_element_count) {
        sizes.push_back(max_element_count);
    }
    return sizes;
}

std::vector<AccessVariantDescriptor>
build_variant_descriptors(const std::array<uint32_t, kBlockSubgroupCounts.size()>& block_widths,
                          uint32_t subgroup_size) {
    static constexpr std::array<std::pair<uint32_t, const char*>, 2> kBlockModes = {
        {{kBlocksSequential, "blocks_seq"}, {kBlocksShuffled, "blocks_shuffled"}},
    };
    static constexpr std::array<std::pair<uint32_t, const char*>, 2> kWithinModes = {
        {{kWithinOrdered, "within_ordered"}, {kWithinReversed, "within_reversed"}},
    };

    std::vector<AccessVariantDescriptor> descriptors;
    descriptors.reserve(block_widths.size() * kBlockModes.size() * kWithinModes.size());
    for (const uint32_t block_width : block_widths) {
        for (const auto& [block_mode, block_token] : kBlockModes) {
            for (const auto& [within_mode, within_token] : kWithinModes) {
                descriptors.push_back(AccessVariantDescriptor{
                    .variant_name = "w" + std::to_string(block_width) + "_" + block_token + "_" + within_token,
                    .block_width = block_width,
                    .subgroup_size = subgroup_size,
                    .within_mode = within_mode,
                    .block_mode = block_mode,
                });
            }
        }
    }
    return descriptors;
}

bool has_evicted_width(uint32_t block_width, const std::array<uint32_t, kBlockSubgroupCounts.size()>& block_widths) {
    return block_width == block_widths.front() || block_width == block_widths.back();
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint64_t compute_estimated_request_bytes(uint32_t element_count) {
    // Every invocation loads one source word and one block-order index, then
    // stores one output word. Repeated index loads count separately even when
    // the implementation coalesces/broadcasts or caches those requests.
    const uint64_t source_bytes = static_cast<uint64_t>(element_count) * sizeof(uint32_t);
    const uint64_t output_bytes = source_bytes;
    const uint64_t perm_bytes = source_bytes;
    return source_bytes + output_bytes + perm_bytes;
}

std::vector<uint32_t> build_reference_outputs(const uint32_t* source_values, const uint32_t* block_perm,
                                              uint32_t element_count, uint32_t block_width, uint32_t within_mode) {
    const uint32_t bits = block_bit_count(block_width);

    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t id = 0U; id < element_count; ++id) {
        const uint32_t block_index = id / block_width;
        const uint32_t lane_in_block = id - (block_index * block_width);
        const uint32_t source_block = block_perm[block_index];
        const uint32_t lane_out = within_mode == kWithinReversed ? reversed_lane(lane_in_block, bits) : lane_in_block;
        const uint32_t source_index = (source_block * block_width) + lane_out;
        reference[id] = transform_value(source_values[source_index], source_index);
    }
    return reference;
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_element_count, uint32_t min_block_width,
                             BufferResources& out_resources) {
    // Device-local placement: source, output, and block permutation buffers are
    // read/written inside the timed dispatch and must live in device memory.
    // TRANSFER_SRC is required for the output readback copy and TRANSFER_DST for
    // the staged uploads.
    const VkBufferUsageFlags device_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                device_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.source_buffer)) {
        std::cerr << "Failed to create block-reorder source buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                device_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create block-reorder output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }
    const VkDeviceSize block_perm_span = compute_span_bytes(max_element_count / min_block_width);
    if (!create_buffer_resource(context.physical_device(), context.device(), block_perm_span, device_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.block_perm_buffer)) {
        std::cerr << "Failed to create block-reorder block-permutation buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), kEvictionBufferBytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                out_resources.eviction_buffer)) {
        std::cerr << "Failed to create block-reorder eviction buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.block_perm_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_in)) {
        std::cerr << "Failed to create block-reorder upload staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.eviction_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_perm_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_out)) {
        std::cerr << "Failed to create block-reorder readback staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.staging_in);
        destroy_buffer_resource(context.device(), out_resources.eviction_buffer);
        destroy_buffer_resource(context.device(), out_resources.block_perm_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.staging_in, "block-reorder upload staging",
                           out_resources.staging_in_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.staging_out, "block-reorder readback staging",
                           out_resources.staging_out_mapped_ptr)) {
        return false;
    }
    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.staging_out_mapped_ptr != nullptr && resources.staging_out.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_out.memory);
        resources.staging_out_mapped_ptr = nullptr;
    }
    if (resources.staging_in_mapped_ptr != nullptr && resources.staging_in.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_in.memory);
        resources.staging_in_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.staging_out);
    destroy_buffer_resource(context.device(), resources.staging_in);
    destroy_buffer_resource(context.device(), resources.eviction_buffer);
    destroy_buffer_resource(context.device(), resources.block_perm_buffer);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.source_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo source_info{
        .buffer = buffers.source_buffer.buffer, .offset = 0U, .range = buffers.source_buffer.size};
    const VkDescriptorBufferInfo perm_info{
        .buffer = buffers.block_perm_buffer.buffer, .offset = 0U, .range = buffers.block_perm_buffer.size};
    const VkDescriptorBufferInfo output_info{
        .buffer = buffers.output_buffer.buffer, .offset = 0U, .range = buffers.output_buffer.size};
    // Main shader binding contract:
    //   0 -> source dataset (the separate sweep never reads this buffer)
    //   1 -> identity or shuffled block-order permutation, loaded by every invocation
    //   2 -> per-invocation transformed outputs
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {
            {.binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = source_info},
            {.binding = 1U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = perm_info},
            {.binding = 2U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = output_info},
        });
}

bool create_main_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                    const BufferResources& buffers, MainPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load block-reorder main shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {.binding = 0U,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1U,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = nullptr},
        {.binding = 1U,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1U,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = nullptr},
        {.binding = 2U,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1U,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create block-reorder descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create block-reorder descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate block-reorder descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0U, .size = static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create block-reorder pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create block-reorder compute pipeline.\n";
        return false;
    }
    return true;
}

bool create_sweep_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                     const BufferResources& buffers, SweepPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load block-reorder sweep shader module: " << shader_path << "\n";
        return false;
    }

    // The eviction sweep uses its own small descriptor set: it only touches the
    // dedicated eviction buffer and the output buffer as a discard sink; it
    // never reads the source dataset.
    const std::vector<VkDescriptorSetLayoutBinding> sweep_bindings = {
        {.binding = 0U,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1U,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = nullptr},
        {.binding = 1U,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1U,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), sweep_bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create block-reorder sweep descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> sweep_pool_sizes = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), sweep_pool_sizes, 1U,
                                                    out_resources.descriptor_pool)) {
        std::cerr << "Failed to create block-reorder sweep descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate block-reorder sweep descriptor set.\n";
        return false;
    }

    const VkDescriptorBufferInfo eviction_info{
        .buffer = buffers.eviction_buffer.buffer, .offset = 0U, .range = buffers.eviction_buffer.size};
    const VkDescriptorBufferInfo output_info{
        .buffer = buffers.output_buffer.buffer, .offset = 0U, .range = buffers.output_buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out_resources.descriptor_set,
        {
            {.binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = eviction_info},
            {.binding = 1U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = output_info},
        });

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .offset = 0U,
         .size = static_cast<uint32_t>(sizeof(SweepPushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create block-reorder sweep pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create block-reorder sweep compute pipeline.\n";
        return false;
    }
    return true;
}

void destroy_main_pipeline_resources(VulkanContext& context, MainPipelineResources& resources) {
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

void destroy_sweep_pipeline_resources(VulkanContext& context, SweepPipelineResources& resources) {
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

double run_source_upload_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.source_buffer.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                buffers.source_buffer.buffer, bytes);
    });
}

double run_block_perm_upload_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.block_perm_buffer.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(
            command_buffer, buffers.block_perm_buffer.buffer, bytes);
    });
}

double run_output_poison_stage(VulkanContext& context, const BufferResources& buffers,
                               const std::vector<uint32_t>& reference) {
    auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
    std::ranges::transform(reference, staging_values, [](uint32_t value) {
        return ~value;
    });
    const VkDeviceSize bytes = compute_span_bytes(static_cast<uint32_t>(reference.size()));
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferMemoryBarrier write_barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                                  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                                  .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                  .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                  .buffer = buffers.output_buffer.buffer,
                                                  .offset = 0U,
                                                  .size = bytes};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0U,
                             0U, nullptr, 1U, &write_barrier, 0U, nullptr);
        const VkBufferCopy copy{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.output_buffer.buffer, 1U, &copy);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                buffers.output_buffer.buffer, bytes);
    });
}

double run_dispatch_stage(VulkanContext& context, const MainPipelineResources& pipeline_resources,
                          uint32_t element_count, uint32_t block_width, uint32_t within_mode,
                          uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants constants{
        .element_count = element_count,
        .block_width = block_width,
        .within_mode = within_mode,
        .block_bits = block_bit_count(block_width),
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

double run_sweep_stage(VulkanContext& context, const SweepPipelineResources& sweep_pipeline,
                       uint32_t output_element_count, uint32_t max_dispatch_groups_x) {
    // Each invocation owns one output word and streams its disjoint sequence of
    // eviction-buffer chunks. No two invocations write the same discard sink.
    const uint64_t eviction_word_count = static_cast<uint64_t>(kEvictionBufferBytes) / sizeof(uint32_t);
    const auto chunk_count = static_cast<uint32_t>(eviction_word_count / kSweepChunkElements);
    const uint32_t invocation_count = std::min(chunk_count, output_element_count);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(invocation_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const SweepPushConstants constants{
        .eviction_element_count = static_cast<uint32_t>(eviction_word_count),
        .output_element_count = output_element_count,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sweep_pipeline.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sweep_pipeline.pipeline_layout, 0U, 1U,
                                &sweep_pipeline.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, sweep_pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

double run_readback_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, buffers.output_buffer.buffer,
                                                                          bytes);
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.output_buffer.buffer, buffers.staging_out.buffer, 1U, &copy_region);

        const VkBufferMemoryBarrier host_read_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffers.staging_out.buffer,
            .offset = 0U,
            .size = bytes,
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0U, 0U,
                             nullptr, 1U, &host_read_barrier, 0U, nullptr);
    });
}

void record_case_notes(std::string& notes, const AccessVariantDescriptor& descriptor, const char* residency,
                       uint32_t element_count, uint64_t request_bytes, uint32_t block_perm_bytes,
                       uint32_t block_perm_count, uint32_t block_perm_seed, uint64_t checksum,
                       uint32_t source_span_bytes, uint64_t scratch_size_bytes, const BufferResources& buffers,
                       bool correctness_pass, bool dispatch_ok) {
    const uint64_t live_payload_bytes = buffers.source_buffer.size + buffers.output_buffer.size +
                                        buffers.block_perm_buffer.size + buffers.eviction_buffer.size +
                                        buffers.staging_in.size + buffers.staging_out.size;
    const uint64_t live_allocation_bytes =
        buffers.source_buffer.allocation_size + buffers.output_buffer.allocation_size +
        buffers.block_perm_buffer.allocation_size + buffers.eviction_buffer.allocation_size +
        buffers.staging_in.allocation_size + buffers.staging_out.allocation_size;
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "block_width=" + std::to_string(descriptor.block_width));
    append_note(notes, "subgroup_size=" + std::to_string(descriptor.subgroup_size));
    append_note(notes, "block_subgroup_count=" + std::to_string(descriptor.block_width / descriptor.subgroup_size));
    append_note(notes, "logical_lane_order=dense_active_subgroup_lane_order");
    append_note(notes, "mapping_shared_memory_bytes=2048");
    append_note(notes, "within_index_arithmetic=common_reversal_and_mask_select");
    append_note(notes, "block_bit_count=" + std::to_string(block_bit_count(descriptor.block_width)));
    append_note(notes,
                "within_mode=" + std::string(descriptor.within_mode == kWithinReversed ? "bit_reversal" : "ordered"));
    append_note(notes, "block_mode=" + std::string(descriptor.block_mode == kBlocksShuffled ? "shuffled_by_block_perm"
                                                                                            : "sequential"));
    append_note(notes, "residency=" + std::string(residency));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "scratch_size_bytes=" + std::to_string(scratch_size_bytes));
    append_note(notes, "scratch_budget_scope=live_buffer_payload");
    append_note(notes, "live_buffer_count=6");
    append_note(notes, "live_payload_bytes=" + std::to_string(live_payload_bytes));
    append_note(notes, "live_allocation_bytes=" + std::to_string(live_allocation_bytes));
    append_note(notes, "workload_buffer_span_bytes=" + std::to_string(buffers.source_buffer.size));
    append_note(notes, "live_block_perm_span_bytes=" + std::to_string(buffers.block_perm_buffer.size));
    append_note(notes, "eviction_buffer_bytes=" + std::to_string(buffers.eviction_buffer.size));
    append_note(notes, "buffer_span_bytes=" + std::to_string(source_span_bytes));
    append_note(notes, "source_span_bytes=" + std::to_string(source_span_bytes));
    append_note(notes, "block_perm_span_bytes=" + std::to_string(block_perm_bytes));
    append_note(notes, "block_perm_count=" + std::to_string(block_perm_count));
    append_note(notes, "block_perm_seed=" + std::to_string(block_perm_seed));
    append_note(notes, "source_pattern_seed=" + std::to_string(kSourcePatternSeed));
    append_note(notes, "visit_contract=bijective_single_source_pass");
    append_note(notes, "estimated_request_bytes=" + std::to_string(request_bytes));
    append_note(notes, "gbps_mode=estimated_shader_request_bytes");
    append_note(notes, "request_bytes_per_element=12");
    append_note(notes, "block_index_requests=one_per_invocation_including_identity_table");
    append_note(notes, "physical_memory_transactions_measured=false");
    append_note(notes, "timing_scope=block_reorder_dispatch");
    append_note(notes, "buffer_placement=device_local_staged");
    append_note(notes, "oracle=reference_complement_before_every_dispatch");
    append_note(notes, "output_preparation=transfer_after_residency_setup_outside_dispatch_timing");
    append_note(notes, "output_preparation_cache_effect=output_copy_may_perturb_cache_state");
    append_note(notes, "warm_residency_setup=unmeasured_main_dispatch_before_output_poison");
    if (std::strcmp(residency, kResidencyEvicted) == 0) {
        append_note(notes, "eviction_pass=dedicated_eviction_buffer_stream_before_dispatch");
        append_note(notes, "eviction_sink=one_writer_per_output_word");
        append_note(notes, "eviction_honesty=sweep_warms_the_eviction_buffer_tail_and_leaves_prefetch_state");
    }
    append_note(notes, "result_checksum=" + std::to_string(checksum));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const MainPipelineResources& main_pipeline,
              const SweepPipelineResources* sweep_pipeline, const BufferResources& buffers,
              const uint32_t* source_values, const uint32_t* block_perm, uint32_t element_count,
              uint32_t max_dispatch_groups_x, const AccessVariantDescriptor& descriptor, const char* residency,
              uint32_t block_perm_bytes, uint32_t block_perm_count, uint32_t block_perm_seed,
              uint64_t scratch_size_bytes, BlockReorderFactorialExperimentOutput& output, bool verbose_progress) {
    const bool evicted = std::strcmp(residency, kResidencyEvicted) == 0;
    if (evicted && sweep_pipeline == nullptr) {
        std::cerr << "[" << kExperimentId << "] Evicted residency requested without a sweep pipeline.\n";
        output.all_points_correct = false;
        return false;
    }

    const VkDeviceSize bytes = compute_span_bytes(element_count);
    auto* output_values = static_cast<uint32_t*>(buffers.staging_out_mapped_ptr);
    if (output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped readback pointer.\n";
        output.all_points_correct = false;
        return false;
    }

    const std::vector<uint32_t> reference_outputs = build_reference_outputs(
        source_values, block_perm, element_count, descriptor.block_width, descriptor.within_mode);
    const uint64_t request_bytes = compute_estimated_request_bytes(element_count);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    const auto prepare_iteration = [&]() {
        // Prime only warm cases. Evicted cases run their dedicated sweep instead;
        // the common output copy then guarantees that every result must be new.
        const double setup_ms = evicted
                                    ? run_sweep_stage(context, *sweep_pipeline, element_count, max_dispatch_groups_x)
                                    : run_dispatch_stage(context, main_pipeline, element_count, descriptor.block_width,
                                                         descriptor.within_mode, max_dispatch_groups_x);
        return std::isfinite(setup_ms) && std::isfinite(run_output_poison_stage(context, buffers, reference_outputs));
    };

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        if (!prepare_iteration()) {
            std::cerr << "[" << kExperimentId << "] Residency/output preparation failed during warmup.\n";
            return false;
        }
        const double dispatch_ms = run_dispatch_stage(context, main_pipeline, element_count, descriptor.block_width,
                                                      descriptor.within_mode, max_dispatch_groups_x);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        if (!prepare_iteration()) {
            std::cerr << "[" << kExperimentId << "] Residency/output preparation failed during a timed iteration.\n";
            return false;
        }
        const double dispatch_ms = run_dispatch_stage(context, main_pipeline, element_count, descriptor.block_width,
                                                      descriptor.within_mode, max_dispatch_groups_x);
        const double readback_ms = run_readback_stage(context, buffers, bytes);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool readback_ok = std::isfinite(readback_ms);
        const bool correctness_pass =
            dispatch_ok && readback_ok && validate_outputs(output_values, reference_outputs, element_count);
        const uint64_t checksum = compute_output_checksum(output_values, element_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, residency, element_count, request_bytes, block_perm_bytes,
                          block_perm_count, block_perm_seed, checksum, static_cast<uint32_t>(bytes), scratch_size_bytes,
                          buffers, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = 1U,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, 1U, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(request_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", checksum=" << checksum
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

} // namespace

BlockReorderFactorialExperimentOutput
run_block_reorder_factorial_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const BlockReorderFactorialExperimentConfig& config) {
    BlockReorderFactorialExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "block-reorder factorial experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.scratch_size_bytes <= kEvictionBufferBytes) {
        std::cerr << "Block-reorder factorial requires --size greater than 64 MiB to fit its eviction buffer "
                     "and workload buffers. Use --size 128M.\n";
        output.all_points_correct = false;
        return output;
    }
    const std::size_t workload_buffer_budget =
        (config.scratch_size_bytes - static_cast<std::size_t>(kEvictionBufferBytes)) / 5U;

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "47_block_reorder_factorial.comp.spv");
    const std::string sweep_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.sweep_shader_path, "47_block_reorder_factorial_sweep.comp.spv");
    if (shader_path.empty() || sweep_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shaders for the block-reorder factorial experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceSubgroupProperties subgroup_properties{.sType =
                                                               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &subgroup_properties};
    vkGetPhysicalDeviceProperties2(context.physical_device(), &properties);
    const uint32_t subgroup_size = subgroup_properties.subgroupSize;
    if ((subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0U ||
        (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0U ||
        (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) == 0U ||
        !std::has_single_bit(subgroup_size) || subgroup_size < 2U || subgroup_size > kWorkgroupSize ||
        properties.properties.limits.maxComputeSharedMemorySize < 2048U) {
        std::cerr << "Block-reorder factorial requires compute-stage basic and ballot subgroup operations, a "
                     "power-of-two subgroup size in [2, 256], and 2048 bytes of shared memory.\n";
        output.all_points_correct = false;
        return output;
    }
    const auto block_widths = build_block_widths(subgroup_size);
    const uint32_t element_multiple = std::max(kWorkgroupSize, block_widths.back());
    const uint32_t max_dispatch_groups_x = properties.properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(
        std::min({config.max_buffer_bytes, workload_buffer_budget,
                  static_cast<std::size_t>(properties.properties.limits.maxStorageBufferRange)}),
        max_dispatch_groups_x, element_multiple);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for block-reorder factorial experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] sweep shader: " << sweep_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_element_count
                  << ", subgroup_size=" << subgroup_size << ", element_multiple=" << element_multiple
                  << ", span_bytes_per_buffer=" << compute_span_bytes(max_element_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    std::vector<uint32_t> source_values(max_element_count);
    for (uint32_t index = 0U; index < max_element_count; ++index) {
        source_values[index] = source_pattern_value(index);
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_element_count, block_widths.front(), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    MainPipelineResources main_pipeline{};
    if (!create_main_pipeline_resources(context, shader_path, buffers, main_pipeline)) {
        destroy_main_pipeline_resources(context, main_pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    SweepPipelineResources sweep_pipeline{};
    if (!create_sweep_pipeline_resources(context, sweep_shader_path, buffers, sweep_pipeline)) {
        destroy_sweep_pipeline_resources(context, sweep_pipeline);
        destroy_main_pipeline_resources(context, main_pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<AccessVariantDescriptor> descriptors = build_variant_descriptors(block_widths, subgroup_size);
    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_element_count, element_multiple);
    for (uint32_t element_count : problem_sizes) {
        // Fill and stage the source dataset once per problem size. The source
        // value pattern is per-index, so the CPU oracle reads the same values
        // from the leading slice of the shared CPU vector.
        const VkDeviceSize bytes = compute_span_bytes(element_count);
        auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
        if (staging_values == nullptr) {
            std::cerr << "[" << kExperimentId << "] Missing mapped upload pointer.\n";
            output.all_points_correct = false;
            continue;
        }
        std::memcpy(staging_values, source_values.data(), static_cast<std::size_t>(bytes));
        const double upload_ms = run_source_upload_stage(context, buffers, bytes);
        if (!std::isfinite(upload_ms)) {
            std::cerr << "[" << kExperimentId << "] Source upload produced a non-finite timing value.\n";
            output.all_points_correct = false;
        }

        for (const uint32_t block_width : block_widths) {
            const uint32_t block_perm_count = element_count / block_width;
            const uint32_t block_perm_bytes = block_perm_count * sizeof(uint32_t);
            std::vector<uint32_t> block_perm(block_perm_count, 0U);
            fill_random_permutation(block_perm.data(), block_perm_count);
            std::vector<uint32_t> identity_perm(block_perm_count, 0U);
            for (uint32_t block = 0U; block < block_perm_count; ++block) {
                identity_perm[block] = block;
            }

            for (const auto& descriptor : descriptors) {
                if (descriptor.block_width != block_width) {
                    continue;
                }
                const auto& active_perm = descriptor.block_mode == kBlocksShuffled ? block_perm : identity_perm;
                std::memcpy(staging_values, active_perm.data(), block_perm_bytes);
                if (!std::isfinite(run_block_perm_upload_stage(context, buffers, block_perm_bytes))) {
                    std::cerr << "[" << kExperimentId << "] Block permutation upload failed.\n";
                    output.all_points_correct = false;
                    continue;
                }
                if (!run_case(context, runner, main_pipeline, nullptr, buffers, source_values.data(),
                              active_perm.data(), element_count, max_dispatch_groups_x, descriptor, kResidencyWarm,
                              block_perm_bytes, block_perm_count, kBlockPermutationSeed, config.scratch_size_bytes,
                              output, config.verbose_progress)) {
                    output.all_points_correct = false;
                }
                if (has_evicted_width(block_width, block_widths)) {
                    AccessVariantDescriptor evicted_descriptor = descriptor;
                    evicted_descriptor.variant_name += "_evicted";
                    if (!run_case(context, runner, main_pipeline, &sweep_pipeline, buffers, source_values.data(),
                                  active_perm.data(), element_count, max_dispatch_groups_x, evicted_descriptor,
                                  kResidencyEvicted, block_perm_bytes, block_perm_count, kBlockPermutationSeed,
                                  config.scratch_size_bytes, output, config.verbose_progress)) {
                        output.all_points_correct = false;
                    }
                }
            }
        }
    }

    destroy_sweep_pipeline_resources(context, sweep_pipeline);
    destroy_main_pipeline_resources(context, main_pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
