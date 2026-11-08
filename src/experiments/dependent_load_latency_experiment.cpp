#include "experiments/dependent_load_latency_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "46_dependent_load_latency";
constexpr uint32_t kWorkgroupSize = 256U;
// Total dependent loads per invocation stays constant across chain variants:
// chain_count * chain_steps == kTotalWalkSteps for every enabled variant.
constexpr uint32_t kTotalWalkSteps = 64U;
// CPU seed for the Fisher-Yates permutation uploaded as the pointer table.
constexpr uint32_t kPermutationSeed = 0x94C0FFEEU;
constexpr uint32_t kMixOffsetBasis = 2166136261U;
constexpr uint32_t kMixMultiplier = 2654435761U;
// Odd stride so the per-chain start indices do not collapse into the same
// residues for power-of-two element counts.
constexpr uint32_t kChainStartStride = 0x9E3779B1U;
constexpr uint32_t kOutputSentinel = 0xDEADBEEFU;

struct ChainVariantDescriptor {
    uint32_t chain_count;
    const char* variant_name;
};

constexpr std::array<ChainVariantDescriptor, 5> kChainVariantDescriptors = {{
    {.chain_count = 1U, .variant_name = "chain_1"},
    {.chain_count = 2U, .variant_name = "chain_2"},
    {.chain_count = 4U, .variant_name = "chain_4"},
    {.chain_count = 8U, .variant_name = "chain_8"},
    {.chain_count = 16U, .variant_name = "chain_16"},
}};

struct BufferResources {
    BufferResource pointer_table{};
    BufferResource output{};
    BufferResource staging_in{};
    BufferResource staging_out{};
    void* staging_in_mapped_ptr = nullptr;
    void* staging_out_mapped_ptr = nullptr;
};

struct PipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t element_count = 0U;
    uint32_t chain_start_stride = kChainStartStride;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 2U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t mix_chain_final(uint32_t state, uint32_t value) {
    // Must stay in lockstep with mix_chain_final() in
    // 46_dependent_load_latency_chains.comp.
    return (state ^ value) * kMixMultiplier;
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

// Fisher-Yates over [0, element_count) so every value stays addressable within
// the per-problem-size range the dispatcher uses.
void fill_random_permutation(uint32_t* permutation, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        permutation[index] = index;
    }

    uint64_t prng_state = kPermutationSeed;
    for (uint32_t index = element_count; index > 1U; --index) {
        const uint32_t swap_target = next_prng_value(prng_state) % index;
        std::swap(permutation[index - 1U], permutation[swap_target]);
    }
}

// After `walk_steps` passes, walked[k] == f^walk_steps(k) where f(x) = permutation[x].
void apply_permutation_walk_steps(const uint32_t* permutation, uint32_t element_count, uint32_t walk_steps,
                                  std::vector<uint32_t>& walked) {
    walked.resize(element_count);
    std::vector<uint32_t> next(element_count);
    for (uint32_t index = 0U; index < element_count; ++index) {
        walked[index] = permutation[index];
    }
    for (uint32_t step = 1U; step < walk_steps; ++step) {
        for (uint32_t index = 0U; index < element_count; ++index) {
            next[index] = walked[permutation[index]];
        }
        walked.swap(next);
    }
}

std::vector<uint32_t> build_reference_outputs(const std::vector<uint32_t>& walked, uint32_t element_count,
                                              uint32_t chain_count) {
    std::vector<uint32_t> reference(element_count, kOutputSentinel);
    for (uint32_t index = 0U; index < element_count; ++index) {
        uint32_t state = kMixOffsetBasis;
        for (uint32_t chain = 0U; chain < chain_count; ++chain) {
            const uint32_t start = (index + (chain * kChainStartStride)) % element_count;
            state = mix_chain_final(state, walked[start]);
        }
        state = mix_chain_final(state, index);
        reference[index] = state;
    }
    return reference;
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

uint32_t determine_element_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    // Four equal buffers hold the pointer table, the output, and the two staging
    // copies. Each buffer stores `element_count` uint32 values.
    const uint64_t element_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped = std::min(element_capacity, dispatch_capacity);
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kWorkgroupSize) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_element_count) {
    const std::array<uint32_t, 6> base_sizes = {
        65536U, 262144U, 1048576U, 4194304U, 8388608U, 16777216U,
    };
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_element_count) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_element_count) {
        sizes.push_back(max_element_count);
    }
    return sizes;
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint64_t compute_round_trip_request_bytes(uint32_t element_count, uint32_t walk_total_steps) {
    // Each walk step requests one 4-byte dependent load; the per-invocation
    // result write adds one 4-byte store. This is a shader-request estimate,
    // not measured external-memory traffic.
    return static_cast<uint64_t>(element_count) * (static_cast<uint64_t>(walk_total_steps) + 1U) * sizeof(uint32_t);
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_element_count, BufferResources& out_resources) {
    // Device-local placement is deliberate: the pointer table must live in the
    // memory the kernel reads during the timed dispatch, staged there once per
    // problem size.
    const VkBufferUsageFlags device_buffer_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                device_buffer_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                out_resources.pointer_table)) {
        std::cerr << "Failed to create dependent-load pointer table buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                device_buffer_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.output)) {
        std::cerr << "Failed to create dependent-load output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.pointer_table);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_in)) {
        std::cerr << "Failed to create dependent-load upload staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output);
        destroy_buffer_resource(context.device(), out_resources.pointer_table);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_out)) {
        std::cerr << "Failed to create dependent-load readback staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.staging_in);
        destroy_buffer_resource(context.device(), out_resources.output);
        destroy_buffer_resource(context.device(), out_resources.pointer_table);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.staging_in, "dependent-load upload staging",
                           out_resources.staging_in_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.staging_out, "dependent-load readback staging",
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
    destroy_buffer_resource(context.device(), resources.output);
    destroy_buffer_resource(context.device(), resources.pointer_table);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo table_info{
        .buffer = buffers.pointer_table.buffer, .offset = 0U, .range = buffers.pointer_table.size};
    const VkDescriptorBufferInfo output_info{
        .buffer = buffers.output.buffer, .offset = 0U, .range = buffers.output.size};
    // Shader binding contract:
    //   0 -> randomized pointer table driving the dependent-load chains
    //   1 -> per-invocation chain-mix results
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {
            {.binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = table_info},
            {.binding = 1U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = output_info},
        });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               uint32_t chain_count, PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load dependent-load latency shader module: " << shader_path << "\n";
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
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create dependent-load latency descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create dependent-load latency descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate dependent-load latency descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0U, .size = static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create dependent-load latency pipeline layout.\n";
        return false;
    }

    // Compile each chain count separately so inactive scalar states disappear
    // before register allocation and all active loads are independent per step.
    const VkSpecializationMapEntry chain_count_entry{.constantID = 0U, .offset = 0U, .size = sizeof(chain_count)};
    const VkSpecializationInfo specialization_info{
        .mapEntryCount = 1U, .pMapEntries = &chain_count_entry, .dataSize = sizeof(chain_count), .pData = &chain_count};
    const VkPipelineShaderStageCreateInfo stage_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                     .pNext = nullptr,
                                                     .flags = 0U,
                                                     .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                     .module = out_resources.shader_module,
                                                     .pName = "main",
                                                     .pSpecializationInfo = &specialization_info};
    const VkComputePipelineCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                    .pNext = nullptr,
                                                    .flags = 0U,
                                                    .stage = stage_info,
                                                    .layout = out_resources.pipeline_layout,
                                                    .basePipelineHandle = VK_NULL_HANDLE,
                                                    .basePipelineIndex = -1};
    const VkResult pipeline_result = vkCreateComputePipelines(context.device(), VK_NULL_HANDLE, 1U, &pipeline_info,
                                                              nullptr, &out_resources.pipeline);
    if (pipeline_result != VK_SUCCESS) {
        std::cerr << "Failed to create dependent-load latency compute pipeline for chain_count=" << chain_count
                  << " with error code " << pipeline_result << ".\n";
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

double run_table_upload_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes,
                              const uint32_t* permutation) {
    auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
    std::memcpy(staging_values, permutation, static_cast<std::size_t>(bytes));
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.pointer_table.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                buffers.pointer_table.buffer, bytes);
    });
}

double run_sentinel_refill_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
    std::fill_n(staging_values, static_cast<std::size_t>(bytes / sizeof(uint32_t)), kOutputSentinel);
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.output.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, buffers.output.buffer,
                                                                                bytes);
    });
}

double run_dispatch_stage(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t element_count,
                          uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants constants{
        .element_count = element_count,
        .chain_start_stride = kChainStartStride,
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

double run_readback_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, buffers.output.buffer, bytes);
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.output.buffer, buffers.staging_out.buffer, 1U, &copy_region);

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

void record_case_notes(std::string& notes, uint32_t element_count, uint32_t chain_count, uint32_t chain_steps,
                       uint32_t walk_total_steps, uint64_t request_bytes, uint64_t checksum, uint32_t table_span_bytes,
                       uint64_t scratch_size_bytes, const BufferResources& buffers, bool correctness_pass,
                       bool dispatch_ok) {
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "chain_count=" + std::to_string(chain_count));
    append_note(notes, "chain_steps=" + std::to_string(chain_steps));
    append_note(notes, "walk_total_loads=" + std::to_string(walk_total_steps));
    append_note(notes, "chain_start_stride=" + std::to_string(kChainStartStride));
    append_note(notes, "permutation_seed=" + std::to_string(kPermutationSeed));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "scratch_size_bytes=" + std::to_string(scratch_size_bytes));
    append_note(notes, "live_buffer_count=4");
    append_note(notes, "buffer_span_bytes=" + std::to_string(table_span_bytes));
    append_note(notes, "active_payload_bytes=" + std::to_string(4ULL * table_span_bytes));
    append_note(notes, "resident_buffer_span_bytes=" + std::to_string(buffers.pointer_table.size));
    append_note(notes, "live_payload_bytes=" + std::to_string(buffers.pointer_table.size + buffers.output.size +
                                                              buffers.staging_in.size + buffers.staging_out.size));
    append_note(notes, "live_allocation_bytes=" +
                           std::to_string(buffers.pointer_table.allocation_size + buffers.output.allocation_size +
                                          buffers.staging_in.allocation_size + buffers.staging_out.allocation_size));
    append_note(notes, "table_bytes=" + std::to_string(table_span_bytes));
    append_note(notes, "estimated_request_bytes=" + std::to_string(request_bytes));
    append_note(notes, "gbps_mode=estimated_shader_request_bytes");
    append_note(notes, "timing_scope=dependent_chain_dispatch");
    append_note(notes, "sentinel_refill_outside_timed_dispatch=true");
    append_note(notes, "buffer_placement=device_local_staged");
    append_note(notes, "walk_annotation=each_walk_step_loads_table_at_previous_result");
    append_note(notes, "chain_issue_order=interleaved_steps");
    append_note(notes, "chain_count_source=pipeline_specialization");
    append_note(notes, "chain_state_storage=explicit_scalars");
    append_note(notes, "result_checksum=" + std::to_string(checksum));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, const uint32_t* permutation, uint32_t element_count,
              uint32_t max_dispatch_groups_x, const ChainVariantDescriptor& descriptor, uint64_t scratch_size_bytes,
              DependentLoadLatencyExperimentOutput& output, bool verbose_progress) {
    const uint32_t chain_steps = kTotalWalkSteps / descriptor.chain_count;
    if (chain_steps * descriptor.chain_count != kTotalWalkSteps) {
        std::cerr << "[" << kExperimentId << "] chain_count " << descriptor.chain_count
                  << " does not divide the walking total steps.\n";
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

    std::vector<uint32_t> walked;
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Building CPU oracle walk for variant=" << descriptor.variant_name
                  << ", element_count=" << element_count << ", chain_steps=" << chain_steps << "\n";
    }
    apply_permutation_walk_steps(permutation, element_count, chain_steps, walked);
    const std::vector<uint32_t> reference_outputs =
        build_reference_outputs(walked, element_count, descriptor.chain_count);
    walked.clear();

    const uint32_t walk_total_steps = chain_steps * descriptor.chain_count;
    const uint64_t request_bytes = compute_round_trip_request_bytes(element_count, walk_total_steps);
    const auto table_span_bytes = static_cast<uint32_t>(bytes);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        run_sentinel_refill_stage(context, buffers, bytes);
        const double dispatch_ms =
            run_dispatch_stage(context, pipeline_resources, element_count, max_dispatch_groups_x);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        run_sentinel_refill_stage(context, buffers, bytes);
        const double dispatch_ms =
            run_dispatch_stage(context, pipeline_resources, element_count, max_dispatch_groups_x);
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
        record_case_notes(notes, element_count, descriptor.chain_count, chain_steps, walk_total_steps, request_bytes,
                          checksum, table_span_bytes, scratch_size_bytes, buffers, correctness_pass, dispatch_ok);

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

DependentLoadLatencyExperimentOutput
run_dependent_load_latency_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const DependentLoadLatencyExperimentConfig& config) {
    DependentLoadLatencyExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "dependent-load latency experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "46_dependent_load_latency_chains.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for dependent-load latency experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for dependent-load latency experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_element_count
                  << ", span_bytes_per_buffer=" << compute_span_bytes(max_element_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    std::vector<uint32_t> permutation(max_element_count);

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_element_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    std::array<PipelineResources, kChainVariantDescriptors.size()> pipelines{};
    for (std::size_t variant_index = 0U; variant_index < kChainVariantDescriptors.size(); ++variant_index) {
        if (!create_pipeline_resources(context, shader_path, buffers,
                                       kChainVariantDescriptors[variant_index].chain_count, pipelines[variant_index])) {
            for (auto& pipeline : std::views::reverse(pipelines)) {
                destroy_pipeline_resources(context, pipeline);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_element_count);
    for (uint32_t element_count : problem_sizes) {
        // Regenerate the permutation for this problem size so every table value
        // stays inside [0, element_count): the kernel reads exactly this range,
        // and the CPU oracle walks the same values.
        fill_random_permutation(permutation.data(), element_count);
        // The pointer table is randomized once per problem size and staged into
        // device-local memory; every chain variant reads the same table.
        const VkDeviceSize bytes = compute_span_bytes(element_count);
        const double upload_ms = run_table_upload_stage(context, buffers, bytes, permutation.data());
        if (!std::isfinite(upload_ms)) {
            std::cerr << "[" << kExperimentId << "] Pointer table upload produced a non-finite timing value.\n";
            output.all_points_correct = false;
            continue;
        }
        for (std::size_t variant_index = 0U; variant_index < kChainVariantDescriptors.size(); ++variant_index) {
            if (!run_case(context, runner, pipelines[variant_index], buffers, permutation.data(), element_count,
                          max_dispatch_groups_x, kChainVariantDescriptors[variant_index], config.scratch_size_bytes,
                          output, config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    for (auto& pipeline : std::views::reverse(pipelines)) {
        destroy_pipeline_resources(context, pipeline);
    }
    destroy_buffer_resources(context, buffers);
    return output;
}
