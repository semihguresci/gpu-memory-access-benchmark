#include "experiments/subgroup_stream_compaction_variants_experiment.hpp"

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

constexpr const char* kExperimentId = "32_subgroup_stream_compaction_variants";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTargetLogicalCount = 65536U;
constexpr uint32_t kOutputSentinelValue = 0xDEADBEEFU;
constexpr uint32_t kHashMul0 = 0x7FEB352DU;
constexpr uint32_t kHashMul1 = 0x846CA68BU;
constexpr uint32_t kInputXorSeed = 0xA511E9B3U;
constexpr uint32_t kInputMultiplier = 0x9E3779B9U;
constexpr uint32_t kInputAddend = 0x7F4A7C15U;
constexpr uint32_t kRatioDivisor = 100U;
constexpr std::array<uint32_t, 5> kValidRatioPercents = {5U, 25U, 50U, 75U, 95U};

enum class VariantKind : uint32_t {
    SharedAtomicBlock,
    SubgroupBallot,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    const char* shader_filename;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::SharedAtomicBlock, "shared_atomic_block", "32_subgroup_compaction_shared_atomic.comp.spv"},
    {VariantKind::SubgroupBallot, "subgroup_ballot", "32_subgroup_compaction_subgroup_ballot.comp.spv"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource output_buffer{};
    BufferResource block_counts_buffer{};
    void* input_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
    void* block_counts_mapped_ptr = nullptr;
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
    uint32_t logical_count = 0U;
    uint32_t valid_ratio_percent = 0U;
    uint32_t pattern_seed = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 3U));

struct SubgroupSupportInfo {
    uint32_t subgroup_size = 0U;
    bool compute_stage_supported = false;
    bool ballot_supported = false;
};

struct ReferenceData {
    std::vector<uint32_t> compacted_values;
    std::vector<uint32_t> block_counts;
    uint32_t total_valid_count = 0U;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

SubgroupSupportInfo query_subgroup_support(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties);

    return SubgroupSupportInfo{
        .subgroup_size = subgroup_properties.subgroupSize,
        .compute_stage_supported = (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0U,
        .ballot_supported = (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0U,
    };
}

uint32_t hash_index(uint32_t index_value, uint32_t pattern_seed) {
    uint32_t value = index_value ^ pattern_seed;
    value ^= value >> 16U;
    value *= kHashMul0;
    value ^= value >> 15U;
    value *= kHashMul1;
    value ^= value >> 16U;
    return value;
}

uint32_t generate_input_value(uint32_t index_value, uint32_t pattern_seed) {
    return hash_index(index_value, pattern_seed) ^ ((index_value * kInputMultiplier) + kInputAddend) ^ kInputXorSeed;
}

bool is_index_valid(uint32_t index_value, uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    return (hash_index(index_value, pattern_seed) % kRatioDivisor) < valid_ratio_percent;
}

void fill_input_values(uint32_t* values, uint32_t logical_count, uint32_t pattern_seed) {
    for (uint32_t index = 0U; index < logical_count; ++index) {
        values[index] = generate_input_value(index, pattern_seed);
    }
}

void fill_output_values(uint32_t* values, uint32_t logical_count) {
    std::fill_n(values, logical_count, kOutputSentinelValue);
}

void fill_block_counts(uint32_t* values, uint32_t block_count) {
    std::fill_n(values, block_count, 0U);
}

bool validate_input_values(const uint32_t* values, uint32_t logical_count, uint32_t pattern_seed) {
    for (uint32_t index = 0U; index < logical_count; ++index) {
        if (values[index] != generate_input_value(index, pattern_seed)) {
            return false;
        }
    }
    return true;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t max_from_buffer = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t max_from_dispatch = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count =
        std::min({max_from_buffer, max_from_dispatch, static_cast<uint64_t>(kTargetLogicalCount)});
    if (effective_count < kWorkgroupSize) {
        return 0U;
    }

    const uint64_t rounded = effective_count - (effective_count % kWorkgroupSize);
    return rounded >= kWorkgroupSize ? static_cast<uint32_t>(rounded) : 0U;
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, uint32_t logical_count, BufferResources& out_resources) {
    const uint32_t block_count = logical_count / kWorkgroupSize;
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(logical_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_buffer)) {
        std::cerr << "Failed to create subgroup compaction input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(logical_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create subgroup compaction output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(block_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.block_counts_buffer)) {
        std::cerr << "Failed to create subgroup compaction block-count buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "subgroup compaction input buffer",
                           out_resources.input_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_buffer, "subgroup compaction output buffer",
                           out_resources.output_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.block_counts_buffer, "subgroup compaction block-count buffer",
                           out_resources.block_counts_mapped_ptr)) {
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.block_counts_mapped_ptr != nullptr && resources.block_counts_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.block_counts_buffer.memory);
        resources.block_counts_mapped_ptr = nullptr;
    }
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.block_counts_buffer);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    const VkDescriptorBufferInfo counts_info{buffers.block_counts_buffer.buffer, 0U, buffers.block_counts_buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                          {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, counts_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load subgroup compaction shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create subgroup compaction descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create subgroup compaction descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate subgroup compaction descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create subgroup compaction pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create subgroup compaction compute pipeline.\n";
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

ReferenceData build_reference_data(const uint32_t* input_values, uint32_t logical_count, uint32_t valid_ratio_percent,
                                   uint32_t pattern_seed) {
    const uint32_t block_count = logical_count / kWorkgroupSize;
    ReferenceData reference{
        .compacted_values = std::vector<uint32_t>(logical_count, kOutputSentinelValue),
        .block_counts = std::vector<uint32_t>(block_count, 0U),
        .total_valid_count = 0U,
    };

    for (uint32_t block_index = 0U; block_index < block_count; ++block_index) {
        const uint32_t block_base = block_index * kWorkgroupSize;
        uint32_t valid_count = 0U;
        for (uint32_t local_index = 0U; local_index < kWorkgroupSize; ++local_index) {
            const uint32_t global_index = block_base + local_index;
            if (is_index_valid(global_index, valid_ratio_percent, pattern_seed)) {
                reference.compacted_values[block_base + valid_count] = input_values[global_index];
                ++valid_count;
            }
        }
        reference.block_counts[block_index] = valid_count;
        reference.total_valid_count += valid_count;
    }

    return reference;
}

bool validate_outputs(const uint32_t* output_values, const uint32_t* block_counts_values,
                      const ReferenceData& reference, uint32_t logical_count, bool stable_ordering_required) {
    const uint32_t block_count = logical_count / kWorkgroupSize;
    for (uint32_t block_index = 0U; block_index < block_count; ++block_index) {
        const uint32_t block_base = block_index * kWorkgroupSize;
        const uint32_t expected_count = reference.block_counts[block_index];
        if (block_counts_values[block_index] != expected_count) {
            return false;
        }

        if (stable_ordering_required) {
            for (uint32_t offset = 0U; offset < expected_count; ++offset) {
                if (output_values[block_base + offset] != reference.compacted_values[block_base + offset]) {
                    return false;
                }
            }
        } else {
            std::vector<uint32_t> actual_values;
            actual_values.reserve(expected_count);
            std::vector<uint32_t> expected_values;
            expected_values.reserve(expected_count);
            for (uint32_t offset = 0U; offset < expected_count; ++offset) {
                actual_values.push_back(output_values[block_base + offset]);
                expected_values.push_back(reference.compacted_values[block_base + offset]);
            }
            std::sort(actual_values.begin(), actual_values.end());
            std::sort(expected_values.begin(), expected_values.end());
            if (actual_values != expected_values) {
                return false;
            }
        }

        for (uint32_t offset = expected_count; offset < kWorkgroupSize; ++offset) {
            if (output_values[block_base + offset] != kOutputSentinelValue) {
                return false;
            }
        }
    }

    return true;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t logical_count,
                    uint32_t valid_ratio_percent, uint32_t pattern_seed) {
    const uint32_t group_count_x = logical_count / kWorkgroupSize;
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{logical_count, valid_ratio_percent, pattern_seed};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t logical_count,
                       uint32_t valid_ratio_percent, uint32_t subgroup_size, const ReferenceData& reference,
                       bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("compaction_strategy=") + descriptor.variant_name);
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "valid_ratio_percent=" + std::to_string(valid_ratio_percent));
    append_note(notes, "valid_count=" + std::to_string(reference.total_valid_count));
    append_note(notes, "block_count=" + std::to_string(logical_count / kWorkgroupSize));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes,
                "stable_ordering=" + std::string(descriptor.kind == VariantKind::SubgroupBallot ? "true" : "false"));
    if (descriptor.kind == VariantKind::SubgroupBallot) {
        append_note(notes, "subgroup_size=" + std::to_string(subgroup_size));
    }
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, const VariantDescriptor& descriptor, uint32_t logical_count,
              uint32_t valid_ratio_percent, uint32_t subgroup_size,
              const SubgroupStreamCompactionVariantsExperimentConfig& config,
              SubgroupStreamCompactionVariantsExperimentOutput& output) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* block_counts_values = static_cast<uint32_t*>(buffers.block_counts_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || block_counts_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffer pointer for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    const ReferenceData reference =
        build_reference_data(input_values, logical_count, valid_ratio_percent, config.pattern_seed);
    const bool stable_ordering_required = descriptor.kind == VariantKind::SubgroupBallot;
    const uint64_t payload_bytes = (static_cast<uint64_t>(logical_count) * sizeof(uint32_t)) +
                                   (static_cast<uint64_t>(reference.total_valid_count) * sizeof(uint32_t)) +
                                   (static_cast<uint64_t>(logical_count / kWorkgroupSize) * sizeof(uint32_t));
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, logical_count);
        fill_block_counts(block_counts_values, logical_count / kWorkgroupSize);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, logical_count, valid_ratio_percent, config.pattern_seed);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_input_values(input_values, logical_count, config.pattern_seed) &&
            validate_outputs(output_values, block_counts_values, reference, logical_count, stable_ordering_required);
        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", valid_ratio_percent=" << valid_ratio_percent
                      << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, logical_count);
        fill_block_counts(block_counts_values, logical_count / kWorkgroupSize);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, logical_count, valid_ratio_percent, config.pattern_seed);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && validate_input_values(input_values, logical_count, config.pattern_seed) &&
            validate_outputs(output_values, block_counts_values, reference, logical_count, stable_ordering_required);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, logical_count, valid_ratio_percent, subgroup_size, reference,
                          correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = std::string(descriptor.variant_name) + "_ratio_" + std::to_string(valid_ratio_percent),
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_ratio_" + std::to_string(valid_ratio_percent) +
            "_elements_" + std::to_string(logical_count),
        dispatch_samples));
    return true;
}

} // namespace

SubgroupStreamCompactionVariantsExperimentOutput
run_subgroup_stream_compaction_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                   const SubgroupStreamCompactionVariantsExperimentConfig& config) {
    SubgroupStreamCompactionVariantsExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "subgroup stream compaction variants experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const SubgroupSupportInfo subgroup_support = query_subgroup_support(context.physical_device());
    if (!subgroup_support.compute_stage_supported || !subgroup_support.ballot_supported ||
        subgroup_support.subgroup_size == 0U) {
        std::cerr
            << "Selected GPU does not support compute-stage subgroup ballot operations required for Experiment 32.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        const std::string user_path = kVariantDescriptors[index].kind == VariantKind::SharedAtomicBlock
                                          ? config.shared_atomic_shader_path
                                          : config.subgroup_ballot_shader_path;
        shader_paths[index] =
            VulkanComputeUtils::resolve_shader_path(user_path, kVariantDescriptors[index].shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for subgroup compaction variant "
                      << kVariantDescriptors[index].variant_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for subgroup stream compaction variants experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, logical_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* block_counts_values = static_cast<uint32_t*>(buffers.block_counts_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || block_counts_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffer pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_input_values(input_values, logical_count, config.pattern_seed);
    fill_output_values(output_values, logical_count);
    fill_block_counts(block_counts_values, logical_count / kWorkgroupSize);

    std::array<PipelineResources, kVariantDescriptors.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (const uint32_t valid_ratio_percent : kValidRatioPercents) {
        for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
            if (!run_case(context, runner, buffers, pipeline_resources[index], kVariantDescriptors[index],
                          logical_count, valid_ratio_percent, subgroup_support.subgroup_size, config, output)) {
                output.all_points_correct = false;
            }
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);
    return output;
}
